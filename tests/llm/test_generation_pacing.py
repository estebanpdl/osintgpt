'''Generation shares retry policy and quota algorithms, without SDK retry multiplication.'''

import httpx
import httpx2
import pytest
from anthropic import APIStatusError as AnthropicStatusError
from openai import OpenAI

from osintgpt.graph import ExtractionOptions, extract_document
from osintgpt.graph.events import GraphCancelled, observe_graph
from osintgpt.llm.embedding_errors import classify
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.generation_pacing import GenerationLimits, GenerationRequestError, GenerationRequests
from osintgpt.rate_settings import RateLimits
from tests.conftest import FAKE_KEY
from tests.llm.test_embedding_retry import failure
from tests.llm.test_generation_contract import provider_fixture
from tests.llm.test_rate_ledger import Clock, ledger


def retrying_provider(status=429, code='rate_limit_exceeded', *, always=False):
    provider, calls = provider_fixture('gemini')
    original = provider.client
    clock, requests = Clock(), []
    def handler(request):
        requests.append(clock.now)
        if len(requests) == 1 or always:
            return failure(status, code=code, headers={'retry-after': '2'})
        return original._client.send(request)
    provider.client = OpenAI(api_key=FAKE_KEY, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    controller = GenerationRequests(ledger=ledger(clock))
    controller.for_provider(provider).retry_jitter = lambda a, b: 0
    return provider, requests, clock, controller, original


def test_graph_retries_count_http_attempts_and_honor_delay():
    provider, times, clock, controller, original = retrying_provider()
    events = []
    try:
        with observe_graph(events.append):
            result = extract_document(provider, 'a.md', 'Source.', requests=controller)
        counts = result.accounting
        assert result.ok and times == [1000, 1002]
        assert counts.requests_attempted == 2 and counts.requests_failed == counts.requests_succeeded == 1
        assert counts.retries == 1 and provider.recorder.calls == 1
        assert {'request', 'retry', 'waiting', 'response_received', 'window_completed'} <= {e.kind for e in events}
        with controller.ledger.transaction() as connection:
            assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 2
    finally:
        provider.client.close()
        original.close()


@pytest.mark.parametrize('status,code', [(401, 'invalid_api_key'), (400, 'bad_request'),
                                        (429, 'insufficient_quota'), (403, 'permission_denied')])
def test_permanent_errors_are_not_retried(status, code):
    provider, times, clock, controller, original = retrying_provider(status, code, always=True)
    try:
        with pytest.raises(GenerationRequestError):
            extract_document(provider, 'a.md', 'Source.', requests=controller)
        assert times == [1000] and not clock.sleeps and provider.recorder.calls == 0
    finally:
        provider.client.close()
        original.close()


def test_generation_retry_budget_disables_sdk_defaults():
    provider, times, clock, controller, original = retrying_provider(always=True)
    controller.for_provider(provider).retry_policy = RetryPolicy(max_attempts=2)
    try:
        with pytest.raises(GenerationRequestError):
            extract_document(provider, 'a.md', 'Source.', requests=controller)
        assert times == [1000, 1002]
    finally:
        provider.client.close()
        original.close()


def test_cancellation_during_quota_wait_sends_no_second_attempt():
    provider, times, clock, controller, original = retrying_provider()
    try:
        with observe_graph(should_cancel=lambda: clock.now > 1000):
            with pytest.raises(GraphCancelled):
                extract_document(provider, 'a.md', 'Source.', requests=controller)
        assert times == [1000] and clock.now == 1001
        with controller.ledger.transaction() as connection:
            assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1
    finally:
        provider.client.close()
        original.close()


@pytest.mark.parametrize('limit', ['record', 'time'])
def test_transport_retry_limits_remain_independent_of_retired_record_budget(limit):
    provider, times, clock, controller, original = retrying_provider(always=True)
    with pytest.warns(DeprecationWarning, match='retired'):
        options = ExtractionOptions(max_calls=1)
    if limit == 'time':
        controller.for_provider(provider).retry_policy = RetryPolicy(max_elapsed=1)
    try:
        with pytest.raises(GenerationRequestError):
            extract_document(provider, 'a.md', 'Source.', requests=controller, options=options)
        expected = 1 if limit == 'time' else controller.for_provider(provider).retry_policy.max_attempts
        assert len(times) == expected
        assert bool(clock.sleeps) == (limit == 'record')
        with controller.ledger.transaction() as connection:
            assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == expected
    finally:
        provider.client.close()
        original.close()


def test_generation_ledger_and_model_scopes_are_separate_from_embeddings(tmp_path):
    provider, _ = provider_fixture('gemini')
    clock = Clock()
    path = tmp_path / 'generation-rates.sqlite'
    controller = GenerationRequests(GenerationLimits(RateLimits(rpm=2)), ledger=ledger(clock, path))
    try:
        extract_document(provider, 'a.md', 'Source.', requests=controller)
        controller = GenerationRequests(GenerationLimits(RateLimits(rpm=2)), ledger=ledger(clock, path))
        extract_document(provider, 'b.md', 'Source.', requests=controller)
        assert clock.now == 1030 and path.is_file()
        assert not (tmp_path / 'embedding-rates.sqlite').exists()
    finally:
        provider.client.close()


def test_anthropic_overload_is_transient_but_billing_is_terminal():
    request = httpx2.Request('POST', 'https://offline.invalid')
    response = httpx2.Response(529, request=request, headers={'retry-after': '7'})
    error = AnthropicStatusError('Overloaded', response=response, body={'type': 'overloaded_error'})
    assert classify(error, 'anthropic', 0, generation=True).delay == 7
    response = httpx2.Response(400, request=request)
    error = AnthropicStatusError('Billing', response=response, body={'type': 'invalid_request_error'})
    assert not classify(error, 'anthropic', 0, generation=True).retryable
