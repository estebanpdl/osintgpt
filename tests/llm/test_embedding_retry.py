'''Bounded retries use fake time and real SDKs over offline HTTP transports.'''

import json

import httpx
import pytest
from openai import RateLimitError

from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.rate_ledger import QuotaWaitExpired
from osintgpt.llm.usage import UsageRecorder
from osintgpt.rate_settings import RateLimits
from tests.llm.test_embedding_pacing import real_sdk, response
from tests.llm.test_rate_ledger import Clock, ledger


def failure(status=429, *, headers=None, code='rate_limit_exceeded', message='temporary rate limit'):
    return httpx.Response(status, headers=headers, json={'error': {'code': code, 'message': message}})


def configure(embedder, **policy):
    embedder.pacer.retry_policy = RetryPolicy(**policy)
    embedder.pacer.retry_jitter = lambda low, high: high
    embedder.pacer.discover = False


def test_openai_retry_obeys_minimum_plus_jitter_and_accounts_each_attempt():
    calls = []
    def handler(request):
        calls.append((clock.now, json.loads(request.content)['input']))
        return failure(message='Please try again in 1.589s.') if len(calls) == 1 else response(calls[-1][1])
    embedder, clock = real_sdk(handler)
    configure(embedder)
    recorder = embedder.recorder = UsageRecorder()
    try:
        assert len(embedder.embed(['a', 'b', 'c'])) == 3
        assert calls == [(1000, ['a', 'b', 'c']), (1001.839, ['a', 'b', 'c'])]
        assert recorder.calls == 1 and recorder.input_tokens == 3
        with embedder.pacer.ledger.transaction() as connection:
            assert [tuple(row) for row in connection.execute('SELECT tokens FROM embedding_requests')] == [(3,), (3,)]
    finally:
        embedder.client.close()


def test_attempt_limit_keeps_original_error_and_has_no_nested_sdk_retries(caplog):
    calls = []
    def handler(request):
        calls.append(clock.now)
        return failure()
    embedder, clock = real_sdk(handler)
    configure(embedder)
    try:
        with pytest.raises(RateLimitError):
            embedder.embed(['a'])
        assert calls == [1000, 1001.25, 1003.75, 1008.75, 1017.75, 1034.75]
        assert 'stopped after 6 attempt(s)' in caplog.text
    finally:
        embedder.client.close()


def test_retry_pacing_time_counts_toward_total_budget(caplog):
    calls = []
    def handler(request):
        calls.append(clock.now)
        return failure()
    embedder, clock = real_sdk(handler)
    configure(embedder, max_elapsed=100)
    embedder.pacer.configured = RateLimits(rpm=1)
    try:
        with pytest.raises(RateLimitError):
            embedder.embed(['a'])
        assert calls == [1000, 1060]
        assert clock.now == 1060
        assert 'quota wait exceeds' in caplog.text
    finally:
        embedder.client.close()


def test_large_hint_is_not_clipped_and_shared_jobs_keep_the_cooldown(tmp_path):
    calls = []
    def handler(request):
        calls.append(request)
        return failure(headers={'Retry-After': '400'})
    embedder, clock = real_sdk(handler)
    configure(embedder)
    path = tmp_path / 'rates.sqlite'
    embedder.pacer.ledger = ledger(clock, path)
    try:
        with pytest.raises(RateLimitError):
            embedder.embed(['a'])
        assert len(calls) == 1 and clock.sleeps == []
        other = ledger(clock, path)
        with pytest.raises(QuotaWaitExpired):
            other.reserve(embedder.pacer.scope, 1, RateLimits(), deadline=1300)
        with other.transaction() as connection:
            assert connection.execute('SELECT next_at FROM embedding_quotas').fetchone()[0] == 1400.25
            assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1
    finally:
        embedder.client.close()


def test_lower_tpm_on_error_repartitions_without_losing_or_reordering_inputs():
    calls = []
    def handler(request):
        batch = json.loads(request.content)['input']
        calls.append(batch)
        if len(calls) == 1:
            return failure(headers={'x-ratelimit-limit-tokens': '2'})
        return response(batch)
    embedder, clock = real_sdk(handler)
    configure(embedder)
    try:
        assert len(embedder.embed(['a', 'b', 'c', 'd', 'e'])) == 5
        assert calls == [['a', 'b', 'c', 'd', 'e'], ['a', 'b'], ['c', 'd'], ['e']]
    finally:
        embedder.client.close()


def test_cancellation_during_wait_makes_no_retry_or_extra_reservation():
    calls = []
    def handler(request):
        calls.append(request)
        return failure()
    embedder, clock = real_sdk(handler)
    configure(embedder)
    def cancel(seconds):
        raise KeyboardInterrupt()
    embedder.pacer.ledger.sleep = cancel
    try:
        with pytest.raises(KeyboardInterrupt):
            embedder.embed(['a'])
        assert len(calls) == 1
        with embedder.pacer.ledger.transaction() as connection:
            assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1
    finally:
        embedder.client.close()


@pytest.mark.parametrize('status,code', [(401, 'invalid_api_key'), (400, 'invalid_request_error'), (429, 'insufficient_quota')])
def test_permanent_errors_stop_on_first_attempt(status, code):
    calls = []
    def handler(request):
        calls.append(request)
        return failure(status, code=code, headers={'Retry-After': '2'})
    embedder, clock = real_sdk(handler)
    configure(embedder)
    try:
        with pytest.raises(Exception):
            embedder.embed(['a'])
        assert len(calls) == 1 and clock.sleeps == []
    finally:
        embedder.client.close()


@pytest.mark.parametrize('mode', ['sink', 'invalid_vectors', 'invalid_response', 'ledger'])
def test_success_processing_failure_does_not_repeat_the_http_request(mode):
    calls = []
    def handler(request):
        calls.append(request)
        return response([] if mode == 'invalid_response' else ['a', 'b'] if mode == 'invalid_vectors' else ['a'])
    embedder, _ = real_sdk(handler)
    configure(embedder)
    def fail(*args, **kwargs):
        raise OSError('disk unavailable')
    if mode == 'ledger':
        embedder.pacer.observe = fail
    try:
        with pytest.raises((OSError, RuntimeError, ValueError)):
            embedder.embed_checkpointed(['a'], on_batch=fail if mode == 'sink' else lambda vectors: None)
        assert len(calls) == 1
    finally:
        embedder.client.close()


@pytest.mark.parametrize('first', ['server', 'connection', 'timeout'])
def test_openai_transient_server_and_transport_errors_recover(first):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            if first == 'server':
                return failure(503, headers={'Retry-After': '2'})
            cls = httpx.ConnectError if first == 'connection' else httpx.ReadTimeout
            raise cls('temporary', request=request)
        return response(['a'])
    embedder, clock = real_sdk(handler)
    configure(embedder)
    try:
        assert len(embedder.embed(['a'])) == 1
        assert len(calls) == 2 and clock.now >= 1001.25
    finally:
        embedder.client.close()


@pytest.mark.parametrize('provider', ['gemini', 'voyage'])
def test_other_paid_sdk_transports_retry_with_provider_hints(provider):
    from google import genai
    from openai import OpenAI
    from osintgpt.llm.gemini import GeminiEmbedding
    from osintgpt.llm.openai_compat import OpenAICompatEmbedding
    calls = []
    def handler(request):
        calls.append(clock.now)
        if len(calls) == 1:
            if provider == 'gemini':
                return httpx.Response(429, json={'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': [
                    {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '3s'}
                ]}})
            return failure(headers={'Retry-After': '3'})
        return httpx.Response(200, json={'embeddings': [{'values': [1.0, 0.0]}]}) if provider == 'gemini' else response(['a'])
    if provider == 'gemini':
        embedder = GeminiEmbedding('gemini-embedding-001', 'test-key')
        options = embedder.client._api_client._http_options.model_copy(update={'client_args': {'transport': httpx.MockTransport(handler)}})
        assert options.retry_options.attempts == 1 and options.timeout == 60000
        embedder.client.close()
        embedder.client = genai.Client(api_key='test-key', http_options=options)
    else:
        embedder = OpenAICompatEmbedding('voyage-3.5', 'test-key', provider='voyage', base_url='https://api.voyageai.com/v1')
        assert embedder.client.max_retries == 0 and embedder.client.timeout == 60
        embedder.client.close()
        embedder.client = OpenAI(api_key='test-key', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    clock = Clock()
    embedder.pacer.ledger = ledger(clock)
    configure(embedder)
    try:
        assert embedder.embed(['a']) == [[1.0, 0.0]]
        assert calls == [1000, 1003.25]
    finally:
        embedder.client.close()


@pytest.mark.parametrize('kwargs', [{'max_attempts': 0}, {'max_attempts': True}, {'max_elapsed': float('inf')}, {'max_elapsed': -1}, {'initial_delay': 0}])
def test_retry_policy_rejects_unbounded_or_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_backoff_caps_without_clipping_server_hints():
    policy = RetryPolicy()
    assert policy.delay(100, 0, lambda low, high: high) == 61
    assert policy.delay(100, 120, lambda low, high: high) == 121
