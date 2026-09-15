'''Token batching and real SDK request paths driven by offline transports.'''

import json

import httpx
import pytest
from openai import OpenAI, RateLimitError

from osintgpt.config import Settings
from osintgpt.ingestion import FieldMapping
from osintgpt.ingestion.recipes import recipe_for
from osintgpt.llm import build_embedding_provider
from osintgpt.llm.embedding_pacing import EmbeddingPacer, quota_scope, token_count
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.openai_compat import OpenAICompatEmbedding
from osintgpt.llm.rate_ledger import TokenBudgetExceeded
from osintgpt.rate_settings import RateLimits
from tests.llm.test_rate_ledger import Clock, ledger


def pacer(provider='openai', model='text-embedding-3-small', **limits):
    clock = Clock()
    return EmbeddingPacer(provider, model, 'https://offline.invalid', 'test-key', RateLimits(**limits), ledger=ledger(clock))


def test_token_budget_splits_batches_and_preserves_every_input_in_order():
    pacing = pacer(tpm=1000, batch_tokens=4)
    inputs = ['a'] * 9
    batches = list(pacing.batches(inputs, 100))
    assert [len(batch) for batch, _ in batches] == [4, 4, 1]
    assert [item for batch, _ in batches for item in batch] == inputs


@pytest.mark.parametrize('provider, maximum', [('openai', 2048), ('voyage', 1000), ('gemini', 100)])
def test_provider_item_ceiling_cannot_be_exceeded(provider, maximum):
    pacing = pacer(provider)
    batches = list(pacing.batches(['a'] * (maximum + 1), maximum + 100))
    assert [len(batch) for batch, _ in batches] == [maximum, 1]


def test_exact_openai_count_accepts_literal_special_tokens():
    assert token_count('openai', 'text-embedding-3-small', 'hello world') == 2
    assert token_count('openai', 'text-embedding-3-small', '<|endoftext|>') > 1


@pytest.mark.parametrize('provider', ['voyage', 'gemini'])
def test_estimate_handles_non_latin_text_and_includes_instruction(provider):
    text = '消息 🔎'
    assert token_count(provider, 'a-model', text) >= len(text.encode('utf-8'))
    pacing = pacer(provider, batch_tokens=80)
    batches = list(pacing.batches([text] * 3, 100, render=lambda value: 'title: none | text: ' + value))
    assert all(len(batch) == 1 for batch, _ in batches)


def test_one_oversized_input_fails_before_any_request_without_truncation():
    pacing = pacer(batch_tokens=1)
    with pytest.raises(TokenBudgetExceeded, match='Reduce input size'):
        list(pacing.batches(['hello world'], 100))
    with pacing.ledger.transaction() as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 0


def test_quota_scopes_separate_providers_and_group_explicit_shared_accounts():
    first = quota_scope('openai', 'model-a', 'https://endpoint/', 'secret-a')
    assert first != quota_scope('openai', 'model-a', 'https://endpoint', 'secret-b')
    shared = quota_scope('openai', 'model-a', 'https://endpoint/', 'secret-a', 'organization')
    assert shared == quota_scope('openai', 'model-b', 'https://endpoint', 'secret-b', 'organization')
    assert shared != quota_scope('voyage', 'model-a', 'https://endpoint', 'secret-a', 'organization')
    assert 'secret' not in shared and 'organization' not in shared


def test_only_paid_providers_receive_pacing(tmp_path):
    settings = Settings(openai_api_key='test', voyage_api_key='test', gemini_api_key='test', embedding_rate_state_path=str(tmp_path / 'rates.sqlite'))
    for name, model in [('openai', 'text-embedding-3-small'), ('voyage', 'voyage-3.5'), ('gemini', 'gemini-embedding-001')]:
        embedder = build_embedding_provider(name, settings, model=model)
        assert embedder.pacer.provider == name
    local = build_embedding_provider('ollama', settings, model='nomic-embed-text')
    assert local.pacer is None


def test_rate_changes_do_not_change_index_recipe(tmp_path):
    base = Settings(openai_api_key='test', embedding_rate_state_path=str(tmp_path / 'rates.sqlite'))
    first = build_embedding_provider('openai', base)
    second = build_embedding_provider('openai', base.with_overrides(openai_embedding_tpm=2_000_000, openai_embedding_rpm=5000, openai_embedding_batch_tokens=20000, openai_embedding_quota_scope='shared'))
    assert recipe_for(FieldMapping(), first) == recipe_for(FieldMapping(), second)


def real_sdk(handler):
    embedder = OpenAICompatEmbedding('text-embedding-3-small', 'test-key', provider='openai')
    embedder.client.close()
    embedder.client = OpenAI(api_key='test-key', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    clock = Clock()
    embedder.pacer.ledger = ledger(clock)
    return embedder, clock


def response(batch, headers=None):
    return httpx.Response(200, headers=headers, json={
        'model': 'text-embedding-3-small', 'object': 'list',
        'usage': {'prompt_tokens': len(batch), 'total_tokens': len(batch)},
        'data': [{'object': 'embedding', 'index': i, 'embedding': [1.0, 0.0]} for i in range(len(batch))]
    })


def test_openai_bootstrap_learns_headers_and_resizes_later_batches():
    calls = []
    def handler(request):
        batch = json.loads(request.content)['input']
        calls.append(batch)
        return response(batch, {'x-ratelimit-limit-requests': '100', 'x-ratelimit-limit-tokens': '4'})
    embedder, clock = real_sdk(handler)
    try:
        assert len(embedder.embed(['a'] * 9)) == 9
        assert [len(batch) for batch in calls] == [1, 4, 4]
        assert clock.now == 1120
    finally:
        embedder.client.close()


def test_rate_limit_error_is_not_retried_inside_the_sdk():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={'x-ratelimit-limit-tokens': '10'}, json={
            'error': {'message': 'temporary rate limit', 'type': 'tokens', 'code': 'rate_limit_exceeded'}
        })
    embedder, _ = real_sdk(handler)
    embedder.pacer.retry_policy = RetryPolicy(max_attempts=1)
    try:
        with pytest.raises(RateLimitError):
            embedder.embed(['a'])
        assert len(calls) == 1
        assert embedder.pacer.ledger.limits(embedder.pacer.scope, RateLimits())[1] == 10
    finally:
        embedder.client.close()


def test_successful_vectors_are_delivered_before_quota_ledger_write_failure():
    embedder, _ = real_sdk(lambda request: response(json.loads(request.content)['input']))
    delivered = []
    def fail(*args, **kwargs):
        raise OSError('ledger unavailable')
    embedder.pacer.observe = fail
    try:
        with pytest.raises(OSError):
            embedder.embed_checkpointed(['a'], on_batch=delivered.extend)
        assert delivered == [[1.0, 0.0]]
    finally:
        embedder.client.close()


@pytest.mark.parametrize('fail', [False, True])
def test_native_gemini_http_requests_are_paced_and_have_one_sdk_attempt(fail):
    from google import genai
    from google.genai import types
    from osintgpt.llm.gemini import GeminiEmbedding
    calls = []
    def handler(request):
        body = json.loads(request.content)
        batch = body.get('requests', [body])
        calls.append(batch)
        if fail:
            return httpx.Response(429, json={'error': {'code': 429, 'message': 'rate limit', 'status': 'RESOURCE_EXHAUSTED'}})
        return httpx.Response(200, json={'embeddings': [{'values': [1.0, 0.0]} for _ in batch]})
    embedder = GeminiEmbedding('gemini-embedding-001', 'test-key', rate_limits=RateLimits(rpm=2, tpm=100, batch_tokens=50))
    embedder.pacer.retry_policy = RetryPolicy(max_attempts=1)
    # Keep production retry options, replacing only its HTTP transport.
    options = embedder.client._api_client._http_options.model_copy(update={
        'client_args': {'transport': httpx.MockTransport(handler)}
    })
    embedder.client.close()
    embedder.client = genai.Client(api_key='test-key', http_options=options)
    clock = Clock()
    embedder.pacer.ledger = ledger(clock)
    try:
        if fail:
            with pytest.raises(Exception) as error:
                embedder.embed(['a'] * 3)
            assert '429' in str(error.value)
            assert len(calls) == 1
        else:
            assert len(embedder.embed(['a'] * 3)) == 3
            assert [len(batch) for batch in calls] == [2, 1]
            assert clock.now == 1030
    finally:
        embedder.client.close()
