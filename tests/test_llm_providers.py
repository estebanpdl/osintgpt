# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_providers.py
# Description: The provider layer — registry lookups, credential resolution,
#   and the one client that serves every OpenAI-compatible backend.
# =================================================================================

# import modules
import pytest

# import osintgpt config
from osintgpt.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    Settings
)

# import osintgpt llm
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    EmbeddingProvider,
    GenerationProvider,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm.openai_compat import (
    MAX_BATCH,
    OpenAICompatEmbedding,
    OpenAICompatGeneration
)
from osintgpt.llm.registry import (
    GEMINI_COMPAT_URL,
    OPENAI_COMPAT,
    VOYAGE_COMPAT_URL,
    backend_spec,
    connection_for,
    resolve_base_url
)

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

from conftest import FAKE_KEY, StubOpenAI


@pytest.fixture
def keyed():
    return Settings(
        openai_api_key=FAKE_KEY,
        gemini_api_key='gemini-key',
        voyage_api_key='voyage-key',
        openai_gpt_model='gpt-4o'
    )


class TestRegistry:
    def test_every_backend_declares_a_kind(self):
        for backends in (EMBEDDING_BACKENDS, GENERATION_BACKENDS):
            for name, spec in backends.items():
                assert spec.kind, name

    def test_an_unknown_id_lists_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            backend_spec('opnai', EMBEDDING_BACKENDS, 'embedding')

        message = str(excinfo.value)

        assert 'opnai' in message
        assert 'embedding' in message
        for name in EMBEDDING_BACKENDS:
            assert name in message

    def test_embedding_offers_more_backends_than_generation(self):
        '''Voyage embeds and does not generate.'''
        assert 'voyage' in EMBEDDING_BACKENDS
        assert 'voyage' not in GENERATION_BACKENDS


class TestBaseUrl:
    def test_openai_uses_the_client_default(self, keyed):
        _, base_url, _ = connection_for(
            'openai', EMBEDDING_BACKENDS, 'embedding', keyed
        )

        assert base_url is None

    @pytest.mark.parametrize('provider, expected', [
        ('gemini', GEMINI_COMPAT_URL),
        ('voyage', VOYAGE_COMPAT_URL)
    ])
    def test_compat_backends_carry_their_endpoint(
        self, keyed, provider, expected
    ):
        _, base_url, _ = connection_for(
            provider, EMBEDDING_BACKENDS, 'embedding', keyed
        )

        assert base_url == expected

    def test_ollama_defaults_to_localhost(self):
        _, base_url, _ = connection_for(
            'ollama', EMBEDDING_BACKENDS, 'embedding', Settings()
        )

        assert base_url == f'{DEFAULT_OLLAMA_BASE_URL}/v1'

    def test_ollama_host_is_configurable(self):
        settings = Settings(ollama_base_url='http://ollama:11434')
        _, base_url, _ = connection_for(
            'ollama', EMBEDDING_BACKENDS, 'embedding', settings
        )

        assert base_url == 'http://ollama:11434/v1'

    def test_a_trailing_slash_does_not_double_up(self):
        settings = Settings(ollama_base_url='http://ollama:11434/')
        spec = EMBEDDING_BACKENDS['ollama']

        assert resolve_base_url(spec, settings) == 'http://ollama:11434/v1'


class TestCredentials:
    def test_each_backend_reads_its_own_key(self, keyed):
        _, _, openai_key = connection_for(
            'openai', EMBEDDING_BACKENDS, 'embedding', keyed
        )
        _, _, gemini_key = connection_for(
            'gemini', EMBEDDING_BACKENDS, 'embedding', keyed
        )

        assert openai_key == FAKE_KEY
        assert gemini_key == 'gemini-key'

    def test_a_missing_key_names_the_variable_and_the_provider(self):
        with pytest.raises(MissingEnvironmentVariableError) as excinfo:
            connection_for(
                'gemini', EMBEDDING_BACKENDS, 'embedding', Settings()
            )

        message = str(excinfo.value)

        assert 'GEMINI_API_KEY' in message
        assert 'gemini' in message

    def test_an_openai_key_does_not_satisfy_gemini(self):
        settings = Settings(openai_api_key=FAKE_KEY)

        with pytest.raises(MissingEnvironmentVariableError):
            connection_for(
                'gemini', EMBEDDING_BACKENDS, 'embedding', settings
            )

    def test_ollama_needs_no_key(self):
        _, _, key = connection_for(
            'ollama', GENERATION_BACKENDS, 'generation', Settings()
        )

        assert key


class TestBuildEmbedding:
    def test_returns_the_interface(self, keyed):
        provider = build_embedding_provider('openai', keyed)

        assert isinstance(provider, EmbeddingProvider)
        assert isinstance(provider, OpenAICompatEmbedding)

    def test_one_class_serves_every_compat_backend(self, keyed):
        compat = [
            name for name, spec in EMBEDDING_BACKENDS.items()
            if spec.kind == OPENAI_COMPAT
        ]

        assert len(compat) > 1

        for name in compat:
            provider = build_embedding_provider(name, keyed, model='a-model')

            assert isinstance(provider, OpenAICompatEmbedding)

    def test_a_backend_without_a_default_model_demands_one(self, keyed):
        '''
        Only backends whose right answer is known and stable carry a default;
        guessing one per vendor is how a hardcoded model goes stale.
        '''
        with pytest.raises(ValueError, match='no model given'):
            build_embedding_provider('voyage', keyed)

    def test_model_falls_back_to_settings_then_the_library_default(self, keyed):
        assert build_embedding_provider('openai', keyed).model == (
            DEFAULT_EMBEDDING_MODEL
        )

        configured = keyed.with_overrides(
            openai_embedding_model='text-embedding-3-large'
        )

        assert build_embedding_provider('openai', configured).model == (
            'text-embedding-3-large'
        )

    def test_an_explicit_model_wins(self, keyed):
        provider = build_embedding_provider('openai', keyed, model='custom')

        assert provider.model == 'custom'

    def test_a_typo_is_caught_at_the_edge(self, keyed):
        with pytest.raises(ValueError, match='unknown embedding provider'):
            build_embedding_provider('openai-embeddings', keyed)


class TestBuildGeneration:
    def test_returns_the_interface(self, keyed):
        provider = build_generation_provider('openai', keyed)

        assert isinstance(provider, GenerationProvider)
        assert provider.model == 'gpt-4o'

    def test_requires_a_model(self):
        settings = Settings(openai_api_key=FAKE_KEY)

        with pytest.raises(ValueError, match='no model given'):
            build_generation_provider('openai', settings)

    def test_a_typo_is_caught_at_the_edge(self, keyed):
        with pytest.raises(ValueError, match='unknown generation provider'):
            build_generation_provider('claude', keyed)


class TestEmbeddingCalls:
    @pytest.fixture
    def provider(self, keyed):
        instance = build_embedding_provider('openai', keyed)
        instance.client = StubOpenAI()

        return instance

    def test_one_vector_per_input(self, provider):
        vectors = provider.embed(['a', 'b', 'c'])

        assert len(vectors) == 3

    def test_batches_at_the_gemini_ceiling(self, provider):
        provider.embed([f'doc {i}' for i in range(250)])

        assert provider.client.embeddings.batches == [100, 100, 50]

    def test_sends_the_configured_model(self, provider):
        provider.embed(['a'])

        assert provider.client.embeddings.models == [DEFAULT_EMBEDDING_MODEL]

    def test_reorders_a_response_that_arrives_shuffled(self, provider):
        '''Nothing in the API promises the batch comes back in order.'''
        from types import SimpleNamespace

        def shuffled(*, model, input):
            items = [
                SimpleNamespace(index=i, embedding=[float(i)])
                for i in range(len(input))
            ]

            return SimpleNamespace(data=list(reversed(items)), model=model)

        provider.client.embeddings.create = shuffled

        assert provider.embed(['a', 'b', 'c']) == [[0.0], [1.0], [2.0]]

    def test_no_texts_makes_no_request(self, provider):
        assert provider.embed([]) == []
        assert provider.client.embeddings.models == []


class TestGenerationCalls:
    @pytest.fixture
    def provider(self, keyed):
        instance = build_generation_provider('openai', keyed)
        instance.client = StubOpenAI()

        return instance

    def test_returns_the_reply(self, provider):
        assert provider.generate('be terse', 'a question') == 'STUB REPLY'

    def test_sends_system_and_user_in_order(self, provider):
        provider.generate('be terse', 'a question')
        messages = provider.client.chat.completions.calls[0]['messages']

        assert [m['role'] for m in messages] == ['system', 'user']
        assert messages[0]['content'] == 'be terse'
        assert messages[1]['content'] == 'a question'

    def test_an_empty_reply_becomes_an_empty_string(self, provider):
        from types import SimpleNamespace

        provider.client.chat.completions.create = lambda **kwargs: (
            SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=None))
            ])
        )

        assert provider.generate('s', 'u') == ''
