# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_discovery.py
# Description: Asking a backend which models it has. Matters most for Ollama,
#   where the answer is what has actually been pulled onto this machine.
# =================================================================================

# import modules
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llm
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm.local import SentenceTransformerEmbedding

from conftest import FAKE_KEY


class StubModels:
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def list(self):
        self.calls += 1

        return [SimpleNamespace(id=name) for name in self.ids]


@pytest.fixture
def keyed():
    return Settings(openai_api_key=FAKE_KEY, openai_gpt_model='gpt-4o')


class TestDeclaredSupport:
    def test_ollama_declares_discovery(self):
        assert EMBEDDING_BACKENDS['ollama'].discovers_models is True
        assert GENERATION_BACKENDS['ollama'].discovers_models is True

    def test_openai_declares_discovery(self):
        assert EMBEDDING_BACKENDS['openai'].discovers_models is True
        assert GENERATION_BACKENDS['openai'].discovers_models is True

    def test_anthropic_declares_discovery(self):
        assert GENERATION_BACKENDS['anthropic'].discovers_models is True

    @pytest.mark.parametrize('provider', ['gemini', 'voyage'])
    def test_unverified_endpoints_do_not_claim_it(self, provider):
        '''
        Claiming discovery and returning a 404 is worse than not offering it,
        so a backend stays false until its endpoint is known to answer.
        '''
        assert EMBEDDING_BACKENDS[provider].discovers_models is False

    def test_the_local_backend_cannot_be_asked(self):
        assert EMBEDDING_BACKENDS['sentence-transformers'].discovers_models is (
            False
        )


class TestOllamaDiscovery:
    def test_reports_the_models_on_this_machine(self):
        provider = build_embedding_provider(
            'ollama', Settings(), model='nomic-embed-text'
        )
        provider.client = SimpleNamespace(
            models=StubModels(['qwen3:8b', 'nomic-embed-text', 'llama3.2'])
        )

        assert provider.list_models() == [
            'llama3.2', 'nomic-embed-text', 'qwen3:8b'
        ]

    def test_the_capability_reaches_the_provider(self):
        provider = build_embedding_provider(
            'ollama', Settings(), model='nomic-embed-text'
        )

        assert provider.supports_model_discovery is True

    def test_generation_side_works_the_same(self):
        provider = build_generation_provider(
            'ollama', Settings(), model='qwen3:8b'
        )
        provider.client = SimpleNamespace(models=StubModels(['qwen3:8b']))

        assert provider.list_models() == ['qwen3:8b']


class TestUnsupportedBackends:
    def test_the_capability_is_false_where_unverified(self, keyed):
        provider = build_embedding_provider(
            'voyage', keyed.with_overrides(voyage_api_key='k'),
            model='a-model'
        )

        assert provider.supports_model_discovery is False

    def test_the_local_backend_says_it_cannot(self):
        provider = SentenceTransformerEmbedding(encoder=object())

        assert provider.supports_model_discovery is False

        with pytest.raises(NotImplementedError, match='cannot list its models'):
            provider.list_models()

    def test_the_error_names_the_backend(self):
        provider = SentenceTransformerEmbedding(encoder=object())

        with pytest.raises(NotImplementedError) as excinfo:
            provider.list_models()

        assert 'SentenceTransformerEmbedding' in str(excinfo.value)
