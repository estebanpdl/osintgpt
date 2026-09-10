# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_local.py
# Description: The in-process embedding backend — same interface as the hosted
#   ones, no key, no endpoint, and no network once the model is present.
# =================================================================================

# import modules
import builtins
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL, Settings

# import osintgpt llm
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    EmbeddingProvider,
    build_embedding_provider
)
from osintgpt.llm.local import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    SentenceTransformerEmbedding
)
from osintgpt.llm.registry import SENTENCE_TRANSFORMERS

LOCAL = 'sentence-transformers'


class StubEncoder:
    '''Stands in for a loaded SentenceTransformer.'''

    def __init__(self, dimensions=3):
        self.calls = []
        self.dimensions = dimensions

    def encode(self, texts):
        self.calls.append(list(texts))

        # The real encoder returns NumPy rows, which expose tolist().
        return [_Row([float(i)] * self.dimensions) for i in range(len(texts))]


class _Row(list):
    def tolist(self):
        return list(self)


@pytest.fixture
def provider():
    return SentenceTransformerEmbedding(encoder=StubEncoder())


class TestRegistration:
    def test_is_an_embedding_backend(self):
        assert LOCAL in EMBEDDING_BACKENDS

    def test_generates_nothing(self):
        '''It embeds; a local chat model is Ollama's job.'''
        assert LOCAL not in GENERATION_BACKENDS

    def test_uses_its_own_kind(self):
        assert EMBEDDING_BACKENDS[LOCAL].kind == SENTENCE_TRANSFORMERS

    def test_declares_the_extra_that_installs_it(self):
        assert EMBEDDING_BACKENDS[LOCAL].extra == 'local'

    def test_needs_no_credential(self):
        assert EMBEDDING_BACKENDS[LOCAL].settings_field is None

    def test_has_no_endpoint(self):
        assert EMBEDDING_BACKENDS[LOCAL].base_url is None
        assert EMBEDDING_BACKENDS[LOCAL].ollama is False


class TestDefaultModel:
    def test_defaults_to_a_local_model_not_an_openai_one(self):
        default = EMBEDDING_BACKENDS[LOCAL].default_model

        assert default == DEFAULT_LOCAL_EMBEDDING_MODEL
        assert default != DEFAULT_EMBEDDING_MODEL

    def test_the_factory_picks_the_backend_default(self, monkeypatch):
        built = {}

        def capture(model, **kwargs):
            built['model'] = model

            return SentenceTransformerEmbedding(
                model=model, encoder=StubEncoder()
            )

        monkeypatch.setattr(
            'osintgpt.llm.SentenceTransformerEmbedding', capture
        )
        build_embedding_provider(LOCAL, Settings())

        assert built['model'] == DEFAULT_LOCAL_EMBEDDING_MODEL

    def test_a_configured_openai_model_does_not_leak_across(self, monkeypatch):
        '''
        Settings may carry an OpenAI embedding model from another project;
        loading that name locally would fail, so the caller must be explicit.
        '''
        built = {}
        monkeypatch.setattr(
            'osintgpt.llm.SentenceTransformerEmbedding',
            lambda model, **kwargs: built.setdefault('model', model)
        )
        build_embedding_provider(
            LOCAL, Settings(), model='BAAI/bge-small-en-v1.5'
        )

        assert built['model'] == 'BAAI/bge-small-en-v1.5'


class TestInterface:
    def test_satisfies_the_shared_interface(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_carries_its_model(self, provider):
        assert provider.model == DEFAULT_LOCAL_EMBEDDING_MODEL

    def test_declares_no_image_support(self, provider):
        assert provider.supports_images is False


class TestEmbed:
    def test_one_vector_per_input(self, provider):
        vectors = provider.embed(['a', 'b', 'c'])

        assert len(vectors) == 3

    def test_returns_plain_lists_of_floats(self, provider):
        vectors = provider.embed(['a'])

        assert isinstance(vectors[0], list)
        assert not isinstance(vectors[0], _Row)
        assert all(isinstance(value, float) for value in vectors[0])

    def test_passes_every_text_to_the_encoder_at_once(self, provider):
        provider.embed(['a', 'b'])

        assert provider.encoder.calls == [['a', 'b']]

    def test_no_texts_makes_no_call(self, provider):
        assert provider.embed([]) == []
        assert provider.encoder.calls == []


class TestMissingPackage:
    def test_names_the_extra_that_installs_it(self, monkeypatch):
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'sentence_transformers':
                raise ImportError('No module named sentence_transformers')

            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        with pytest.raises(ImportError) as excinfo:
            SentenceTransformerEmbedding()

        message = str(excinfo.value)

        assert 'sentence-transformers' in message
        assert 'osintgpt[local]' in message


class TestImageSupport:
    '''
    Whether a local model can embed images is a property of the model, and it
    is worth asking rather than inferring: a wrong yes embeds an image into a
    text-only space and reports success.
    '''

    def test_a_model_declaring_text_only_refuses_images(self):
        encoder = SimpleNamespace(modalities=['text'])
        provider = SentenceTransformerEmbedding(encoder=encoder)

        assert provider.supports_images is False

    def test_a_model_declaring_images_accepts_them(self):
        encoder = SimpleNamespace(modalities=['text', 'image'])
        provider = SentenceTransformerEmbedding(encoder=encoder)

        assert provider.supports_images is True

    def test_a_model_that_does_not_say_is_taken_at_text_only(self):
        '''
        Older sentence-transformers declares nothing. Refusing an image the
        operator can then enable is recoverable; embedding one into a
        text-only space silently is not.
        '''
        provider = SentenceTransformerEmbedding(encoder=SimpleNamespace())

        assert provider.supports_images is False

    def test_refusing_names_the_model_that_has_to_change(self):
        provider = SentenceTransformerEmbedding(
            model='some-text-model', encoder=SimpleNamespace(modalities=['text'])
        )

        with pytest.raises(NotImplementedError, match='some-text-model'):
            provider.embed_images([b'bytes'])

    def test_no_images_is_not_a_provider_call(self):
        provider = SentenceTransformerEmbedding(
            encoder=SimpleNamespace(modalities=['text'])
        )

        assert provider.embed_images([]) == []
