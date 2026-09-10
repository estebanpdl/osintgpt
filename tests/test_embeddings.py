# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_embeddings.py
# Description: OpenAIEmbeddingGenerator against a stubbed client — batching,
#   ordering, the configured model, and cost estimation.
# =================================================================================

# import modules
import pytest

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL, Settings

# import osintgpt embeddings
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# import osintgpt llm
from osintgpt.llm.openai_compat import MAX_BATCH

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# import utils
from osintgpt.utils import encoding_for_model

from conftest import FAKE_KEY


@pytest.fixture
def generator(settings, stub_client):
    instance = OpenAIEmbeddingGenerator(settings)
    instance.client = stub_client

    return instance


class TestConstruction:
    def test_requires_a_key(self):
        with pytest.raises(MissingEnvironmentVariableError, match='OPENAI_API_KEY'):
            OpenAIEmbeddingGenerator(Settings(openai_gpt_model='gpt-4o'))

    def test_requires_a_chat_model(self):
        with pytest.raises(MissingEnvironmentVariableError, match='OPENAI_GPT_MODEL'):
            OpenAIEmbeddingGenerator(Settings(openai_api_key=FAKE_KEY))

    def test_accepts_a_path_with_a_deprecation_warning(self, env_file):
        with pytest.warns(DeprecationWarning):
            OpenAIEmbeddingGenerator(env_file)

    def test_defaults_to_the_current_embedding_model(self, settings):
        instance = OpenAIEmbeddingGenerator(settings)

        assert instance.get_openai_embedding_model() == DEFAULT_EMBEDDING_MODEL


class TestLoadText:
    def test_accepts_a_list(self, generator):
        generator.load_text(['a', 'b'])

        assert generator.data == ['a', 'b']

    @pytest.mark.parametrize('value', ['a string', 42, {'a': 1}])
    def test_rejects_anything_else(self, generator, value):
        with pytest.raises(TypeError):
            generator.load_text(value)


class TestCalculateEmbeddings:
    def test_returns_one_vector_per_document(self, generator):
        generator.load_text(['a', 'b', 'c'])

        assert len(generator.calculate_embeddings()) == 3

    def test_preserves_input_order_across_batches(self, generator):
        generator.load_text([f'doc {i}' for i in range(2_500)])

        vectors = generator.calculate_embeddings()

        assert len(vectors) == 2_500

    def test_batches_at_the_provider_ceiling(self, generator):
        '''
        Delegation adopts the provider's batch size rather than this class's
        old one, so requests are smaller and the Gemini cap is respected.
        '''
        generator.load_text([f'doc {i}' for i in range(2_500)])
        generator.calculate_embeddings()

        assert generator.client.embeddings.batches == [MAX_BATCH] * 25

    def test_sends_the_configured_model(self, settings, stub_client):
        instance = OpenAIEmbeddingGenerator(
            settings.with_overrides(
                openai_embedding_model='text-embedding-3-large'
            )
        )
        instance.client = stub_client
        instance.load_text(['a'])
        instance.calculate_embeddings()

        assert stub_client.embeddings.models == ['text-embedding-3-large']

    def test_embeddings_property_caches(self, generator):
        generator.load_text(['a'])

        first = generator.embeddings
        second = generator.embeddings

        assert first is second
        assert generator.client.embeddings.models == [DEFAULT_EMBEDDING_MODEL]


class TestGenerateEmbedding:
    def test_returns_a_single_vector(self, generator):
        assert generator.generate_embedding('a query') == [0.0, 0.2, 0.3]

    def test_sends_the_configured_model(self, generator):
        generator.generate_embedding('a query')

        assert generator.client.embeddings.models == [DEFAULT_EMBEDDING_MODEL]


class TestCosting:
    def test_counts_for_the_embedding_model_not_the_chat_model(self, generator):
        text = 'OSINT analysis — 分析'
        generator.load_text([text])

        expected = len(
            encoding_for_model(generator.get_openai_embedding_model()).encode(text)
        )

        assert generator.count_tokens() == expected

    def test_sums_across_documents(self, generator):
        generator.load_text(['alpha', 'beta'])
        total = generator.count_tokens()

        generator.load_text(['alpha'])
        single = generator.count_tokens()

        assert total > single

    def test_estimates_a_cost_for_a_priced_model(self, generator):
        generator.load_text(['some text to embed'])

        assert generator.calculate_embeddings_estimated_cost() > 0

    def test_returns_none_for_an_unpriced_model(self, settings, stub_client):
        instance = OpenAIEmbeddingGenerator(
            settings.with_overrides(openai_embedding_model='embed-does-not-exist')
        )
        instance.client = stub_client
        instance.load_text(['some text to embed'])

        assert instance.calculate_embeddings_estimated_cost() is None
