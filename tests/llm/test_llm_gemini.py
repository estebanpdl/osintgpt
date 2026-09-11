# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_gemini.py
# Description: The native Gemini embedding backend — that a retrieval pair is
#   embedded as a pair, whichever way the model wants to be told.
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
    EmbeddingProvider,
    EmbeddingPurpose,
    build_embedding_provider
)
from osintgpt.llm.gemini import (
    INSTRUCTION_STYLE,
    MAX_BATCH,
    TASK_TYPE_STYLE,
    GeminiEmbedding
)
from osintgpt.llm.registry import GEMINI, OPENAI_COMPAT

INSTRUCTION_MODEL = 'gemini-embedding-2'
TASK_TYPE_MODEL = 'gemini-embedding-001'


class StubModels:
    '''Records every request; returns one vector per input.'''

    def __init__(self, token_count=None):
        self.calls = []
        self.token_count = token_count

    def embed_content(self, *, model, contents, config):
        self.calls.append(
            SimpleNamespace(model=model, contents=contents, config=config)
        )
        statistics = (
            None if self.token_count is None
            else SimpleNamespace(token_count=self.token_count)
        )

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[float(i), 0.2, 0.3], statistics=statistics
                )
                for i in range(len(contents))
            ]
        )


def provider(model, token_count=None, **kwargs):
    instance = GeminiEmbedding(model=model, api_key='gemini-key', **kwargs)
    instance.client = SimpleNamespace(models=StubModels(token_count))

    return instance


class TestRegistry:
    def test_embedding_uses_the_native_sdk(self):
        assert EMBEDDING_BACKENDS['gemini'].kind == GEMINI
        assert EMBEDDING_BACKENDS['gemini'].base_url is None

    def test_generation_is_untouched(self):
        assert GENERATION_BACKENDS['gemini'].kind == OPENAI_COMPAT

    def test_the_factory_builds_it(self):
        built = build_embedding_provider(
            'gemini',
            Settings(gemini_api_key='gemini-key'),
            model=TASK_TYPE_MODEL
        )

        assert isinstance(built, EmbeddingProvider)
        assert isinstance(built, GeminiEmbedding)


class TestTaskStyle:
    def test_a_model_taking_the_parameter_is_detected(self):
        assert provider(TASK_TYPE_MODEL).task_style == TASK_TYPE_STYLE

    def test_anything_else_is_instructed(self):
        '''
        Google is moving away from the parameter, so a model this set has
        never heard of inherits the current behaviour rather than an error.
        '''
        assert provider(INSTRUCTION_MODEL).task_style == INSTRUCTION_STYLE
        assert provider('gemini-embedding-9').task_style == INSTRUCTION_STYLE

    def test_the_operator_can_override_it(self):
        instance = provider(INSTRUCTION_MODEL, task_style=TASK_TYPE_STYLE)

        assert instance.task_style == TASK_TYPE_STYLE

    def test_a_typo_is_caught_at_construction(self):
        with pytest.raises(ValueError, match='unknown task style'):
            provider(TASK_TYPE_MODEL, task_style='retrieval')


class TestTaskTypeModels:
    def test_a_document_is_sent_as_a_document(self):
        instance = provider(TASK_TYPE_MODEL)
        instance.embed(['a passage'], purpose=EmbeddingPurpose.DOCUMENT)

        assert instance.client.models.calls[0].config.task_type == (
            'RETRIEVAL_DOCUMENT'
        )

    def test_a_query_is_sent_as_a_query(self):
        instance = provider(TASK_TYPE_MODEL)
        instance.embed(['a question'], purpose=EmbeddingPurpose.QUERY)

        assert instance.client.models.calls[0].config.task_type == (
            'RETRIEVAL_QUERY'
        )

    def test_the_text_is_sent_unchanged(self):
        instance = provider(TASK_TYPE_MODEL)
        instance.embed(['a passage'], purpose=EmbeddingPurpose.DOCUMENT)

        assert instance.client.models.calls[0].contents == ['a passage']


class TestInstructedModels:
    def test_no_task_type_is_sent(self):
        '''The models taking an instruction reject the parameter outright.'''
        instance = provider(INSTRUCTION_MODEL)
        instance.embed(['a passage'], purpose=EmbeddingPurpose.DOCUMENT)

        assert instance.client.models.calls[0].config is None

    def test_a_document_carries_the_document_template(self):
        instance = provider(INSTRUCTION_MODEL)
        instance.embed(['a passage'], purpose=EmbeddingPurpose.DOCUMENT)
        content = instance.client.models.calls[0].contents[0]

        assert content.parts[0].text == 'title: none | text: a passage'

    def test_a_query_carries_the_query_template(self):
        instance = provider(INSTRUCTION_MODEL)
        instance.embed(['who paid'], purpose=EmbeddingPurpose.QUERY)
        content = instance.client.models.calls[0].contents[0]

        assert content.parts[0].text == 'task: search result | query: who paid'

    def test_each_input_is_its_own_content(self):
        '''
        Bare strings would come back as one aggregated vector for the whole
        batch, which nothing downstream could tell from a real result.
        '''
        instance = provider(INSTRUCTION_MODEL)
        vectors = instance.embed(['one', 'two', 'three'])

        assert len(instance.client.models.calls[0].contents) == 3
        assert len(vectors) == 3


class TestEmbedding:
    def test_one_vector_per_input(self):
        assert len(provider(TASK_TYPE_MODEL).embed(['a', 'b', 'c'])) == 3

    def test_no_texts_makes_no_request(self):
        instance = provider(TASK_TYPE_MODEL)

        assert instance.embed([]) == []
        assert instance.client.models.calls == []

    def test_batches_at_the_gemini_ceiling(self):
        instance = provider(TASK_TYPE_MODEL)
        instance.embed([f'doc {i}' for i in range(250)])
        sizes = [len(call.contents) for call in instance.client.models.calls]

        assert sizes == [MAX_BATCH, MAX_BATCH, 50]

    def test_a_short_response_is_refused(self):
        instance = provider(INSTRUCTION_MODEL)
        instance.client.models.embed_content = lambda **kwargs: (
            SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])
        )

        with pytest.raises(RuntimeError, match='1 vectors for 3 inputs'):
            instance.embed(['one', 'two', 'three'])

    def test_the_purpose_defaults_to_document(self):
        instance = provider(TASK_TYPE_MODEL)
        instance.embed(['a passage'])

        assert instance.client.models.calls[0].config.task_type == (
            'RETRIEVAL_DOCUMENT'
        )


class TestUsage:
    def test_an_absent_token_count_says_so(self):
        '''
        The developer API reports billable characters, which is not a token
        count; reporting it as one would price a run wrongly and quietly.
        '''
        from osintgpt.llm.usage import UsageRecorder

        recorder = UsageRecorder()
        instance = provider(TASK_TYPE_MODEL, recorder=recorder)
        instance.embed(['a passage'])

        assert recorder.records[0].counted is False
        assert recorder.records[0].input_tokens == 0

    def test_a_reported_count_is_summed(self):
        from osintgpt.llm.usage import UsageRecorder

        recorder = UsageRecorder()
        instance = provider(TASK_TYPE_MODEL, token_count=7, recorder=recorder)
        instance.embed(['one', 'two'])

        assert recorder.records[0].counted is True
        assert recorder.records[0].input_tokens == 14
        assert recorder.records[0].provider == 'gemini'
