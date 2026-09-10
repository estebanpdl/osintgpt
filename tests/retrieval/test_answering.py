# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_answering.py
# Description: Grounded answering. The property worth protecting is that the
#   model is never asked a question it has no passages for.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, answer_question, index_project
from osintgpt.answering import (
    DEFAULT_PASSAGES,
    NOTHING_RETRIEVED,
    Answer,
    build_prompt
)
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.vector_store import SearchResult, StoredChunk

MODEL = 'test-embedding'


class WordEmbedder(EmbeddingProvider):
    '''A vector per known word, so retrieval is predictable and offline.'''

    model = MODEL
    VOCABULARY = 'aardvark zebra quokka enforcement narwhal'.split()

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        low = text.lower()
        counts = [float(low.count(w)) for w in self.VOCABULARY]
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


class RecordingGenerator(GenerationProvider):
    '''Returns a fixed reply and keeps every prompt it was given.'''

    model = 'test-generation'

    def __init__(self, reply='A grounded answer [1].'):
        self.reply = reply
        self.prompts = []

    def generate(self, system, user, **kwargs):
        self.prompts.append((system, user))

        return self.reply


def result(ref, text, path='', score=0.9):
    return SearchResult(
        chunk=StoredChunk(
            ref=ref, sequence=0, text=text, embedding_model=MODEL, path=path
        ),
        score=score
    )


@pytest.fixture
def embedder():
    return WordEmbedder()


@pytest.fixture
def generator():
    return RecordingGenerator()


@pytest.fixture
def project(tmp_path, embedder):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()
    (material / 'alpha.md').write_text(
        '# Alpha\n\nA report about aardvark enforcement.', encoding='utf-8'
    )
    (material / 'beta.md').write_text(
        '# Beta\n\nA report about zebra sightings.', encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register('material')
    index_project(instance, embedder)

    return instance


class TestGrounding:
    def test_the_answer_carries_the_passages_it_used(
        self, project, embedder, generator
    ):
        answer = answer_question(project, 'aardvark', embedder, generator)

        assert answer.passages
        assert answer.generated is True

    def test_the_passages_reach_the_model(
        self, project, embedder, generator
    ):
        answer_question(project, 'aardvark', embedder, generator)
        system, _ = generator.prompts[0]

        assert 'aardvark' in system.lower()

    def test_the_question_reaches_the_model(
        self, project, embedder, generator
    ):
        answer_question(project, 'aardvark enforcement', embedder, generator)
        _, user = generator.prompts[0]

        assert user == 'aardvark enforcement'

    def test_retrieval_can_be_restricted(
        self, project, embedder, generator
    ):
        answer = answer_question(
            project, 'aardvark', embedder, generator,
            refs=['material/beta.md']
        )

        assert all(p.ref == 'material/beta.md' for p in answer.passages)


class TestNothingRetrieved:
    '''
    The property this design exists for: a model given no passages answers
    from its training, and that answer is indistinguishable from a grounded
    one until someone checks it.
    '''

    def test_no_generation_call_is_made(self, tmp_path, embedder, generator):
        empty = Project.create('Empty', home=tmp_path)

        answer_question(empty, 'anything', embedder, generator)

        assert generator.prompts == []

    def test_it_says_so_rather_than_answering(
        self, tmp_path, embedder, generator
    ):
        empty = Project.create('Empty', home=tmp_path)

        answer = answer_question(empty, 'anything', embedder, generator)

        assert answer.text == NOTHING_RETRIEVED
        assert answer.generated is False

    def test_it_offers_no_sources(self, tmp_path, embedder, generator):
        empty = Project.create('Empty', home=tmp_path)

        assert answer_question(empty, 'q', embedder, generator).sources == []


class TestCitations:
    def test_passages_are_numbered_from_one(self):
        text = build_prompt('q', [result('a.md', 'first'),
                                  result('b.md', 'second')])

        assert '[1] a.md' in text
        assert '[2] b.md' in text

    def test_a_citation_names_the_section_when_there_is_one(self):
        text = build_prompt('q', [result('a.md', 'x', path='Report › Part')])

        assert 'a.md › Report › Part' in text

    def test_the_answer_lists_its_citations_in_order(self):
        answer = Answer(
            question='q', text='a',
            passages=[result('a.md', 'x'), result('b.md', 'y')]
        )

        assert answer.citations == ['[1] a.md', '[2] b.md']

    def test_sources_deduplicate_across_passages(self):
        '''
        A document that contributed three passages is one source to read, not
        three, and a reader should not be sent to it three times.
        '''
        answer = Answer(
            question='q', text='a',
            passages=[result('a.md', 'x'), result('a.md', 'y'),
                      result('b.md', 'z')]
        )

        assert answer.sources == ['a.md', 'b.md']

    def test_sources_keep_the_ranking_order(self):
        answer = Answer(
            question='q', text='a',
            passages=[result('b.md', 'x', score=0.9),
                      result('a.md', 'y', score=0.5)]
        )

        assert answer.sources == ['b.md', 'a.md']


class TestThePrompt:
    def test_it_forbids_answering_from_outside_the_passages(self):
        text = build_prompt('q', [result('a.md', 'x')]).lower()

        assert 'only from the passages' in text

    def test_it_asks_for_a_refusal_rather_than_a_guess(self):
        text = build_prompt('q', [result('a.md', 'x')]).lower()

        assert 'do not contain the answer' in text

    def test_it_does_not_name_a_language(self):
        '''
        The corpus and the analyst may be in any language, and an example
        naming one is a recommendation whether or not it was meant as one.
        '''
        text = build_prompt('¿Cuántos estados?', [result('a.md', 'x')]).lower()

        for language in ('english', 'spanish', 'français', 'in english'):
            assert language not in text

    def test_a_passage_in_another_script_survives_intact(self):
        text = build_prompt('q', [result('a.md', 'Анализ нарративов 分析')])

        assert 'Анализ нарративов 分析' in text

    def test_it_carries_no_unrendered_placeholder(self):
        text = build_prompt('q', [result('a.md', 'x')])

        assert '{{' not in text and '{%' not in text


class TestPassageCount:
    def test_the_default_is_what_an_analyst_would_read(self):
        assert 2 <= DEFAULT_PASSAGES <= 20

    def test_it_is_configurable(self, project, embedder, generator):
        answer = answer_question(
            project, 'aardvark zebra', embedder, generator, passages=1
        )

        assert len(answer.passages) <= 1
