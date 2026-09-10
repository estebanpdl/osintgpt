# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_followups.py
# Description: Questions to ask next. Two properties matter more than the
#   wording: a suggestion never breaks an answer, and it works when sent alone.
# =================================================================================

# import modules
import json
import math
import pytest

# import osintgpt
from osintgpt import Project, answer_question, index_project, suggest_followups
from osintgpt.followups import (
    DEFAULT_SUGGESTIONS,
    MAX_PASSAGES,
    PASSAGE_CHARS,
    _parse
)
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.projects import asked_questions, record_question
from osintgpt.vector_store import SearchResult, StoredChunk

MODEL = 'test-embedding'


class FlatEmbedder(EmbeddingProvider):
    model = MODEL

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Replier(GenerationProvider):
    '''Answers, then suggests, from a script.'''

    model = 'test-generation'

    def __init__(self, *replies, error=None):
        self.replies = list(replies)
        self.error = error
        self.prompts = []
        self.calls = 0

    def generate(self, system, user, **kwargs):
        self.calls += 1
        self.prompts.append(system)
        if self.error and self.calls > 1:
            raise self.error
        if not self.replies:
            return 'answer'

        return self.replies.pop(0)


def passage(ref, text='some passage text', citation=None):
    return SearchResult(
        chunk=StoredChunk(
            ref=ref, sequence=0, text=text, embedding_model=MODEL,
            path=citation or ''
        ),
        score=0.8
    )


@pytest.fixture
def embedder():
    return FlatEmbedder()


@pytest.fixture
def project(tmp_path, embedder):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()
    (material / 'alpha.md').write_text(
        '# Alpha\n\nAlpha Corp funded Beta Ltd in March.', encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register('material')
    index_project(instance, embedder)

    return instance


class TestItNeverBreaksAnAnswer:
    '''
    The suggestion call happens after an answer already succeeded. Anything
    going wrong there must cost the suggestions and nothing else.
    '''

    def test_a_provider_error_returns_nothing(self):
        generator = Replier(error=RuntimeError('the endpoint refused'))
        generator.calls = 1  # the answer call already happened

        assert suggest_followups(
            generator, 'q', 'a', [passage('a.md')]
        ) == []

    def test_an_unparseable_reply_returns_nothing(self):
        assert suggest_followups(
            Replier('not json at all'), 'q', 'a', [passage('a.md')]
        ) == []

    def test_a_reply_that_is_not_a_list_returns_nothing(self):
        assert suggest_followups(
            Replier('{"a": 1}'), 'q', 'a', [passage('a.md')]
        ) == []

    def test_the_answer_survives_a_failing_suggestion(self, project, embedder):
        generator = Replier('The answer.', 'broken json')

        answer = answer_question(project, 'q', embedder, generator)

        assert answer.text == 'The answer.'
        assert answer.followups == []

    def test_no_passages_means_no_call_at_all(self):
        '''
        Nothing was retrieved, so there is nothing to be curious about. A
        model asked anyway would invent questions from its training.
        '''
        generator = Replier('["invented"]')

        assert suggest_followups(generator, 'q', 'a', []) == []
        assert generator.calls == 0


class TestGrounding:
    def test_the_passages_reach_the_model(self):
        generator = Replier('[]')
        suggest_followups(
            generator, 'q', 'a',
            [passage('a.md', 'Alpha Corp funded Beta Ltd.')]
        )

        assert 'Alpha Corp funded Beta Ltd.' in generator.prompts[0]

    def test_the_prompt_forbids_general_knowledge(self):
        generator = Replier('[]')
        suggest_followups(generator, 'q', 'a', [passage('a.md')])

        assert 'answerable from **this** material' in generator.prompts[0]

    def test_the_prompt_requires_self_contained_questions(self):
        '''
        Each is sent as written — a numbered line, or a button — so one
        depending on this conversation arrives meaningless.
        '''
        generator = Replier('[]')
        suggest_followups(generator, 'q', 'a', [passage('a.md')])
        system = generator.prompts[0]

        assert 'stand on its own' in system
        assert 'as written' in system

    def test_material_is_bounded(self):
        generator = Replier('[]')
        many = [passage(f'{i}.md', 'x' * 5000) for i in range(50)]
        suggest_followups(generator, 'q', 'a', many)
        system = generator.prompts[0]

        assert system.count('.md]') <= MAX_PASSAGES
        assert 'x' * (PASSAGE_CHARS + 50) not in system

    def test_it_accepts_the_agentic_payload_shape(self):
        '''
        The static path carries SearchResult objects and the agentic path
        carries tool payloads. Suggestions follow either.
        '''
        generator = Replier('["A question about Beta Ltd."]')

        found = suggest_followups(
            generator, 'q', 'a',
            [{'citation': 'a.md', 'text': 'Beta Ltd was named.'}]
        )

        assert found == ['A question about Beta Ltd.']


class TestNotRepeating:
    def test_questions_already_asked_reach_the_model(self):
        generator = Replier('[]')
        suggest_followups(
            generator, 'q', 'a', [passage('a.md')],
            asked=['What did Alpha do?', 'Who funded Beta?']
        )

        assert 'Who funded Beta?' in generator.prompts[0]

    def test_the_question_log_records_what_was_asked(self, project, embedder):
        answer_question(project, 'A first question', embedder, Replier('x'))

        assert [q.text for q in asked_questions(project)] == [
            'A first question'
        ]

    def test_the_log_is_append_only(self, project):
        record_question(project, 'one')
        record_question(project, 'two')

        assert [q.text for q in asked_questions(project)] == ['one', 'two']

    def test_a_blank_question_is_not_recorded(self, project):
        assert record_question(project, '   ') is None
        assert asked_questions(project) == []

    def test_a_truncated_line_does_not_hide_the_history_above_it(
        self, project
    ):
        from osintgpt.projects.questions import questions_file

        record_question(project, 'kept')
        with questions_file(project).open('a', encoding='utf-8') as handle:
            handle.write('{"text": "truncated')

        assert [q.text for q in asked_questions(project)] == ['kept']

    def test_the_limit_returns_the_most_recent(self, project):
        for i in range(5):
            record_question(project, f'q{i}')

        assert [q.text for q in asked_questions(project, limit=2)] == [
            'q3', 'q4'
        ]

    def test_a_project_never_asked_has_no_log(self, project):
        assert asked_questions(project) == []


class TestTheSetting:
    def test_it_is_on_by_default(self):
        from osintgpt.projects import ProjectSettings

        assert ProjectSettings().suggest_followups is True

    def test_turning_it_off_makes_no_extra_call(self, project, embedder):
        off = project.with_settings(suggest_followups=False)
        off.save()
        generator = Replier('The answer.', '["should not be asked"]')

        answer = answer_question(off, 'q', embedder, generator)

        assert answer.followups == []
        assert generator.calls == 1

    def test_leaving_it_on_makes_one_extra_call(self, project, embedder):
        generator = Replier('The answer.', '["What else about Alpha Corp?"]')

        answer = answer_question(project, 'q', embedder, generator)

        assert answer.followups == ['What else about Alpha Corp?']
        assert generator.calls == 2


class TestParsing:
    @pytest.mark.parametrize('reply, expected', [
        ('["a", "b"]', ['a', 'b']),
        ('```json\n["a"]\n```', ['a']),
        ('Here you go: ["a"] — hope that helps', ['a']),
        ('["a", "a"]', ['a']),
        ('["a", "", "   "]', ['a']),
        ('["a", 3, null]', ['a']),
        ('[]', []),
        ('', []),
        ('[unclosed', [])
    ])
    def test_it_reads_what_it_can(self, reply, expected):
        assert _parse(reply) == expected

    def test_the_count_is_capped(self):
        many = '[' + ','.join(f'"q{i}"' for i in range(20)) + ']'

        found = suggest_followups(
            Replier(many), 'q', 'a', [passage('a.md')], n=3
        )

        assert len(found) == 3

    def test_the_default_is_few_enough_to_read(self):
        assert 1 <= DEFAULT_SUGGESTIONS <= 5
