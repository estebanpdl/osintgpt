# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_lexical.py
# Description: The exact-match leg. Most of what it exists to catch is
#   non-English or non-alphabetic, so most of these tests are too.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, derive_search_terms, index_project, lexical_search
from osintgpt.ingestion import Corpus
from osintgpt.lexical import (
    MAX_TERMS,
    MIN_TERM_LENGTH,
    LexicalHit,
    _parse_terms,
    _usable
)
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.vector_store import SQLiteVectorStore, StoredChunk

MODEL = 'test-embedding'


class FlatEmbedder(EmbeddingProvider):
    '''Vectors are irrelevant here; the lexical leg never reads one.'''

    model = MODEL

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ScriptedGenerator(GenerationProvider):
    '''Replies with whatever it was handed, or raises.'''

    model = 'test-generation'

    def __init__(self, reply='["term"]', error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def generate(self, system, user, **kwargs):
        self.prompts.append((system, user))
        if self.error:
            raise self.error

        return self.reply


@pytest.fixture
def embedder():
    return FlatEmbedder()


@pytest.fixture
def store():
    with SQLiteVectorStore(':memory:') as engine:
        yield engine


def add(store, ref, text):
    store.upsert(
        ref,
        [StoredChunk(ref=ref, sequence=0, text=text, embedding_model=MODEL)],
        [[1.0, 0.0]]
    )


@pytest.fixture
def project(tmp_path, embedder):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()
    (material / 'alpha.md').write_text(
        '# Alpha\n\nContacted @acct_1 about hash 3f2a9c1b.', encoding='utf-8'
    )
    (material / 'beta.md').write_text(
        '# Beta\n\nA report mentioning @acct_1 only.', encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register('material')
    index_project(instance, embedder)

    return instance


class TestWhatEmbeddingsBlur:
    '''
    The identifiers this leg exists for. Each is a token a tokenizer splits
    and a vector cannot distinguish from its neighbours.
    '''

    @pytest.mark.parametrize('text, term', [
        ('contacted @acct_1 twice', '@acct_1'),
        ('see https://example.invalid/a/b', 'example.invalid'),
        ('sha 3f2a9c1bd4e5', '3f2a9c1b'),
        ('error E_0042 raised', 'E_0042'),
        ('case 2026/CR/118 filed', '2026/CR/118'),
        ('file report_final_v2.docx', 'report_final_v2'),
        ('+34 600 123 456 called', '600 123 456')
    ])
    def test_an_identifier_is_found_inside_a_sentence(
        self, store, text, term
    ):
        add(store, 'a.md', text)

        assert [c.ref for c in store.match_text(term)] == ['a.md']


class TestScripts:
    '''
    Case folding is the single most likely place to reintroduce an
    English-only assumption, so every script it could break on is here.
    '''

    @pytest.mark.parametrize('stored, term', [
        ('Анализ НАРРАТИВОВ в сети', 'нарративов'),
        ('ΑΝΑΛΥΣΗ δικτύων', 'αναλυση'),
        ('İSTANBUL raporu', 'raporu'),
        ('GROSSE Straße', 'straße'),
        ('تحليل الروايات المتعددة', 'الروايات'),
        ('多语言叙事分析', '叙事'),
        ('ANÁLISIS de la campaña', 'análisis'),
        ('נרטיבים מרובים', 'נרטיבים')
    ])
    def test_case_and_script_do_not_decide_a_match(self, store, stored, term):
        add(store, 'a.md', stored)

        assert [c.ref for c in store.match_text(term)] == ['a.md']

    def test_an_accented_term_does_not_match_its_unaccented_form(self, store):
        '''
        Folding case is not folding accents. `campana` and `campaña` are
        different words, and an exact leg that conflates them is not exact.
        '''
        add(store, 'a.md', 'la campaña de desinformación')

        assert store.match_text('campana') == []


class TestPatternCharacters:
    def test_a_percent_is_literal(self, store):
        add(store, 'a.md', 'margin was 50% overall')
        add(store, 'b.md', 'no figure here')

        assert [c.ref for c in store.match_text('50%')] == ['a.md']

    def test_a_bare_wildcard_does_not_match_everything(self, store):
        add(store, 'a.md', 'contains a % sign')
        add(store, 'b.md', 'contains none')

        assert [c.ref for c in store.match_text('%')] == ['a.md']

    def test_an_underscore_is_literal(self, store):
        add(store, 'a.md', 'account_1 was named')
        add(store, 'b.md', 'accountX1 was named')

        assert [c.ref for c in store.match_text('account_1')] == ['a.md']


class TestRanking:
    def test_more_terms_matched_ranks_higher(self, project, embedder):
        results = lexical_search(
            project, ['@acct_1', '3f2a9c1b'], embedding_model=MODEL
        )

        assert results[0].ref == 'material/alpha.md'
        assert results[0].score > results[1].score

    def test_the_score_is_term_coverage(self, project, embedder):
        '''
        Coverage rather than frequency: a chunk mentioning three of four
        things asked about beats one repeating a single term twenty times.
        '''
        results = lexical_search(
            project, ['@acct_1', '3f2a9c1b'], embedding_model=MODEL
        )
        best = next(r for r in results if 'alpha' in r.ref)
        other = next(r for r in results if 'beta' in r.ref)

        assert best.score == pytest.approx(1.0)
        assert other.score == pytest.approx(0.5)

    def test_a_chunk_found_by_several_terms_appears_once(
        self, project, embedder
    ):
        results = lexical_search(
            project, ['@acct_1', '3f2a9c1b'], embedding_model=MODEL
        )

        assert len(results) == len({(r.ref, r.chunk.sequence) for r in results})

    def test_hit_score_of_nothing_is_zero_rather_than_dividing(self):
        assert LexicalHit(chunk=None, terms=[]).score(0) == 0.0


class TestTermHygiene:
    def test_terms_are_deduplicated_case_insensitively(self):
        assert _usable(['Nimbus', 'nimbus', 'NIMBUS']) == ['Nimbus']

    def test_a_term_too_short_to_be_selective_is_dropped(self):
        assert _usable(['a', 'ab']) == ['ab']
        assert MIN_TERM_LENGTH >= 2

    def test_whitespace_is_trimmed(self):
        assert _usable(['  Nimbus  ']) == ['Nimbus']

    def test_no_usable_terms_searches_nothing(self, project):
        assert lexical_search(project, ['a', '']) == []

    def test_an_empty_term_list_searches_nothing(self, project):
        assert lexical_search(project, []) == []


class TestDerivingTerms:
    '''
    The plan's requirement: terms come from the model, never a stopword list,
    because every stopword list is bound to one language.
    '''

    def test_it_returns_what_the_model_chose(self):
        generator = ScriptedGenerator('["Project Nimbus", "Article 40"]')

        assert derive_search_terms(generator, 'q') == [
            'Project Nimbus', 'Article 40'
        ]

    def test_it_reads_json_out_of_surrounding_prose(self):
        generator = ScriptedGenerator(
            'Here are the terms:\n```json\n["Nimbus"]\n```\nHope that helps.'
        )

        assert derive_search_terms(generator, 'q') == ['Nimbus']

    def test_a_model_failure_sits_the_leg_out(self):
        '''
        Fail soft. The semantic leg still answers, so an unavailable model
        degrades retrieval rather than breaking the query.
        '''
        generator = ScriptedGenerator(error=RuntimeError('provider refused'))

        assert derive_search_terms(generator, 'q') == []

    def test_an_unparseable_reply_sits_the_leg_out(self):
        assert derive_search_terms(ScriptedGenerator('no json here'), 'q') == []

    def test_a_reply_that_is_not_a_list_sits_the_leg_out(self):
        assert derive_search_terms(ScriptedGenerator('{"a": 1}'), 'q') == []

    def test_an_empty_list_is_a_valid_answer(self):
        '''
        A question naming nothing exact should yield nothing. A bad term is
        worse than no term: it matches everything and buries what matters.
        '''
        assert derive_search_terms(ScriptedGenerator('[]'), 'q') == []

    def test_the_ceiling_is_enforced_whatever_the_model_returns(self):
        many = '[' + ','.join(f'"term{i}"' for i in range(50)) + ']'

        assert len(derive_search_terms(ScriptedGenerator(many), 'q')) == MAX_TERMS

    def test_the_question_reaches_the_model_unchanged(self):
        generator = ScriptedGenerator('[]')
        derive_search_terms(generator, '¿Qué dijo @acct_1?')

        assert generator.prompts[0][1] == '¿Qué dijo @acct_1?'

    def test_the_prompt_names_no_language(self):
        generator = ScriptedGenerator('[]')
        derive_search_terms(generator, 'q')
        system = generator.prompts[0][0].lower()

        for language in ('english', 'spanish', 'in english', 'stopword'):
            assert language not in system


class TestParsing:
    @pytest.mark.parametrize('reply, expected', [
        ('["a", "b"]', ['a', 'b']),
        ('  ["a"]  ', ['a']),
        ('text ["a"] text', ['a']),
        ('["a", "", "  "]', ['a']),
        ('["a", 3, null]', ['a']),
        ('', []),
        ('[]', []),
        ('[unclosed', [])
    ])
    def test_it_reads_what_it_can_and_drops_the_rest(self, reply, expected):
        assert _parse_terms(reply) == expected


class TestRestriction:
    def test_it_can_be_limited_to_documents(self, project):
        results = lexical_search(
            project, ['@acct_1'], embedding_model=MODEL,
            refs=['material/beta.md']
        )

        assert all(r.ref == 'material/beta.md' for r in results)

    def test_an_empty_restriction_returns_nothing(self, project):
        assert lexical_search(project, ['@acct_1'], refs=[]) == []

    def test_a_project_with_no_store_returns_nothing(self, tmp_path):
        empty = Project.create('Empty', home=tmp_path)

        assert lexical_search(empty, ['anything']) == []
