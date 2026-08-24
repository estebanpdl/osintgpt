# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_evaluation.py
# Description: The harness that makes a retrieval claim checkable. Its own
#   correctness matters more than most: a wrong scorer would make every
#   measurement taken with it wrong in the same direction.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, Question, evaluate, index_project
from osintgpt.evaluation import (
    EvaluationReport,
    QuestionResult,
    load_questions,
    save_questions
)
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider


class WordEmbedder(EmbeddingProvider):
    '''
    A vector per word present, so a query matches text sharing its words.
    Deterministic, offline, and enough to rank documents predictably.
    '''
    model = 'test-embedding'

    VOCABULARY = (
        'aardvark zebra quokka enforcement propaganda timeline '
        'infrastructure contamination'
    ).split()

    def embed(self, texts):
        return [self._vector(text) for text in texts]

    def _vector(self, text):
        lowered = text.lower()
        counts = [float(lowered.count(word)) for word in self.VOCABULARY]
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


@pytest.fixture
def embedder():
    return WordEmbedder()


@pytest.fixture
def project(tmp_path):
    '''Three documents, each about something different.'''
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()

    (material / 'alpha.md').write_text(
        '# Alpha\n\nA report about aardvark enforcement.', encoding='utf-8'
    )
    (material / 'beta.md').write_text(
        '# Beta\n\nA report about zebra infrastructure.', encoding='utf-8'
    )
    (material / 'gamma.md').write_text(
        '# Gamma\n\nA report about quokka contamination.', encoding='utf-8'
    )

    Corpus.load(instance.paths.sources).register('material')

    return instance


@pytest.fixture
def indexed(project, embedder):
    index_project(project, embedder)

    return project


class TestScoring:
    def test_a_question_whose_document_ranks_first(self, indexed, embedder):
        report = evaluate(
            indexed,
            [Question('aardvark', ['material/alpha.md'])],
            embedder
        )

        assert report.found == 1
        assert report.results[0].first_hit == 1
        assert report.hit_rate == 1.0

    def test_a_question_nothing_answers(self, indexed, embedder):
        '''
        top_k=1 because a small corpus returns everything at depth ten, and a
        document the reader never sees is what a miss actually means.
        '''
        report = evaluate(
            indexed,
            [Question('zebra', ['material/gamma.md'])],
            embedder,
            top_k=1
        )

        assert report.found == 0
        assert report.results[0].first_hit is None
        assert report.hit_rate == 0.0

    def test_a_shallow_corpus_returns_everything(self, indexed, embedder):
        '''
        Worth pinning: with three documents and a depth of ten, every
        question "finds" its answer. A hit rate is only meaningful when the
        corpus is deeper than the cut.
        '''
        report = evaluate(
            indexed,
            [Question('zebra', ['material/gamma.md'])],
            embedder
        )

        assert report.found == 1
        assert report.results[0].first_hit > 1

    def test_rank_is_where_the_first_expected_document_landed(
        self, indexed, embedder
    ):
        report = evaluate(
            indexed,
            [Question('aardvark zebra', ['material/beta.md'])],
            embedder
        )

        assert report.results[0].first_hit in (1, 2)

    def test_reciprocal_rank_rewards_finding_it_early(self):
        first = QuestionResult(Question('q', ['a']), ['a'], first_hit=1)
        fourth = QuestionResult(Question('q', ['a']), ['x', 'y', 'z', 'a'], 4)

        assert first.reciprocal_rank == 1.0
        assert fourth.reciprocal_rank == 0.25

    def test_a_miss_scores_zero_rather_than_undefined(self):
        missed = QuestionResult(Question('q', ['a']), ['x'], first_hit=None)

        assert missed.reciprocal_rank == 0.0
        assert missed.recall == 0.0

    def test_recall_counts_every_expected_document(self):
        partial = QuestionResult(
            Question('q', ['a', 'b']), ['a', 'x'], first_hit=1
        )

        assert partial.recall == 0.5


class TestDocumentLevelRanking:
    def test_several_chunks_of_one_document_count_once(
        self, project, embedder
    ):
        '''
        What is scored is whether the document was reached. A document that
        fills the top ten with its own chunks has been found once, not ten
        times, and must not crowd out the measurement.
        '''
        long_body = 'A paragraph about aardvark enforcement. ' * 80
        (project.paths.root / 'material' / 'alpha.md').write_text(
            f'# Alpha\n\n{long_body}', encoding='utf-8'
        )
        index_project(project, embedder, force=True)

        report = evaluate(
            project, [Question('aardvark', ['material/alpha.md'])], embedder
        )

        assert report.results[0].retrieved.count('material/alpha.md') == 1


class TestUnscorable:
    def test_a_question_with_no_expected_documents(self, indexed, embedder):
        '''
        A question nobody can score is reported rather than counted, or it
        would drag the hit rate down as though retrieval had failed.
        '''
        report = evaluate(indexed, [Question('aardvark', [])], embedder)

        assert report.scored == 0
        assert len(report.unscorable) == 1

    def test_an_expected_document_the_store_has_never_held(
        self, indexed, embedder
    ):
        report = evaluate(
            indexed,
            [Question('aardvark', ['material/typo.md'])],
            embedder,
            known_refs=['material/alpha.md']
        )

        assert report.scored == 0
        assert 'not in the store' in report.unscorable[0]

    def test_without_known_refs_nothing_is_checked(self, indexed, embedder):
        '''
        The check is opt-in: a caller that does not pass the store's refs is
        scoring against whatever it believes, which is its own business.
        '''
        report = evaluate(
            indexed, [Question('aardvark', ['material/typo.md'])], embedder
        )

        assert report.scored == 1
        assert report.found == 0

    def test_the_summary_names_them(self, indexed, embedder):
        report = evaluate(
            indexed,
            [Question('aardvark', ['material/alpha.md']),
             Question('unscorable', [])],
            embedder
        )

        assert 'unscorable' in report.summary


class TestAggregates:
    @pytest.fixture
    def report(self, indexed, embedder):
        return evaluate(
            indexed,
            [
                Question('aardvark', ['material/alpha.md']),
                Question('zebra', ['material/beta.md']),
                Question('quokka', ['material/gamma.md'])
            ],
            embedder
        )

    def test_hit_rate_counts_questions_answered_at_all(self, report):
        assert 0.0 <= report.hit_rate <= 1.0
        assert report.hit_rate == report.found / report.scored

    def test_mrr_is_the_mean_of_the_reciprocal_ranks(self, report):
        expected = sum(r.reciprocal_rank for r in report.results) / 3

        assert report.mean_reciprocal_rank == pytest.approx(expected)

    def test_misses_are_listed_for_reading(self, indexed, embedder):
        '''
        An aggregate says a change helped; the misses say what it still
        cannot answer, which is the more useful half.
        '''
        report = evaluate(
            indexed,
            [Question('zebra', ['material/gamma.md'])],
            embedder,
            top_k=1
        )

        assert len(report.misses) == 1
        assert report.misses[0].question.text == 'zebra'

    def test_an_empty_set_scores_nothing_rather_than_dividing(self):
        empty = EvaluationReport()

        assert empty.hit_rate == 0.0
        assert empty.mean_reciprocal_rank == 0.0
        assert empty.summary == 'nothing scored'


class TestComparison:
    def test_it_reports_what_moved(self):
        before = EvaluationReport(results=[
            QuestionResult(Question('q', ['a']), ['x', 'a'], first_hit=2)
        ])
        after = EvaluationReport(results=[
            QuestionResult(Question('q', ['a']), ['a'], first_hit=1)
        ])

        moved = after.against(before)

        assert 'MRR' in moved
        assert '+' in moved

    def test_a_regression_reads_as_one(self):
        before = EvaluationReport(results=[
            QuestionResult(Question('q', ['a']), ['a'], first_hit=1)
        ])
        after = EvaluationReport(results=[
            QuestionResult(Question('q', ['a']), ['x'], first_hit=None)
        ])

        moved = after.against(before)

        assert '-' in moved

    def test_no_change_says_so(self):
        report = EvaluationReport(results=[
            QuestionResult(Question('q', ['a']), ['a'], first_hit=1)
        ])

        assert report.against(report) == 'no change'


class TestQuestionSet:
    def test_it_round_trips(self, tmp_path):
        path = tmp_path / 'questions.toml'
        questions = [
            Question('a question', ['a.md', 'b.md'], note='why it is here'),
            Question('another', ['c.md'])
        ]
        save_questions(path, questions)

        loaded = load_questions(path)

        assert loaded == questions

    def test_an_absent_file_is_an_empty_set(self, tmp_path):
        assert load_questions(tmp_path / 'nothing.toml') == []

    def test_the_file_says_what_makes_a_good_set(self, tmp_path):
        path = tmp_path / 'questions.toml'
        save_questions(path, [Question('a question', ['a.md'])])

        text = path.read_text(encoding='utf-8')

        assert 'Keep them real' in text
        assert 'measures nothing' in text


class TestTopK:
    def test_a_document_below_the_cut_is_a_miss(self, indexed, embedder):
        '''
        A passage a reading model never sees has not been retrieved, whatever
        rank it holds in a longer list.
        '''
        report = evaluate(
            indexed,
            [Question('zebra aardvark quokka', ['material/gamma.md'])],
            embedder,
            top_k=1
        )

        assert report.top_k == 1
        assert len(report.results[0].retrieved) <= 1

    def test_it_is_recorded_with_the_report(self, indexed, embedder):
        report = evaluate(
            indexed, [Question('aardvark', ['material/alpha.md'])],
            embedder, top_k=5
        )

        assert report.top_k == 5
        assert report.embedding_model == embedder.model
