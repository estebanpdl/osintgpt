# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_fusion.py
# Description: Combining legs that score on different scales. The property
#   that matters is that agreement between legs outranks depth in one.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, hybrid_search, index_project
from osintgpt.fusion import RRF_K, reciprocal_rank_fusion
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.vector_store import SearchResult, StoredChunk

MODEL = 'test-embedding'


def result(ref, sequence=0, text='text', score=0.5):
    return SearchResult(
        chunk=StoredChunk(
            ref=ref, sequence=sequence, text=text, embedding_model=MODEL
        ),
        score=score
    )


class WordEmbedder(EmbeddingProvider):
    model = MODEL
    VOCABULARY = 'aardvark zebra quokka narwhal'.split()

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        low = text.lower()
        counts = [float(low.count(w)) for w in self.VOCABULARY]
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


class TermGenerator(GenerationProvider):
    model = 'test-generation'

    def __init__(self, reply='["@acct_1"]'):
        self.reply = reply
        self.calls = 0

    def generate(self, system, user, **kwargs):
        self.calls += 1

        return self.reply


class TestAgreementWins:
    '''
    The property that makes fusion worth doing: a passage both legs return
    beats one either found deeply on its own.
    '''

    def test_a_chunk_both_legs_found_outranks_one_leg_s_favourite(self):
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md'), result('b.md')],
            'lexical': [result('c.md'), result('b.md')]
        })

        assert fused[0].ref == 'b.md'
        assert fused[0].found_by_all is True

    def test_it_records_where_each_leg_placed_a_chunk(self):
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md'), result('b.md')],
            'lexical': [result('b.md')]
        })
        best = next(f for f in fused if f.ref == 'b.md')

        assert best.ranks == {'semantic': 2, 'lexical': 1}

    def test_legs_are_listed_best_first(self):
        fused = reciprocal_rank_fusion({
            'semantic': [result('x.md'), result('x.md', 1), result('b.md')],
            'lexical': [result('b.md')]
        })
        best = next(f for f in fused if f.ref == 'b.md')

        assert best.legs == ['lexical', 'semantic']

    def test_a_chunk_only_one_leg_found_still_appears(self):
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md')],
            'lexical': [result('b.md')]
        })

        assert {f.ref for f in fused} == {'a.md', 'b.md'}


class TestRankNotScore:
    '''
    Scores from different legs are not comparable. A cosine of 0.62 and a term
    coverage of 0.5 measure different things, and normalizing them would
    invent a relationship that is not there.
    '''

    def test_the_incoming_scores_do_not_change_the_order(self):
        low = reciprocal_rank_fusion({
            'semantic': [result('a.md', score=0.01), result('b.md', score=0.005)]
        })
        high = reciprocal_rank_fusion({
            'semantic': [result('a.md', score=0.99), result('b.md', score=0.98)]
        })

        assert [f.ref for f in low] == [f.ref for f in high]

    def test_a_leg_scoring_everything_the_same_still_ranks(self):
        fused = reciprocal_rank_fusion({
            'lexical': [result('a.md', score=1.0), result('b.md', score=1.0)]
        })

        assert [f.ref for f in fused] == ['a.md', 'b.md']

    def test_position_one_scores_more_than_position_two(self):
        fused = reciprocal_rank_fusion({'semantic': [result('a.md'),
                                                     result('b.md')]})

        assert fused[0].score > fused[1].score

    def test_the_damping_constant_is_the_published_one(self):
        assert RRF_K == 60


class TestWeights:
    def test_a_weighted_leg_contributes_more(self):
        fused = reciprocal_rank_fusion(
            {'semantic': [result('a.md')], 'lexical': [result('b.md')]},
            weights={'lexical': 5.0}
        )

        assert fused[0].ref == 'b.md'

    def test_an_unweighted_leg_counts_once(self):
        plain = reciprocal_rank_fusion({'semantic': [result('a.md')]})
        weighted = reciprocal_rank_fusion(
            {'semantic': [result('a.md')]}, weights={'semantic': 1.0}
        )

        assert plain[0].score == weighted[0].score


class TestEdges:
    def test_no_legs_fuse_to_nothing(self):
        assert reciprocal_rank_fusion({}) == []

    def test_an_empty_leg_is_not_an_error(self):
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md')], 'lexical': []
        })

        assert [f.ref for f in fused] == ['a.md']

    def test_a_leg_returning_a_chunk_twice_counts_it_once(self):
        '''Otherwise a duplicate would promote itself.'''
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md'), result('a.md'), result('b.md')]
        })

        assert len(fused) == 2

    def test_chunks_of_one_document_stay_distinct(self):
        '''
        A document contributes several passages, and fusing them into one
        would lose the passage a reader needs.
        '''
        fused = reciprocal_rank_fusion({
            'semantic': [result('a.md', 0), result('a.md', 1)]
        })

        assert len(fused) == 2

    def test_the_limit_is_respected(self):
        fused = reciprocal_rank_fusion(
            {'semantic': [result(f'{i}.md') for i in range(20)]}, limit=3
        )

        assert len(fused) == 3

    def test_the_order_is_deterministic(self):
        legs = {
            'semantic': [result('b.md'), result('a.md')],
            'lexical': [result('a.md'), result('b.md')]
        }

        assert [f.ref for f in reciprocal_rank_fusion(legs)] == [
            f.ref for f in reciprocal_rank_fusion(legs)
        ]


class TestHybridSearch:
    @pytest.fixture
    def project(self, tmp_path):
        instance = Project.create('Case', home=tmp_path)
        material = instance.paths.root / 'material'
        material.mkdir()
        (material / 'alpha.md').write_text(
            '# Alpha\n\nA report about aardvark sightings near @acct_1.',
            encoding='utf-8'
        )
        (material / 'beta.md').write_text(
            '# Beta\n\nA report about zebra migration.', encoding='utf-8'
        )
        Corpus.load(instance.paths.sources).register('material')
        index_project(instance, WordEmbedder())

        return instance

    def test_both_legs_run_when_a_generator_is_given(self, project):
        generator = TermGenerator('["@acct_1"]')
        fused = hybrid_search(project, 'aardvark', WordEmbedder(), generator)

        assert generator.calls == 1
        assert any(f.found_by_all for f in fused)

    def test_without_a_generator_it_degrades_to_semantic(self, project):
        '''
        Missing terms sit the lexical leg out rather than failing the search.
        '''
        fused = hybrid_search(project, 'aardvark', WordEmbedder())

        assert fused
        assert all(list(f.ranks) == ['semantic'] for f in fused)

    def test_explicit_terms_skip_derivation(self, project):
        generator = TermGenerator()
        hybrid_search(
            project, 'aardvark', WordEmbedder(), generator, terms=['@acct_1']
        )

        assert generator.calls == 0

    def test_empty_terms_sit_the_lexical_leg_out_deliberately(self, project):
        generator = TermGenerator()
        fused = hybrid_search(
            project, 'aardvark', WordEmbedder(), generator, terms=[]
        )

        assert generator.calls == 0
        assert all('lexical' not in f.ranks for f in fused)

    def test_a_term_only_the_lexical_leg_can_find_reaches_the_results(
        self, project
    ):
        '''
        `@acct_1` is not in the embedder's vocabulary, so semantic cannot
        distinguish it. This is the case tri-retrieval exists for.
        '''
        fused = hybrid_search(
            project, 'unrelated question', WordEmbedder(),
            terms=['@acct_1']
        )
        found = next(f for f in fused if 'alpha' in f.ref)

        assert 'lexical' in found.ranks

    def test_it_can_be_restricted_to_documents(self, project):
        fused = hybrid_search(
            project, 'aardvark', WordEmbedder(),
            terms=['@acct_1'], refs=['material/beta.md']
        )

        assert all(f.ref == 'material/beta.md' for f in fused)
