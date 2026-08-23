# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_vector_store.py
# Description: The SQLite store — what it keeps, what it refuses to compare,
#   and what it does when a document changes underneath it.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt vector store
from osintgpt.vector_store import (
    BRUTE_FORCE_CEILING,
    BaseVectorEngine,
    SQLiteVectorStore,
    StoredChunk
)

MODEL = 'text-embedding-3-small'
OTHER_MODEL = 'all-MiniLM-L6-v2'


def chunk(ref, sequence=0, text='some text', model=MODEL, **fields):
    return StoredChunk(
        ref=ref, sequence=sequence, text=text, embedding_model=model, **fields
    )


def unit(*values):
    '''A normalized vector, so cosine scores read as similarity.'''
    length = math.sqrt(sum(v * v for v in values)) or 1.0

    return [v / length for v in values]


@pytest.fixture
def store():
    with SQLiteVectorStore(':memory:') as instance:
        yield instance


class TestInterface:
    def test_it_is_a_vector_engine(self, store):
        assert isinstance(store, BaseVectorEngine)

    def test_a_file_store_creates_its_parent(self, tmp_path):
        path = tmp_path / 'project' / 'store.sqlite'

        with SQLiteVectorStore(path):
            assert path.exists()


class TestUpsert:
    def test_chunks_and_vectors_must_match(self, store):
        '''
        They are paired by position, so a mismatch would attach text to the
        wrong vector — wrong answers rather than an error, later.
        '''
        with pytest.raises(ValueError, match='matched by position'):
            store.upsert('a.md', [chunk('a.md')], [unit(1, 0), unit(0, 1)])

    def test_it_stores_what_it_was_given(self, store):
        stored = store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])

        assert stored == 1
        assert store.count() == 1

    def test_re_indexing_replaces_rather_than_appends(self, store):
        '''
        Chunk boundaries move when a document changes, so its old chunks
        describe text that no longer exists.
        '''
        store.upsert(
            'a.md',
            [chunk('a.md', 0, 'first'), chunk('a.md', 1, 'second')],
            [unit(1, 0), unit(0, 1)]
        )
        store.upsert('a.md', [chunk('a.md', 0, 'rewritten')], [unit(1, 1)])

        assert store.count() == 1
        assert store.chunks_for('a.md')[0].text == 'rewritten'

    def test_it_leaves_other_documents_alone(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md')], [unit(0, 1)])
        store.upsert('a.md', [chunk('a.md', text='new')], [unit(1, 1)])

        assert sorted(store.refs()) == ['a.md', 'b.md']

    def test_an_empty_document_clears_its_chunks(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('a.md', [], [])

        assert store.count() == 0


class TestSearch:
    @pytest.fixture
    def populated(self, store):
        store.upsert('a.md', [chunk('a.md', 0, 'about alpha')], [unit(1, 0, 0)])
        store.upsert('b.md', [chunk('b.md', 0, 'about beta')], [unit(0, 1, 0)])
        store.upsert('c.md', [chunk('c.md', 0, 'about gamma')], [unit(0, 0, 1)])

        return store

    def test_the_closest_vector_comes_first(self, populated):
        results = populated.search(unit(1, 0.1, 0), MODEL)

        assert results[0].ref == 'a.md'

    def test_scores_are_similarities(self, populated):
        results = populated.search(unit(1, 0, 0), MODEL)

        assert results[0].score == pytest.approx(1.0)
        assert results[-1].score == pytest.approx(0.0, abs=1e-6)

    def test_results_are_ordered_best_first(self, populated):
        scores = [r.score for r in populated.search(unit(1, 0.5, 0.2), MODEL)]

        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_the_results(self, populated):
        assert len(populated.search(unit(1, 0, 0), MODEL, top_k=2)) == 2

    def test_asking_for_more_than_exists_returns_what_exists(self, populated):
        assert len(populated.search(unit(1, 0, 0), MODEL, top_k=100)) == 3

    def test_it_can_be_restricted_to_documents(self, populated):
        results = populated.search(unit(1, 0, 0), MODEL, refs=['b.md', 'c.md'])

        assert {r.ref for r in results} == {'b.md', 'c.md'}

    def test_an_empty_restriction_returns_nothing(self, populated):
        assert populated.search(unit(1, 0, 0), MODEL, refs=[]) == []

    def test_an_empty_store_returns_nothing(self, store):
        assert store.search(unit(1, 0), MODEL) == []

    def test_a_zero_vector_matches_nothing_rather_than_dividing(self, store):
        store.upsert('a.md', [chunk('a.md')], [[0.0, 0.0]])

        results = store.search(unit(1, 0), MODEL)

        assert results[0].score == pytest.approx(0.0)


class TestModelIsolation:
    '''
    Vectors from different models are not comparable. Comparing them returns
    confident nonsense rather than an error, which is why the filter is not
    optional.
    '''

    @pytest.fixture
    def mixed(self, store):
        store.upsert('a.md', [chunk('a.md', model=MODEL)], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md', model=OTHER_MODEL)], [unit(1, 0)])

        return store

    def test_search_sees_only_its_own_model(self, mixed):
        results = mixed.search(unit(1, 0), MODEL)

        assert [r.ref for r in results] == ['a.md']

    def test_an_unknown_model_finds_nothing(self, mixed):
        assert mixed.search(unit(1, 0), 'never-used') == []

    def test_both_models_are_named(self, mixed):
        assert mixed.models() == sorted([MODEL, OTHER_MODEL])

    def test_counts_can_be_per_model(self, mixed):
        assert mixed.count() == 2
        assert mixed.count(MODEL) == 1

    def test_refs_can_be_per_model(self, mixed):
        assert mixed.refs(MODEL) == ['a.md']

    def test_purging_reclaims_the_leftovers(self, mixed):
        '''
        A model switch leaves vectors that search will never return and the
        store still holds. Naming them is what makes them reclaimable.
        '''
        removed = mixed.purge_other_models(keep=MODEL)

        assert removed == 1
        assert mixed.models() == [MODEL]


class TestDelete:
    def test_it_forgets_a_document(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md')], [unit(0, 1)])

        assert store.delete(['a.md']) == 1
        assert store.refs() == ['b.md']

    def test_deleting_nothing_is_not_an_error(self, store):
        assert store.delete([]) == 0

    def test_deleting_an_absent_document_removes_nothing(self, store):
        assert store.delete(['never-stored.md']) == 0


class TestProvenance:
    def test_everything_a_citation_needs_survives(self, store):
        store.upsert(
            'report.md',
            [chunk(
                'report.md',
                path='Report › Findings',
                timestamp='2026-04-22',
                author='an analyst',
                metadata={'type': 'synthesis'}
            )],
            [unit(1, 0)]
        )

        found = store.search(unit(1, 0), MODEL)[0].chunk

        assert found.path == 'Report › Findings'
        assert found.timestamp == '2026-04-22'
        assert found.author == 'an analyst'
        assert found.metadata == {'type': 'synthesis'}

    def test_a_citation_names_the_section_when_there_is_one(self):
        assert chunk('report.md', path='A › B').citation == 'report.md › A › B'

    def test_a_citation_is_just_the_document_otherwise(self):
        assert chunk('report.md').citation == 'report.md'

    def test_metadata_survives_non_ascii(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', metadata={'título': 'Análisis — 分析'})],
            [unit(1, 0)]
        )

        found = store.search(unit(1, 0), MODEL)[0].chunk

        assert found.metadata['título'] == 'Análisis — 分析'

    def test_chunks_come_back_in_reading_order(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'part {i}') for i in range(5)],
            [unit(1, i) for i in range(5)]
        )

        texts = [c.text for c in store.chunks_for('a.md')]

        assert texts == [f'part {i}' for i in range(5)]


class TestPersistence:
    def test_a_store_survives_reopening(self, tmp_path):
        path = tmp_path / 'store.sqlite'

        with SQLiteVectorStore(path) as first:
            first.upsert('a.md', [chunk('a.md', text='kept')], [unit(1, 0)])

        with SQLiteVectorStore(path) as second:
            assert second.count() == 1
            assert second.search(unit(1, 0), MODEL)[0].text == 'kept'

    def test_two_projects_do_not_share_a_store(self, tmp_path):
        with SQLiteVectorStore(tmp_path / 'a' / 'store.sqlite') as a, \
             SQLiteVectorStore(tmp_path / 'b' / 'store.sqlite') as b:
            a.upsert('doc.md', [chunk('doc.md', text='alpha')], [unit(1, 0)])
            b.upsert('doc.md', [chunk('doc.md', text='beta')], [unit(1, 0)])

            assert a.search(unit(1, 0), MODEL)[0].text == 'alpha'
            assert b.search(unit(1, 0), MODEL)[0].text == 'beta'


class TestScale:
    def test_vectors_round_trip_at_a_realistic_size(self, store):
        '''1536 dimensions is what a current OpenAI embedding returns.'''
        vector = unit(*[float(i % 7) + 0.5 for i in range(1536)])
        store.upsert('a.md', [chunk('a.md')], [vector])

        assert store.search(vector, MODEL)[0].score == pytest.approx(1.0)

    def test_search_holds_up_over_a_few_thousand_chunks(self, store):
        import random

        # Random rather than a cycling pattern: a formula over the index
        # repeats, and identical vectors make "the closest one" ambiguous.
        generator = random.Random(11)
        chunks, vectors = [], []
        for i in range(2_000):
            chunks.append(chunk('corpus.csv', i, f'record {i}'))
            vectors.append(unit(*[generator.random() for _ in range(64)]))
        store.upsert('corpus.csv', chunks, vectors)

        results = store.search(vectors[1_337], MODEL, top_k=1)

        assert results[0].text == 'record 1337'
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_the_brute_force_ceiling_is_documented(self, store):
        assert BRUTE_FORCE_CEILING > 10_000
        assert store.is_past_brute_force is False
