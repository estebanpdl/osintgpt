# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_vector_store.py
# Description: What is true of the SQLite store and not of every store. The
#   behaviour both backends share lives in test_store_contract.py.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt vector store
from osintgpt.vector_store import (
    BRUTE_FORCE_CEILING,
    SQLiteVectorStore,
    StoredChunk
)

MODEL = 'text-embedding-3-small'


def chunk(ref, sequence=0, text='some text', model=MODEL, **fields):
    return StoredChunk(
        ref=ref, sequence=sequence, text=text, embedding_model=model, **fields
    )


def unit(*values):
    length = math.sqrt(sum(v * v for v in values)) or 1.0

    return [v / length for v in values]


@pytest.fixture
def store():
    with SQLiteVectorStore(':memory:') as instance:
        yield instance


class TestOneFilePerProject:
    def test_a_file_store_creates_its_parent(self, tmp_path):
        path = tmp_path / 'project' / 'store.sqlite'

        with SQLiteVectorStore(path):
            assert path.exists()

    def test_a_store_survives_reopening(self, tmp_path):
        path = tmp_path / 'store.sqlite'

        with SQLiteVectorStore(path) as first:
            first.upsert('a.md', [chunk('a.md', text='kept')], [unit(1, 0)])

        with SQLiteVectorStore(path) as second:
            assert second.count() == 1
            assert second.search(unit(1, 0), MODEL)[0].text == 'kept'

    def test_two_projects_do_not_share_a_store(self, tmp_path):
        '''
        Isolation is structural rather than a filter: two projects holding a
        document at the same path cannot see each other's copy.
        '''
        with SQLiteVectorStore(tmp_path / 'a' / 'store.sqlite') as a, \
             SQLiteVectorStore(tmp_path / 'b' / 'store.sqlite') as b:
            a.upsert('doc.md', [chunk('doc.md', text='alpha')], [unit(1, 0)])
            b.upsert('doc.md', [chunk('doc.md', text='beta')], [unit(1, 0)])

            assert a.search(unit(1, 0), MODEL)[0].text == 'alpha'
            assert b.search(unit(1, 0), MODEL)[0].text == 'beta'


class TestNumPyRanking:
    def test_a_zero_vector_matches_nothing_rather_than_dividing(self, store):
        '''
        A zero-length vector has no direction. The cosine is one matrix
        multiply over every row, so a zero norm would divide by zero and
        return NaN rather than raising.
        '''
        store.upsert('a.md', [chunk('a.md')], [[0.0, 0.0]])

        assert store.search(unit(1, 0), MODEL)[0].score == pytest.approx(0.0)

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
        '''
        Past this the store still answers and a dedicated backend is the
        better one. The number exists so that is a decision rather than a
        surprise.
        '''
        assert BRUTE_FORCE_CEILING > 10_000
        assert store.is_past_brute_force is False
