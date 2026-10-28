"""Cosine ranking guarantees through the modern store contract."""

import math
import random

import pytest

from osintgpt.vector_store import SQLiteVectorStore, StoredChunk


def test_ranking_matches_per_record_cosine(tmp_path):
    rng = random.Random(19)
    vectors = [[rng.uniform(-1, 1) for _ in range(12)] for _ in range(30)]
    query = [rng.uniform(-1, 1) for _ in range(12)]
    scores = [sum(a * b for a, b in zip(query, vector)) /
              (math.sqrt(sum(x * x for x in query)) * math.sqrt(sum(x * x for x in vector)))
              for vector in vectors]
    expected = sorted(range(len(vectors)), key=scores.__getitem__, reverse=True)
    chunks = [StoredChunk(ref='source', sequence=i, text=f'record {i}', embedding_model='synthetic')
              for i in range(len(vectors))]
    with SQLiteVectorStore(tmp_path / 'store.sqlite') as store:
        store.upsert('source', chunks, vectors)
        results = store.search(query, 'synthetic', top_k=len(vectors))
    assert [result.chunk.sequence for result in results] == expected
    assert [result.score for result in results] == pytest.approx([scores[i] for i in expected])


def test_zero_vector_scores_zero(tmp_path):
    chunk = StoredChunk(ref='zero', sequence=0, text='zero vector', embedding_model='synthetic')
    with SQLiteVectorStore(tmp_path / 'store.sqlite') as store:
        store.upsert('zero', [chunk], [[0.0, 0.0]])
        result, = store.search([1.0, 0.0], 'synthetic')
    assert math.isfinite(result.score)
    assert result.score == pytest.approx(0.0)
