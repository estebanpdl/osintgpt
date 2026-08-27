# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_dataframe_search.py
# Description: Similarity search over embeddings held in a DataFrame.
# =================================================================================

import math
import random

import pandas as pd
import pytest

from osintgpt.llms.search import SearchMixin


def unit(*values):
    length = math.sqrt(sum(value * value for value in values)) or 1.0

    return [value / length for value in values]


def search(vectors, query, top_k=10):
    frame = pd.DataFrame({
        'embeddings': vectors,
        'text': [f'record {i}' for i in range(len(vectors))]
    })

    return SearchMixin().search_results_from_dataframe(
        frame, embeddings=query, top_k=top_k
    )


def test_ranking_matches_per_row_cosine():
    generator = random.Random(19)
    vectors = [
        unit(*[generator.uniform(-1, 1) for _ in range(12)])
        for _ in range(30)
    ]
    query = unit(*[generator.uniform(-1, 1) for _ in range(12)])
    expected = sorted(
        range(len(vectors)),
        key=lambda i: sum(a * b for a, b in zip(query, vectors[i])),
        reverse=True
    )

    result = search(vectors, query, top_k=len(vectors))

    assert [text for _, text, _ in result['results']] == [
        f'record {i}' for i in expected
    ]


def test_scalar_relatedness_remains_available():
    score = SearchMixin()._relatedness_fn(unit(1, 1), unit(1, 1))

    assert score == pytest.approx(1.0)


def test_an_empty_dataframe_returns_no_results():
    result = search([], unit(1, 0))

    assert result['results'] == []


def test_a_single_row_dataframe_can_be_searched():
    vector = unit(2, 3, 4)

    result = search([vector], vector)

    assert result['results'] == [(vector, 'record 0', pytest.approx(1.0))]


def test_a_zero_vector_scores_zero():
    result = search([[0.0, 0.0], unit(1, 0)], unit(0, 1))

    zero_result = next(item for item in result['results'] if item[0] == [0.0, 0.0])
    assert zero_result[2] == pytest.approx(0.0)
    assert math.isnan(zero_result[2]) is False


def test_top_k_larger_than_the_dataframe_returns_every_row():
    result = search([unit(1, 0), unit(0, 1), unit(-1, 0)], unit(1, 0), top_k=20)

    assert len(result['results']) == 3


def test_the_published_result_shape_is_unchanged():
    vector = unit(1, 2)

    result = search([vector], vector)

    assert list(result) == ['query', 'query_embedding', 'results']
    assert result['query'] is None
    assert result['query_embedding'] == vector
    embedding, text, score = result['results'][0]
    assert embedding == vector
    assert text == 'record 0'
    assert score == pytest.approx(1.0)


def test_results_are_ordered_best_first():
    result = search(
        [unit(-1, 0), unit(1, 1), unit(1, 0)], unit(1, 0)
    )

    scores = [score for _, _, score in result['results']]
    assert scores == sorted(scores, reverse=True)
