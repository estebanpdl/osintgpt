# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_pricing.py
# Description: The price table and the estimator over it. The property that
#   matters is that an unpriced model estimates to None, never to zero.
# =================================================================================

# import modules
import datetime
import pytest

# import osintgpt pricing
from osintgpt.pricing import (
    PRICES,
    PRICES_UPDATED,
    TOKENS_PER_UNIT,
    estimate_cost,
    price_per_million,
    priced_models
)


class TestTable:
    def test_every_model_carries_an_input_price(self):
        for model, prices in PRICES.items():
            assert 'input' in prices, f'{model} has no input price'

    def test_prices_are_positive_numbers(self):
        for model, prices in PRICES.items():
            for kind, value in prices.items():
                assert isinstance(value, (int, float)), f'{model}.{kind}'
                assert value > 0, f'{model}.{kind}'

    def test_updated_stamp_is_a_real_date(self):
        datetime.date.fromisoformat(PRICES_UPDATED)

    def test_priced_models_lists_the_table(self):
        assert priced_models() == sorted(PRICES)


class TestLookup:
    def test_returns_the_input_price(self):
        assert price_per_million('gpt-4o') == PRICES['gpt-4o']['input']

    def test_returns_the_output_price(self):
        assert price_per_million('gpt-4o', 'output') == PRICES['gpt-4o']['output']

    def test_embedding_models_have_no_output_price(self):
        assert price_per_million('text-embedding-3-small', 'output') is None

    def test_unknown_model_is_none(self):
        assert price_per_million('gpt-does-not-exist') is None


class TestEstimate:
    def test_one_unit_costs_the_unit_price(self):
        assert estimate_cost('gpt-4o', TOKENS_PER_UNIT) == PRICES['gpt-4o']['input']

    def test_scales_linearly(self):
        single = estimate_cost('gpt-4o', 1_000)
        double = estimate_cost('gpt-4o', 2_000)

        assert double == pytest.approx(single * 2)

    def test_zero_tokens_cost_nothing(self):
        assert estimate_cost('gpt-4o', 0) == 0.0

    def test_unknown_model_is_none_not_zero(self):
        result = estimate_cost('gpt-does-not-exist', TOKENS_PER_UNIT)

        assert result is None
        assert result != 0
