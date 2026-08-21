# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: pricing.py
# Description: The pricing.py file contains the model price table and the cost
#   estimator built on it. Prices are indicative; an unpriced model estimates to
#   None rather than to a confident wrong number.
# =================================================================================

# type hints
from typing import Optional

# Published list prices in USD per 1M tokens, as of PRICES_UPDATED. This table
# is a convenience for pre-flight estimates, never a billing record — providers
# reprice without notice, so treat a figure older than the date below as stale
# and verify against the provider before relying on it.
PRICES_UPDATED = '2026-08-21'

# Chat models carry separate input and output prices; embedding models bill
# input only.
PRICES = {
    'gpt-4o':                 {'input': 2.50,  'output': 10.00},
    'gpt-4o-mini':            {'input': 0.15,  'output': 0.60},
    'gpt-4.1':                {'input': 2.00,  'output': 8.00},
    'gpt-4.1-mini':           {'input': 0.40,  'output': 1.60},
    'gpt-4.1-nano':           {'input': 0.10,  'output': 0.40},
    'gpt-4-turbo':            {'input': 10.00, 'output': 30.00},
    'gpt-3.5-turbo':          {'input': 0.50,  'output': 1.50},
    'text-embedding-3-small': {'input': 0.02},
    'text-embedding-3-large': {'input': 0.13},
    'text-embedding-ada-002': {'input': 0.10}
}

# tokens per pricing unit
TOKENS_PER_UNIT = 1_000_000

# get the price for a model
def price_per_million(model: str, kind: str = 'input') -> Optional[float]:
    '''
    Look up a model's price.

    Args:
        model (str): Model name.
        kind (str): 'input' or 'output'.

    Returns:
        Optional[float]: USD per 1M tokens, or None when the model or the \
            direction is not priced.
    '''
    return PRICES.get(model, {}).get(kind)

# estimate the cost of a number of tokens
def estimate_cost(model: str, tokens: int,
    kind: str = 'input') -> Optional[float]:
    '''
    Estimate what a number of tokens costs on a model.

    Args:
        model (str): Model name.
        tokens (int): Token count.
        kind (str): 'input' or 'output'.

    Returns:
        Optional[float]: Estimated USD, or None when the model is not priced. \
            None means unknown — callers must not treat it as zero.
    '''
    price = price_per_million(model, kind)
    if price is None:
        return None

    return (tokens / TOKENS_PER_UNIT) * price

# list the priced models
def priced_models() -> list:
    '''
    Every model the table carries a price for.

    Returns:
        list: Sorted model names.
    '''
    return sorted(PRICES)
