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
PRICES_UPDATED = '2026-09-02'

# Chat models carry separate input and output prices; embedding models bill
# input only.
PRICES = {
    # OpenAI
    'gpt-4o':                     {'input': 2.50,  'output': 10.00},
    'gpt-4o-mini':                {'input': 0.15,  'output': 0.60},
    'gpt-4.1':                    {'input': 2.00,  'output': 8.00},
    'gpt-4.1-mini':               {'input': 0.40,  'output': 1.60},
    'gpt-4.1-nano':               {'input': 0.10,  'output': 0.40},
    'gpt-4-turbo':                {'input': 10.00, 'output': 30.00},
    'gpt-3.5-turbo':              {'input': 0.50,  'output': 1.50},
    'gpt-5.6-sol':                {'input': 4.00,  'output': 20.00},
    'gpt-5.6-terra':              {'input': 2.00,  'output': 12.00},
    'gpt-5.6-luna':               {'input': 0.20,  'output': 1.20},
    'gpt-5':                      {'input': 1.25,  'output': 10.00},
    'gpt-5-mini':                 {'input': 0.25,  'output': 2.00},
    'gpt-5-nano':                 {'input': 0.05,  'output': 0.40},
    'o1':                         {'input': 15.00, 'output': 60.00},
    'o3':                         {'input': 2.00,  'output': 8.00},
    'o3-mini':                    {'input': 1.10,  'output': 4.40},
    'o4-mini':                    {'input': 1.10,  'output': 4.40},
    'text-embedding-3-small':     {'input': 0.02},
    'text-embedding-3-large':     {'input': 0.13},
    'text-embedding-ada-002':     {'input': 0.10},

    # Anthropic (generation only)
    'claude-fable-5-1':           {'input': 10.00, 'output': 50.00},
    'claude-fable-5':             {'input': 10.00, 'output': 50.00},
    'claude-opus-5':              {'input': 5.00,  'output': 25.00},
    'claude-opus-4-8':            {'input': 5.00,  'output': 25.00},
    'claude-opus-4-7':            {'input': 5.00,  'output': 25.00},
    'claude-opus-4-6':            {'input': 5.00,  'output': 25.00},
    'claude-opus-4-5-20251101':   {'input': 5.00,  'output': 25.00},
    'claude-sonnet-5':            {'input': 2.00,  'output': 10.00},
    'claude-sonnet-4-6':          {'input': 3.00,  'output': 15.00},
    'claude-sonnet-4-5-20250929': {'input': 3.00,  'output': 15.00},
    'claude-haiku-4-5-20251001':  {'input': 1.00,  'output': 5.00},

    # Gemini. gemini-2.5-pro's input/output here are its <=200k-token tier;
    # the >200k tier (2.50 / 15.00) is not represented.
    'gemini-2.5-pro':             {'input': 1.25,  'output': 10.00},
    'gemini-2.5-flash':           {'input': 0.30,  'output': 2.50},
    'gemini-2.5-flash-lite':      {'input': 0.10,  'output': 0.40},
    'gemini-embedding-001':       {'input': 0.15},

    # Voyage (embeddings only)
    'voyage-4-large':             {'input': 0.12},
    'voyage-4':                   {'input': 0.06},
    'voyage-4-lite':              {'input': 0.02},
    'voyage-context-4':           {'input': 0.12},
    'voyage-code-4':              {'input': 0.12},
    'voyage-3-large':             {'input': 0.18},
    'voyage-code-3':              {'input': 0.18},
    'voyage-context-3':           {'input': 0.18},
    'voyage-3.5':                 {'input': 0.06},
    'voyage-3.5-lite':            {'input': 0.02},
    'voyage-3':                   {'input': 0.06},
    'voyage-3-lite':              {'input': 0.02},
    'voyage-finance-2':           {'input': 0.12},
    'voyage-law-2':               {'input': 0.12},
    'voyage-code-2':              {'input': 0.12}
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
