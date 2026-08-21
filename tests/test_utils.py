# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_utils.py
# Description: Token counting and id generation. Encodings differ between
#   models, so counting for the wrong model is silently wrong — pinned here.
# =================================================================================

# import modules
import pytest

# import utils
from osintgpt.utils import (
    DEFAULT_ENCODING,
    count_tokens,
    create_unique_id,
    encoding_for_model
)

# Mixed scripts, because a multilingual corpus is the point of the tool.
SAMPLE = 'OSINT analysis of multilingual narratives — 分析 — análisis'


class TestEncodingForModel:
    def test_resolves_a_known_chat_model(self):
        assert encoding_for_model('gpt-4o').name == 'o200k_base'

    def test_resolves_a_known_embedding_model(self):
        assert encoding_for_model('text-embedding-3-small').name == 'cl100k_base'

    def test_chat_and_embedding_encodings_differ(self):
        '''Why count_tokens takes a model rather than assuming one.'''
        chat = encoding_for_model('gpt-4o')
        embedding = encoding_for_model('text-embedding-3-small')

        assert chat.name != embedding.name
        assert len(chat.encode(SAMPLE)) != len(embedding.encode(SAMPLE))

    def test_unknown_model_falls_back(self):
        '''A model newer than the installed tiktoken must not raise.'''
        assert encoding_for_model('gpt-99-unreleased').name == DEFAULT_ENCODING


class TestCountTokens:
    def test_counts_with_the_named_model(self):
        expected = len(encoding_for_model('gpt-4o').encode(SAMPLE))

        assert count_tokens(SAMPLE, 'gpt-4o') == expected

    def test_differs_between_models(self):
        chat = count_tokens(SAMPLE, 'gpt-4o')
        embedding = count_tokens(SAMPLE, 'text-embedding-3-small')

        assert chat != embedding

    def test_empty_string_is_zero(self):
        assert count_tokens('', 'gpt-4o') == 0

    def test_unknown_model_still_counts(self):
        assert count_tokens(SAMPLE, 'gpt-99-unreleased') > 0


class TestCreateUniqueId:
    def test_returns_a_plain_hex_string(self):
        unique_id = create_unique_id()

        assert len(unique_id) == 32
        assert '-' not in unique_id
        int(unique_id, 16)

    def test_avoids_ids_already_taken(self):
        first = create_unique_id()
        second = create_unique_id([first])

        assert second != first

    def test_successive_ids_differ(self):
        assert create_unique_id() != create_unique_id()
