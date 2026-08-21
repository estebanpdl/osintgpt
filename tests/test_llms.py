# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llms.py
# Description: OpenAIGPT against a stubbed client — response handling, costing,
#   conversation persistence, and the settings it passes to collaborators.
# =================================================================================

# import modules
import sqlite3
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llms
from osintgpt.llms import OpenAIGPT

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

from conftest import FAKE_KEY, StubCompletions


@pytest.fixture
def gpt(settings, stub_client):
    instance = OpenAIGPT(settings)
    instance.client = stub_client

    return instance


def conversation_rows(settings):
    connection = sqlite3.connect(settings.sql_db_file_path)
    try:
        return connection.execute(
            'SELECT role, message FROM chat_gpt_conversations'
        ).fetchall()
    finally:
        connection.close()


class TestConstruction:
    def test_requires_a_key(self, tmp_path):
        with pytest.raises(MissingEnvironmentVariableError, match='OPENAI_API_KEY'):
            OpenAIGPT(Settings(
                openai_gpt_model='gpt-4o',
                sql_db_file_path=str(tmp_path / 'x.db')
            ))

    def test_accepts_a_path_with_a_deprecation_warning(self, env_file):
        with pytest.warns(DeprecationWarning):
            OpenAIGPT(env_file)

    def test_does_not_keep_an_env_file_path(self, settings):
        assert not hasattr(OpenAIGPT(settings), 'env_file_path')

    def test_two_instances_hold_independent_configuration(self, tmp_path):
        first = OpenAIGPT(Settings(
            openai_api_key='sk-one', openai_gpt_model='gpt-4o',
            sql_db_file_path=str(tmp_path / 'one.db')
        ))
        second = OpenAIGPT(Settings(
            openai_api_key='sk-two', openai_gpt_model='gpt-4o-mini',
            sql_db_file_path=str(tmp_path / 'two.db')
        ))

        assert first.OPENAI_GPT_MODEL != second.OPENAI_GPT_MODEL
        assert first.client.api_key != second.client.api_key


class TestResponseAccessors:
    @pytest.fixture
    def response(self):
        return StubCompletions().create(model='gpt-4o', messages=[])

    def test_reads_the_id(self, gpt, response):
        assert gpt._get_completion_response_id(response) == 'chatcmpl-stub'

    def test_reads_usage_as_a_mapping(self, gpt, response):
        assert gpt._get_completion_response_usage(response)['total_tokens'] == 18

    def test_tolerates_absent_usage(self, gpt):
        '''Some OpenAI-compatible gateways omit usage entirely.'''
        response = SimpleNamespace(
            id='x', created=1, usage=None,
            choices=[SimpleNamespace(
                message=SimpleNamespace(role='assistant', content='c')
            )]
        )

        assert gpt._get_completion_response_usage(response) == {}

    def test_reads_the_role_and_message(self, gpt, response):
        assert gpt._get_completion_response_role_and_message(response) == (
            'assistant', StubCompletions.REPLY
        )


class TestCompletions:
    def test_returns_the_message_content(self, gpt):
        assert gpt.get_model_completion('a question') == StubCompletions.REPLY

    def test_sends_the_configured_model(self, gpt):
        gpt.get_model_completion('a question')

        assert gpt.client.chat.completions.calls[0]['model'] == 'gpt-4o'

    def test_builds_a_user_message_when_none_given(self, gpt):
        gpt.get_model_completion('a question')
        messages = gpt.client.chat.completions.calls[0]['messages']

        assert messages == [{'role': 'user', 'content': 'a question'}]

    def test_system_role_variant_forwards_keyword_arguments(self, gpt):
        gpt.get_model_completion_using_system_role(
            messages=[
                {'role': 'system', 'content': 'be terse'},
                {'role': 'user', 'content': 'summarize this'}
            ],
            verbose=False,
            temperature=0.4
        )

        assert gpt.client.chat.completions.calls[0]['temperature'] == 0.4

    def test_analyze_sentence_details_returns_content(self, gpt):
        result = gpt.analyze_sentence_details('¿Qué es la energía solar?')

        assert result == StubCompletions.REPLY


class TestPersistence:
    def test_logs_the_exchange(self, gpt, settings):
        gpt.get_model_completion('a question', verbose=False)
        rows = conversation_rows(settings)

        assert ('user', 'a question') in rows
        assert ('assistant', StubCompletions.REPLY) in rows

    def test_reuses_one_conversation_id(self, gpt, settings):
        gpt.get_model_completion('first', verbose=False)
        gpt.get_model_completion('second', verbose=False)

        connection = sqlite3.connect(settings.sql_db_file_path)
        try:
            ids = connection.execute(
                'SELECT DISTINCT ref_id FROM chat_gpt_conversations'
            ).fetchall()
        finally:
            connection.close()

        assert len(ids) == 1


class TestCosting:
    def test_counts_tokens_for_the_chat_model(self, gpt):
        assert gpt.count_tokens('a prompt') > 0

    def test_estimates_a_cost_for_a_priced_model(self, gpt):
        assert gpt.estimated_prompt_cost('a prompt') > 0

    def test_returns_none_for_an_unpriced_model(self, tmp_path):
        instance = OpenAIGPT(Settings(
            openai_api_key=FAKE_KEY,
            openai_gpt_model='gpt-99-unreleased',
            sql_db_file_path=str(tmp_path / 'x.db')
        ))

        assert instance.estimated_prompt_cost('a prompt') is None


class TestVectorSearchGuards:
    def test_rejects_a_call_with_neither_query_nor_embeddings(self, gpt):
        with pytest.raises(ValueError, match='query or embeddings'):
            gpt.search_results_from_vector(vector_engine=None)

    def test_rejects_a_non_engine(self, gpt):
        with pytest.raises(ValueError, match='Invalid vector engine'):
            gpt.search_results_from_vector(
                vector_engine=object(), query='a question'
            )
