# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_tool_calling.py
# Description: Translating one neutral tool-calling shape into two vendors'
#   wire formats. Untested, this is where a provider silently sends nothing.
# =================================================================================

# import modules
import json
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt llm
from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.base import GenerationProvider
from osintgpt.llm.calling import (
    Exchange,
    ModelTurn,
    ToolCall,
    ToolCallingUnsupported,
    tool_spec
)
from osintgpt.llm.openai_compat import OpenAICompatGeneration

SPEC = tool_spec(
    'semantic_search', 'find things',
    properties={'query': {'type': 'string'}}, required=['query']
)

HISTORY = [Exchange(
    turn=ModelTurn(
        text='surveying',
        calls=[ToolCall(id='call_1', name='semantic_search',
                        arguments={'query': 'x'})]
    ),
    results={'call_1': '{"passages": []}'}
)]


class Recorder:
    '''Captures the request and returns a prepared reply.'''

    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def create(self, **kwargs):
        self.seen = kwargs

        return self.reply


def openai_provider(reply):
    recorder = Recorder(reply)
    provider = OpenAICompatGeneration.__new__(OpenAICompatGeneration)
    provider.model, provider.provider = 'm', 'openai'
    provider.billable, provider.recorder = True, None
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=recorder)
    )

    return provider, recorder


def anthropic_provider(reply):
    recorder = Recorder(reply)
    provider = AnthropicGeneration.__new__(AnthropicGeneration)
    provider.model, provider.max_tokens, provider.recorder = 'c', 4096, None
    provider.client = SimpleNamespace(messages=recorder)

    return provider, recorder


def openai_reply(text='ok', calls=()):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        message=SimpleNamespace(
            content=text,
            tool_calls=[
                SimpleNamespace(
                    id=cid,
                    function=SimpleNamespace(name=name, arguments=args)
                )
                for cid, name, args in calls
            ] or None
        )
    )])


def anthropic_reply(blocks):
    return SimpleNamespace(content=blocks)


class TestOpenAIShape:
    def test_a_tool_result_follows_the_assistant_turn_that_asked(self):
        provider, recorder = openai_provider(openai_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)

        assert [m['role'] for m in recorder.seen['messages']] == [
            'system', 'user', 'assistant', 'tool'
        ]

    def test_the_result_carries_the_call_id_back(self):
        '''
        Without it the result refers to nothing and the request is refused.
        '''
        provider, recorder = openai_provider(openai_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)
        tool_message = [
            m for m in recorder.seen['messages'] if m['role'] == 'tool'
        ][0]

        assert tool_message['tool_call_id'] == 'call_1'

    def test_the_assistant_turn_repeats_the_calls_it_made(self):
        provider, recorder = openai_provider(openai_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)
        assistant = [
            m for m in recorder.seen['messages'] if m['role'] == 'assistant'
        ][0]

        assert assistant['tool_calls'][0]['id'] == 'call_1'
        assert json.loads(
            assistant['tool_calls'][0]['function']['arguments']
        ) == {'query': 'x'}

    def test_tools_are_sent_as_function_schemas(self):
        provider, recorder = openai_provider(openai_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['tools'][0]['function']['name'] == (
            'semantic_search'
        )

    def test_no_tools_key_when_none_are_offered(self):
        '''
        The final ask offers nothing, and sending an empty list is not the
        same request as sending none.
        '''
        provider, recorder = openai_provider(openai_reply())
        provider.generate_with_tools('sys', 'q', [], [])

        assert 'tools' not in recorder.seen

    def test_calls_come_back_parsed(self):
        provider, _ = openai_provider(openai_reply(
            text='', calls=[('c1', 'semantic_search', '{"query": "found"}')]
        ))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert turn.calls[0].arguments == {'query': 'found'}

    def test_malformed_arguments_do_not_fail_the_round(self):
        '''
        A model writing broken JSON is its own mistake to correct; an empty
        mapping lets the tool say what it needed.
        '''
        provider, _ = openai_provider(openai_reply(
            text='', calls=[('c1', 'semantic_search', '{not json')]
        ))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert turn.calls[0].arguments == {}


class TestAnthropicShape:
    def test_a_tool_result_is_a_user_turn_not_a_role_of_its_own(self):
        provider, recorder = anthropic_provider(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)

        assert [m['role'] for m in recorder.seen['messages']] == [
            'user', 'assistant', 'user'
        ]

    def test_the_blocks_are_the_shape_the_vendor_expects(self):
        provider, recorder = anthropic_provider(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)
        blocks = [
            b['type'] for m in recorder.seen['messages']
            if isinstance(m['content'], list) for b in m['content']
        ]

        assert blocks == ['text', 'tool_use', 'tool_result']

    def test_the_system_instruction_is_a_parameter_not_a_message(self):
        provider, recorder = anthropic_provider(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['system'] == 'sys'

    def test_tools_use_an_input_schema(self):
        provider, recorder = anthropic_provider(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert 'input_schema' in recorder.seen['tools'][0]

    def test_tool_use_blocks_come_back_as_calls(self):
        provider, _ = anthropic_provider(anthropic_reply([
            SimpleNamespace(type='text', text='thinking'),
            SimpleNamespace(type='tool_use', id='tu_1',
                            name='semantic_search', input={'query': 'y'})
        ]))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert turn.text == 'thinking'
        assert turn.calls[0].id == 'tu_1'
        assert turn.calls[0].arguments == {'query': 'y'}


class TestBothProvidersAgree:
    '''
    A trace from one provider must read against a trace from another, which
    is only true if both produce the same neutral shape.
    '''

    def test_both_return_a_model_turn(self):
        openai, _ = openai_provider(openai_reply(text='same'))
        anthropic, _ = anthropic_provider(
            anthropic_reply([SimpleNamespace(type='text', text='same')])
        )

        first = openai.generate_with_tools('s', 'q', [SPEC], [])
        second = anthropic.generate_with_tools('s', 'q', [SPEC], [])

        assert first.text == second.text == 'same'
        assert first.wants_tools is second.wants_tools is False

    def test_both_declare_tool_support(self):
        assert OpenAICompatGeneration.supports_tools is True
        assert AnthropicGeneration.supports_tools is True

    def test_a_backend_that_cannot_is_the_default(self):
        assert GenerationProvider.supports_tools is False

    def test_the_default_refusal_is_catchable_on_its_own(self):
        '''
        A distinct type, so the loop degrades on this and not on every
        NotImplementedError, which would hide real bugs.
        '''
        class Bare(GenerationProvider):
            model = 'bare'

            def generate(self, system, user, **kwargs):
                return ''

        with pytest.raises(ToolCallingUnsupported):
            Bare().generate_with_tools('s', 'q', [SPEC], [])
