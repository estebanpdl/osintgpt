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
from osintgpt.llm.openai_responses import (
    OpenAIResponsesGeneration,
    _ResponsesTurn
)

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


def responses_provider(reply):
    recorder = Recorder(reply)
    provider = OpenAIResponsesGeneration.__new__(OpenAIResponsesGeneration)
    provider.model, provider.provider = 'm', 'openai'
    provider.billable, provider.recorder = True, None
    provider.client = SimpleNamespace(responses=recorder)

    return provider, recorder


def responses_item(**fields):
    item = SimpleNamespace(**fields)
    item.model_dump = lambda exclude_none=False: dict(fields)

    return item


def responses_reply(text='ok', calls=()):
    output = [
        responses_item(
            type='function_call', id=f'fc_{cid}', call_id=cid,
            name=name, arguments=args
        )
        for cid, name, args in calls
    ]

    return SimpleNamespace(usage=None, output=output, output_text=text)


# What a reasoning model returns alongside its calls, and what a later round
# has to send back for the model to keep the thinking it already paid for.
REASONING = responses_item(
    type='reasoning', id='rs_1', summary=[], encrypted_content='opaque'
)
CALL_ITEM = responses_item(
    type='function_call', id='fc_1', call_id='call_1',
    name='semantic_search', arguments='{"query": "x"}'
)

RESPONSES_HISTORY = [Exchange(
    turn=_ResponsesTurn(
        text='surveying',
        calls=[ToolCall(id='call_1', name='semantic_search',
                        arguments={'query': 'x'})],
        items=(REASONING.model_dump(), CALL_ITEM.model_dump())
    ),
    results={'call_1': '{"passages": []}'}
)]


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


class TestOpenAIResponsesShape:
    '''
    OpenAI's own endpoint, which is the only one that serves function tools
    and reasoning together. Its wire format is not the compatible one.
    '''

    def test_the_system_prompt_is_instructions_not_a_message(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['instructions'] == 'sys'
        assert recorder.seen['input'] == [{'role': 'user', 'content': 'q'}]

    def test_tools_are_flat_not_nested_under_a_function_key(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['tools'][0]['name'] == 'semantic_search'
        assert recorder.seen['tools'][0]['type'] == 'function'

    def test_no_tools_key_when_none_are_offered(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [], [])

        assert 'tools' not in recorder.seen

    def test_a_result_follows_the_items_that_asked_for_it(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], RESPONSES_HISTORY)

        assert [
            item.get('role') or item['type']
            for item in recorder.seen['input']
        ] == ['user', 'reasoning', 'function_call', 'function_call_output']

    def test_the_result_carries_the_call_id_back(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], RESPONSES_HISTORY)
        output = recorder.seen['input'][-1]

        assert output['call_id'] == 'call_1'
        assert output['output'] == '{"passages": []}'

    def test_reasoning_is_replayed_verbatim(self):
        '''
        Dropping it makes the model re-derive what it already worked out, so
        the reasoning tokens are paid for once per round instead of once.
        '''
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], RESPONSES_HISTORY)
        replayed = [
            item for item in recorder.seen['input']
            if item.get('type') == 'reasoning'
        ]

        assert replayed == [REASONING.model_dump()]

    def test_a_turn_from_another_provider_replays_no_items(self):
        '''
        The raw items ride on this provider's own turn; a neutral one carries
        none, and asking for them must not raise.
        '''
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], HISTORY)

        assert [
            item.get('role') or item['type']
            for item in recorder.seen['input']
        ] == ['user', 'function_call_output']

    def test_the_conversation_is_never_stored_on_the_vendor(self):
        provider, recorder = responses_provider(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['store'] is False

    def test_calls_come_back_parsed(self):
        provider, _ = responses_provider(responses_reply(
            text='', calls=[('c1', 'semantic_search', '{"query": "found"}')]
        ))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert turn.calls[0].id == 'c1'
        assert turn.calls[0].arguments == {'query': 'found'}

    def test_parallel_calls_all_come_back(self):
        provider, _ = responses_provider(responses_reply(text='', calls=[
            ('c1', 'semantic_search', '{"query": "a"}'),
            ('c2', 'semantic_search', '{"query": "b"}')
        ]))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert [call.id for call in turn.calls] == ['c1', 'c2']

    def test_malformed_arguments_do_not_fail_the_round(self):
        provider, _ = responses_provider(responses_reply(
            text='', calls=[('c1', 'semantic_search', '{not json')]
        ))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert turn.calls[0].arguments == {}

    def test_the_turn_is_one_the_loop_understands(self):
        provider, _ = responses_provider(responses_reply(
            text='', calls=[('c1', 'semantic_search', '{}')]
        ))

        turn = provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert isinstance(turn, ModelTurn)
        assert turn.wants_tools


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
