# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_conversation.py
# Description: Earlier turns reaching the model. Each vendor places them in its
#   own format, and the failure worth preventing is sending them as the wrong
#   kind of context — or not at all.
# =================================================================================

# import submodules
from types import SimpleNamespace

# import osintgpt llm
from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.calling import tool_spec
from osintgpt.llm.openai_compat import OpenAICompatGeneration
from osintgpt.llm.openai_responses import OpenAIResponsesGeneration

from test_llm_tool_calling import (
    Recorder,
    anthropic_reply,
    openai_reply,
    responses_reply
)

SPEC = tool_spec('semantic_search', 'find things')

PAIRS = [
    ('What is the Pravda network?', 'A disinformation infrastructure.'),
    ('How large is it?', 'More than 150 sites.')
]


def compat(reply):
    recorder = Recorder(reply)
    provider = OpenAICompatGeneration.__new__(OpenAICompatGeneration)
    provider.model, provider.provider = 'm', 'openai'
    provider.billable, provider.recorder = True, None
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=recorder)
    )

    return provider, recorder


def responses(reply):
    recorder = Recorder(reply)
    provider = OpenAIResponsesGeneration.__new__(OpenAIResponsesGeneration)
    provider.model, provider.provider = 'm', 'openai'
    provider.billable, provider.recorder = True, None
    provider.client = SimpleNamespace(responses=recorder)

    return provider, recorder


def anthropic(reply):
    recorder = Recorder(reply)
    provider = AnthropicGeneration.__new__(AnthropicGeneration)
    provider.model, provider.max_tokens, provider.recorder = 'c', 4096, None
    provider.client = SimpleNamespace(messages=recorder)

    return provider, recorder


class TestTheCompatibleEndpoint:
    def test_pairs_come_before_the_question(self):
        provider, recorder = compat(openai_reply())
        provider.generate_with_tools(
            'sys', 'And who runs it?', [SPEC], [], conversation=PAIRS
        )
        roles = [m['role'] for m in recorder.seen['messages']]

        assert roles == [
            'system', 'user', 'assistant', 'user', 'assistant', 'user'
        ]
        assert recorder.seen['messages'][-1]['content'] == 'And who runs it?'

    def test_the_pairs_are_verbatim(self):
        provider, recorder = compat(openai_reply())
        provider.generate_with_tools(
            'sys', 'q', [SPEC], [], conversation=PAIRS
        )
        contents = [m['content'] for m in recorder.seen['messages']]

        assert PAIRS[0][0] in contents
        assert PAIRS[0][1] in contents

    def test_no_conversation_is_the_shape_it_always_was(self):
        provider, recorder = compat(openai_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert [m['role'] for m in recorder.seen['messages']] == [
            'system', 'user'
        ]


class TestTheResponsesEndpoint:
    def test_pairs_come_before_the_question(self):
        provider, recorder = responses(responses_reply())
        provider.generate_with_tools(
            'sys', 'And who runs it?', [SPEC], [], conversation=PAIRS
        )
        roles = [item.get('role') for item in recorder.seen['input']]

        assert roles == ['user', 'assistant', 'user', 'assistant', 'user']
        assert recorder.seen['input'][-1]['content'] == 'And who runs it?'

    def test_the_system_prompt_stays_instructions(self):
        '''
        Prior turns are turns. Folding them into the instructions would make
        what the analyst said read as something the tool told the model.
        '''
        provider, recorder = responses(responses_reply())
        provider.generate_with_tools(
            'sys', 'q', [SPEC], [], conversation=PAIRS
        )

        assert recorder.seen['instructions'] == 'sys'

    def test_no_conversation_is_the_shape_it_always_was(self):
        provider, recorder = responses(responses_reply())
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert recorder.seen['input'] == [{'role': 'user', 'content': 'q'}]


class TestAnthropic:
    def test_pairs_come_before_the_question(self):
        provider, recorder = anthropic(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools(
            'sys', 'And who runs it?', [SPEC], [], conversation=PAIRS
        )
        roles = [m['role'] for m in recorder.seen['messages']]

        assert roles == ['user', 'assistant', 'user', 'assistant', 'user']
        assert recorder.seen['messages'][-1]['content'] == 'And who runs it?'

    def test_no_conversation_is_the_shape_it_always_was(self):
        provider, recorder = anthropic(
            anthropic_reply([SimpleNamespace(type='text', text='ok')])
        )
        provider.generate_with_tools('sys', 'q', [SPEC], [])

        assert [m['role'] for m in recorder.seen['messages']] == ['user']


class TestTheToolHistoryIsNotTheConversation:
    '''
    `history` is the calls made answering this question; `conversation` is what
    came before it. Sending one as the other gives the model the wrong material.
    '''

    def test_both_can_travel_together(self):
        from osintgpt.llm.calling import Exchange, ModelTurn, ToolCall

        history = [Exchange(
            turn=ModelTurn(
                text='surveying',
                calls=[ToolCall(id='c1', name='semantic_search',
                                arguments={'query': 'x'})]
            ),
            results={'c1': '{"passages": []}'}
        )]

        provider, recorder = compat(openai_reply())
        provider.generate_with_tools(
            'sys', 'q', [SPEC], history, conversation=PAIRS
        )
        roles = [m['role'] for m in recorder.seen['messages']]

        # Pairs, then the question, then this answer's own tool round.
        assert roles == [
            'system', 'user', 'assistant', 'user', 'assistant',
            'user', 'assistant', 'tool'
        ]
