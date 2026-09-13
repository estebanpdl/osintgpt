# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_cli_lazy_generator.py
# Description: The CLI's lazy provider wrapper. It adds laziness and nothing
#   else, so anything it fails to forward is a capability the CLI silently
#   loses — the loop degrades rather than raising, which hides it.
# =================================================================================

# import submodules
from types import SimpleNamespace

# import osintgpt
from osintgpt.cli.retrieval import _LazyGenerator
from osintgpt.llm.calling import ModelTurn


class Provider:
    model = 'stub'
    supports_tools = True

    def __init__(self):
        self.seen = None

    def generate(self, system, user):
        return 'answered'

    def generate_with_tools(
        self, system, user, tools, history=None, conversation=None
    ):
        self.seen = {
            'system': system, 'user': user, 'tools': tools,
            'history': history, 'conversation': conversation
        }

        return ModelTurn(text='answered')


def wrapped(provider):
    lazy = _LazyGenerator.__new__(_LazyGenerator)
    lazy.effective = SimpleNamespace(
        generation_provider='openai', generation_model='m'
    )
    lazy.config, lazy.recorder = None, None
    lazy._built = provider

    return lazy


class TestItForwardsEverything:
    def test_the_conversation_reaches_the_provider(self):
        '''
        A parameter the wrapper does not forward raises a TypeError the loop
        catches, so the CLI falls back to the static pipeline: no tools, no
        conversation context, and nothing but the degraded note to show it.
        '''
        provider = Provider()
        pairs = [('What is alpha?', 'Aardvarks.')]

        wrapped(provider).generate_with_tools(
            'sys', 'q', [], [], conversation=pairs
        )

        assert provider.seen['conversation'] == pairs

    def test_the_tool_history_still_reaches_it(self):
        provider = Provider()

        wrapped(provider).generate_with_tools('sys', 'q', [], ['exchange'])

        assert provider.seen['history'] == ['exchange']

    def test_a_call_without_extras_still_works(self):
        provider = Provider()

        turn = wrapped(provider).generate_with_tools('sys', 'q', [], [])

        assert turn.text == 'answered'
        assert provider.seen['conversation'] is None

    def test_it_forwards_a_parameter_it_has_never_heard_of(self):
        '''
        The wrapper exists to defer construction. Naming the provider's
        parameters means editing it every time one is added, and forgetting is
        a capability lost with no error.
        '''
        captured = {}

        class Future:
            model = 'future'
            supports_tools = True

            def generate_with_tools(self, system, user, tools, history=None,
                                    **kwargs):
                captured.update(kwargs)

                return ModelTurn(text='ok')

        wrapped(Future()).generate_with_tools(
            'sys', 'q', [], [], something_new=1
        )

        assert captured == {'something_new': 1}
