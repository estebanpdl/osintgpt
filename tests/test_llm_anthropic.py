# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_anthropic.py
# Description: The native Anthropic backend — that it satisfies the same
#   interface as every other provider, and fails legibly when uninstalled.
# =================================================================================

# import modules
import builtins
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llm
from osintgpt.llm import (
    GENERATION_BACKENDS,
    GenerationProvider,
    build_generation_provider
)
from osintgpt.llm.anthropic_native import (
    DEFAULT_MAX_TOKENS,
    AnthropicGeneration
)
from osintgpt.llm.registry import ANTHROPIC, OPENAI_COMPAT

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

from conftest import FAKE_KEY


class StubMessages:
    '''Records requests; replies with the blocks it was configured with.'''

    DEFAULT = [SimpleNamespace(type='text', text='STUB REPLY')]

    def __init__(self, blocks=None):
        self.calls = []
        # `blocks or DEFAULT` would swallow an explicit empty list, which is
        # exactly the case a refusal produces.
        self.blocks = self.DEFAULT if blocks is None else blocks

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(content=self.blocks)


class StubAnthropic:
    def __init__(self, blocks=None):
        self.messages = StubMessages(blocks)


@pytest.fixture
def provider():
    return AnthropicGeneration(
        model='claude-opus-5', api_key=FAKE_KEY, client=StubAnthropic()
    )


class TestRegistration:
    def test_is_a_generation_backend(self):
        assert 'anthropic' in GENERATION_BACKENDS

    def test_uses_its_own_kind(self):
        assert GENERATION_BACKENDS['anthropic'].kind == ANTHROPIC
        assert GENERATION_BACKENDS['anthropic'].kind != OPENAI_COMPAT

    def test_declares_the_extra_that_installs_it(self):
        assert GENERATION_BACKENDS['anthropic'].extra == 'anthropic'

    def test_offers_no_embeddings(self):
        '''Anthropic publishes no embedding model.'''
        from osintgpt.llm import EMBEDDING_BACKENDS

        assert 'anthropic' not in EMBEDDING_BACKENDS

    def test_reads_its_own_key(self):
        settings = Settings(
            openai_api_key=FAKE_KEY, openai_gpt_model='claude-opus-5'
        )

        with pytest.raises(MissingEnvironmentVariableError, match='ANTHROPIC'):
            build_generation_provider('anthropic', settings)


class TestInterface:
    def test_satisfies_the_shared_interface(self, provider):
        assert isinstance(provider, GenerationProvider)

    def test_carries_its_model(self, provider):
        assert provider.model == 'claude-opus-5'

    def test_returns_the_reply(self, provider):
        assert provider.generate('be terse', 'a question') == 'STUB REPLY'


class TestRequestShape:
    def test_system_is_a_parameter_not_a_message(self, provider):
        provider.generate('be terse', 'a question')
        sent = provider.client.messages.calls[0]

        assert sent['system'] == 'be terse'
        assert sent['messages'] == [
            {'role': 'user', 'content': 'a question'}
        ]

    def test_sends_a_token_ceiling(self, provider):
        provider.generate('s', 'u')

        assert provider.client.messages.calls[0]['max_tokens'] == (
            DEFAULT_MAX_TOKENS
        )

    def test_the_ceiling_is_configurable(self):
        provider = AnthropicGeneration(
            model='claude-opus-5', api_key=FAKE_KEY,
            max_tokens=512, client=StubAnthropic()
        )
        provider.generate('s', 'u')

        assert provider.client.messages.calls[0]['max_tokens'] == 512

    def test_does_not_configure_thinking(self, provider):
        '''
        Valid thinking configuration differs per Claude model, and the model is
        the caller's choice, so the request stays silent and lets it default.
        '''
        provider.generate('s', 'u')

        assert 'thinking' not in provider.client.messages.calls[0]


class TestResponseHandling:
    def test_joins_several_text_blocks(self):
        provider = AnthropicGeneration(
            model='claude-opus-5', api_key=FAKE_KEY,
            client=StubAnthropic([
                SimpleNamespace(type='text', text='one '),
                SimpleNamespace(type='text', text='two')
            ])
        )

        assert provider.generate('s', 'u') == 'one two'

    def test_ignores_non_text_blocks(self):
        provider = AnthropicGeneration(
            model='claude-opus-5', api_key=FAKE_KEY,
            client=StubAnthropic([
                SimpleNamespace(type='thinking', thinking='reasoning'),
                SimpleNamespace(type='text', text='the answer')
            ])
        )

        assert provider.generate('s', 'u') == 'the answer'

    def test_a_reply_with_no_text_is_an_empty_string(self):
        provider = AnthropicGeneration(
            model='claude-opus-5', api_key=FAKE_KEY,
            client=StubAnthropic([])
        )

        assert provider.generate('s', 'u') == ''


class TestMissingPackage:
    def test_names_the_extra_that_installs_it(self, monkeypatch):
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'anthropic':
                raise ImportError('No module named anthropic')

            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        with pytest.raises(ImportError) as excinfo:
            AnthropicGeneration(model='claude-opus-5', api_key=FAKE_KEY)

        message = str(excinfo.value)

        assert 'anthropic' in message
        assert 'osintgpt[anthropic]' in message
