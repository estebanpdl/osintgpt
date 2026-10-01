# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_litellm.py
# Description: The LiteLLM backend — that it satisfies the same interfaces as
#   every other provider, routes keys and proxies correctly, and fails legibly
#   when uninstalled. The litellm module is stubbed, never imported for real.
# =================================================================================

# import modules
import builtins
import json
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llm
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    EmbeddingProvider,
    GenerationProvider,
    UsageRecorder,
    audit_locality,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm import litellm_sdk
from osintgpt.llm.calling import Exchange, ModelTurn, ToolCall, tool_spec
from osintgpt.llm.litellm_sdk import (
    LiteLLMEmbedding,
    LiteLLMGeneration,
    resolve_litellm_key
)
from osintgpt.llm.registry import LITELLM

from tests.conftest import FAKE_KEY


def _reply(content='STUB REPLY', tool_calls=None, usage=True):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)

    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3)
        if usage else None
    )


class StubLiteLLM:
    '''Stands in for the litellm module; records every request.'''

    def __init__(self, reply=None, vectors=None, vision=True):
        self.completions = []
        self.embeddings = []
        self.reply = reply or _reply()
        self.vectors = vectors
        self.vision = vision

    def completion(self, **kwargs):
        self.completions.append(kwargs)

        return self.reply

    def embedding(self, **kwargs):
        self.embeddings.append(kwargs)
        batch = kwargs['input']
        # Reversed on purpose: providers need not return a batch in order.
        data = self.vectors(batch) if self.vectors else [
            {'index': i, 'embedding': [float(i), float(len(text))]}
            for i, text in reversed(list(enumerate(batch)))
        ]

        return SimpleNamespace(
            data=data, usage=SimpleNamespace(prompt_tokens=len(batch))
        )

    def supports_vision(self, model):
        return self.vision


@pytest.fixture
def stub(monkeypatch):
    fake = StubLiteLLM()
    monkeypatch.setattr(litellm_sdk, '_load_litellm', lambda: fake)

    return fake


class TestRegistration:
    def test_registered_for_both_roles(self):
        for backends in (GENERATION_BACKENDS, EMBEDDING_BACKENDS):
            spec = backends['litellm']
            assert spec.kind == LITELLM
            assert spec.extra == 'litellm'
            # The key belongs to the route the model names, not to the backend.
            assert spec.settings_field is None
            assert not spec.local

    def test_builders_return_the_litellm_backends(self, stub):
        settings = Settings(openai_gpt_model='anthropic/claude-sonnet-5')

        generation = build_generation_provider('litellm', settings)
        embedding = build_embedding_provider(
            'litellm', settings, model='cohere/embed-english-v3.0'
        )

        assert isinstance(generation, LiteLLMGeneration)
        assert isinstance(generation, GenerationProvider)
        assert generation.model == 'anthropic/claude-sonnet-5'
        assert isinstance(embedding, LiteLLMEmbedding)
        assert isinstance(embedding, EmbeddingProvider)

    def test_judged_not_local(self):
        report = audit_locality(Settings(), 'litellm', 'litellm')

        assert not report.is_local

    def test_uninstalled_fails_legibly(self, monkeypatch):
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'litellm':
                raise ImportError('No module named litellm')

            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        with pytest.raises(ImportError) as excinfo:
            LiteLLMGeneration(model='anthropic/claude-sonnet-5')

        assert "osintgpt[litellm]" in str(excinfo.value)


class TestKeys:
    def test_stored_key_serves_the_matching_route(self):
        settings = Settings(anthropic_api_key='sk-ant', openai_api_key=FAKE_KEY)

        assert resolve_litellm_key('anthropic/claude-sonnet-5', settings) == 'sk-ant'
        assert resolve_litellm_key('gpt-5-mini', settings) == FAKE_KEY

    def test_other_routes_defer_to_litellm(self):
        # None, so LiteLLM reads AWS_*, AZURE_*, ... itself.
        settings = Settings(anthropic_api_key='sk-ant')

        assert resolve_litellm_key('bedrock/anthropic.claude-v2', settings) is None

    def test_builder_hands_over_the_route_key(self, stub):
        settings = Settings(anthropic_api_key='sk-ant')
        provider = build_generation_provider(
            'litellm', settings, model='anthropic/claude-sonnet-5'
        )

        assert provider.api_key == 'sk-ant'

class TestGeneration:
    def test_generate(self):
        stub = StubLiteLLM()
        recorder = UsageRecorder()
        provider = LiteLLMGeneration(
            model='anthropic/claude-sonnet-5', recorder=recorder, client=stub
        )

        assert provider.generate('SYSTEM', 'USER') == 'STUB REPLY'

        request = stub.completions[0]
        assert request['model'] == 'anthropic/claude-sonnet-5'
        assert request['messages'] == [
            {'role': 'system', 'content': 'SYSTEM'},
            {'role': 'user', 'content': 'USER'}
        ]
        assert request['drop_params'] is True
        # Unset credentials are left out, so LiteLLM reads its own.
        assert 'api_key' not in request

        usage = recorder.records[0]
        assert (usage.provider, usage.model) == ('litellm', 'anthropic/claude-sonnet-5')
        assert (usage.input_tokens, usage.output_tokens) == (12, 3)
        assert usage.counted

    def test_missing_usage_is_not_a_zero(self):
        stub = StubLiteLLM(reply=_reply(usage=False))
        recorder = UsageRecorder()
        LiteLLMGeneration(
            model='anthropic/claude-sonnet-5', recorder=recorder, client=stub
        ).generate('S', 'U')

        assert not recorder.records[0].counted

    def test_empty_reply_is_empty_text(self):
        stub = StubLiteLLM(reply=_reply(content=None))

        assert LiteLLMGeneration(model='m', client=stub).generate('S', 'U') == ''

    def test_vision_follows_the_model_map(self):
        assert LiteLLMGeneration(model='m', client=StubLiteLLM(vision=True)).supports_vision
        assert not LiteLLMGeneration(model='m', client=StubLiteLLM(vision=False)).supports_vision

    def test_describe_image(self):
        stub = StubLiteLLM()
        provider = LiteLLMGeneration(model='anthropic/claude-sonnet-5', client=stub)

        assert provider.describe_image('S', 'What is this?', b'PNG', 'image/png') == 'STUB REPLY'

        content = stub.completions[0]['messages'][1]['content']
        assert content[0] == {'type': 'text', 'text': 'What is this?'}
        assert content[1]['image_url']['url'] == 'data:image/png;base64,UE5H'


class TestToolCalling:
    def test_tools_and_turn(self):
        calls = [
            SimpleNamespace(id='call_1', function=SimpleNamespace(
                name='search', arguments=json.dumps({'query': 'wagner'})
            )),
            # A malformed argument string is the model's mistake, not a crash.
            SimpleNamespace(id='call_2', function=SimpleNamespace(
                name='search', arguments='{not json'
            ))
        ]
        stub = StubLiteLLM(reply=_reply(content='', tool_calls=calls))
        provider = LiteLLMGeneration(model='bedrock/claude', client=stub)
        tools = [tool_spec('search', 'Search the corpus', {'query': {'type': 'string'}}, ['query'])]

        turn = provider.generate_with_tools('S', 'Q', tools)

        assert provider.supports_tools
        assert turn == ModelTurn(text='', calls=[
            ToolCall(id='call_1', name='search', arguments={'query': 'wagner'}),
            ToolCall(id='call_2', name='search', arguments={})
        ])
        assert stub.completions[0]['tools'] == [{
            'type': 'function',
            'function': {
                'name': 'search',
                'description': 'Search the corpus',
                'parameters': {
                    'type': 'object',
                    'properties': {'query': {'type': 'string'}},
                    'required': ['query']
                }
            }
        }]

    def test_history_and_conversation(self):
        stub = StubLiteLLM()
        provider = LiteLLMGeneration(model='bedrock/claude', client=stub)
        call = ToolCall(id='call_1', name='search', arguments={'query': 'x'})
        history = [Exchange(turn=ModelTurn(text='', calls=[call]), results={'call_1': 'FOUND'})]

        provider.generate_with_tools(
            'S', 'Q', [], history=history, conversation=[('earlier?', 'earlier.')]
        )

        messages = stub.completions[0]['messages']
        assert [m['role'] for m in messages] == [
            'system', 'user', 'assistant', 'user', 'assistant', 'tool'
        ]
        assert messages[4]['tool_calls'][0]['function'] == {
            'name': 'search', 'arguments': json.dumps({'query': 'x'})
        }
        assert messages[5] == {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'FOUND'}
        # An empty tool list means answer now: no tools are offered.
        assert 'tools' not in stub.completions[0]


class TestEmbedding:
    def test_batches_in_order(self):
        stub = StubLiteLLM()
        recorder = UsageRecorder()
        provider = LiteLLMEmbedding(
            model='cohere/embed-english-v3.0', batch_size=2,
            recorder=recorder, client=stub
        )

        vectors = provider.embed(['a', 'bb', 'ccc'])

        assert vectors == [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]]
        assert [r['input'] for r in stub.embeddings] == [['a', 'bb'], ['ccc']]
        assert stub.embeddings[0]['model'] == 'cohere/embed-english-v3.0'
        assert stub.embeddings[0]['drop_params'] is True
        assert [u.input_tokens for u in recorder.records] == [2, 1]
        assert {u.provider for u in recorder.records} == {'litellm'}

    def test_object_items(self):
        stub = StubLiteLLM(vectors=lambda batch: [
            SimpleNamespace(index=i, embedding=[1.0]) for i in range(len(batch))
        ])

        assert LiteLLMEmbedding(model='m', client=stub).embed(['a', 'b']) == [[1.0], [1.0]]

    def test_each_batch_is_delivered_before_the_next_call(self):
        stub = StubLiteLLM()
        provider = LiteLLMEmbedding(model='m', batch_size=1, client=stub)
        delivered = []

        def sink(vectors):
            # The next request has not been made yet when a batch lands.
            delivered.append((len(stub.embeddings), vectors))

        provider.embed_checkpointed(['a', 'b'], on_batch=sink)

        assert [calls for calls, _ in delivered] == [1, 2]

    def test_invalid_indices_raise(self):
        stub = StubLiteLLM(vectors=lambda batch: [{'index': 5, 'embedding': [0.0]}])

        with pytest.raises(RuntimeError, match='invalid vector indices'):
            LiteLLMEmbedding(model='m', client=stub).embed(['a'])

class TestDiscovery:
    def test_no_model_discovery(self):
        # Direct SDK routes have no endpoint of their own to ask.
        assert not LiteLLMGeneration(model='m', client=StubLiteLLM()).supports_model_discovery
        assert not LiteLLMEmbedding(model='m', client=StubLiteLLM()).supports_model_discovery
