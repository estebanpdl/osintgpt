'''Exercise native request serialization and completion metadata without network access.'''

import json

import httpx
import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI

from osintgpt.graph.contract import EXTRACTION_SCHEMA
from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.openai_compat import OpenAICompatGeneration
from osintgpt.llm.openai_responses import OpenAIResponsesGeneration
from osintgpt.llm.usage import UsageRecorder
from tests.conftest import FAKE_KEY
from tests.llm.anthropic_fixtures import message_stream, model_metadata

TEXT = '{"entities":[],"edges":[]}'


def provider_fixture(kind, *, reason=None, refusal=False):
    requests = []
    if kind == 'responses':
        content = [{'type': 'refusal', 'refusal': 'Unavailable'}] if refusal else [
            {'type': 'output_text', 'text': TEXT, 'annotations': []}]
        body = {'id': 'resp_test', 'object': 'response', 'created_at': 0, 'model': 'offline',
                'status': 'incomplete' if reason else 'completed',
                'incomplete_details': {'reason': reason} if reason else None,
                'output': [{'id': 'msg_test', 'type': 'message', 'role': 'assistant',
                            'status': 'completed', 'content': content}],
                'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}
    elif kind == 'anthropic':
        body = {'id': 'msg_test', 'type': 'message', 'model': 'offline', 'role': 'assistant',
                'content': [{'type': 'text', 'text': TEXT}], 'stop_sequence': None,
                'stop_reason': 'refusal' if refusal else reason or 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 5}}
    else:
        body = {'id': 'chatcmpl_test', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
                'choices': [{'index': 0, 'finish_reason': reason or 'stop',
                             'message': {'role': 'assistant', 'content': TEXT,
                                         'refusal': 'Unavailable' if refusal else None}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}

    transport_api = httpx2 if kind == 'anthropic' else httpx

    def respond(request):
        if request.method == 'GET' and kind == 'anthropic':
            return transport_api.Response(200, json=model_metadata())
        data = json.loads(request.content)
        requests.append(data)
        if kind not in {'responses', 'anthropic'} and any(
                tool.get('function', {}).get('name') == 'submit_extraction' for tool in data.get('tools', [])):
            from copy import deepcopy
            submitted = deepcopy(body)
            if not reason and not refusal:
                submitted['choices'][0].update(finish_reason='tool_calls', message={'role': 'assistant', 'content': None,
                    'tool_calls': [{'id': 'submit', 'type': 'function', 'function': {
                        'name': 'submit_extraction', 'arguments': TEXT}}]})
            return transport_api.Response(200, json=submitted)
        if kind == 'anthropic' and data.get('stream'):
            return transport_api.Response(200, headers={'content-type': 'text/event-stream'}, content=message_stream(body))
        return transport_api.Response(200, json=body)

    client = transport_api.Client(transport=transport_api.MockTransport(respond))
    recorder = UsageRecorder()
    if kind == 'anthropic':
        provider = AnthropicGeneration('offline', FAKE_KEY, recorder=recorder,
                                        client=Anthropic(api_key=FAKE_KEY, http_client=client))
    else:
        if kind == 'responses':
            provider = OpenAIResponsesGeneration('offline', FAKE_KEY, recorder=recorder)
        else:
            provider = OpenAICompatGeneration('offline', FAKE_KEY, recorder=recorder,
                                              base_url='https://endpoint.invalid/v1', provider=kind)
        provider.client.close()
        provider.client = OpenAI(api_key=FAKE_KEY, http_client=client)
    return provider, requests


@pytest.mark.parametrize('kind', ['responses', 'anthropic', 'gemini', 'ollama'])
def test_native_schema_and_output_limit_survive_sdk_serialization(kind):
    provider, requests = provider_fixture(kind)
    try:
        result = provider.generate_response('Instructions', 'Material', schema=EXTRACTION_SCHEMA,
                                             max_output_tokens=1234)
        assert result.require_complete() == TEXT and result.has_completion_status
        assert provider.recorder.calls == 1 and provider.recorder.total_tokens == 15
        request = requests[0]
        if kind == 'responses':
            contract = request['text']['format']
            assert request['store'] is False and request['max_output_tokens'] == 1234
            assert request['instructions'] == 'Instructions'
        elif kind == 'anthropic':
            contract = request['output_config']['format']
            assert request['max_tokens'] == 1234 and request['system'] == 'Instructions'
        else:
            contract = request['response_format']['json_schema']
            assert request['max_tokens'] == 1234
            assert request['messages'][0] == {'role': 'system', 'content': 'Instructions'}
        assert contract['schema'] == EXTRACTION_SCHEMA
        if kind != 'anthropic':
            assert contract['strict'] is True
        provider.generate('Instructions', 'Material')
        assert not {'text', 'output_config', 'response_format'} & requests[1].keys()
    finally:
        provider.client.close()


@pytest.mark.parametrize('kind,reason', [
    ('responses', 'max_output_tokens'), ('responses', 'content_filter'),
    ('anthropic', 'max_tokens'), ('anthropic', 'model_context_window_exceeded'),
    ('gemini', 'length'), ('ollama', 'length'), ('gemini', 'content_filter'),
])
def test_provider_completion_signals_cannot_be_hidden_by_valid_json(kind, reason):
    provider, requests = provider_fixture(kind, reason=reason)
    try:
        response = provider.generate_response('Instructions', 'Material', schema=EXTRACTION_SCHEMA)
        assert response.text == TEXT
        with pytest.raises(ValueError, match='truncated|not complete|without completion'):
            response.require_complete()
        assert len(requests) == provider.recorder.calls == 1
    finally:
        provider.client.close()


@pytest.mark.parametrize('kind', ['responses', 'anthropic', 'gemini'])
def test_provider_refusal_is_explicit_and_still_accounts_usage(kind):
    provider, _ = provider_fixture(kind, refusal=True)
    try:
        response = provider.generate_response('Instructions', 'Material', schema=EXTRACTION_SCHEMA)
        with pytest.raises(ValueError, match='refused'):
            response.require_complete()
        assert provider.recorder.calls == 1
    finally:
        provider.client.close()


def test_unknown_compatible_endpoint_requires_explicit_schema_capability():
    provider, requests = provider_fixture('custom')
    try:
        assert not provider.supports_structured_output
        with pytest.raises(ValueError, match='does not declare'):
            provider.generate_response('Instructions', 'Material', schema=EXTRACTION_SCHEMA)
        assert not requests
        assert provider.generate_response('Instructions', 'Material').require_complete() == TEXT
    finally:
        provider.client.close()
