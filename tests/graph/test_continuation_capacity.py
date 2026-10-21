'''Real adapter serialization over local transports; no external model requests.'''

import json
from dataclasses import replace

import httpx
import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI

from osintgpt.graph import ExtractionOptions, extract_document
from osintgpt.graph.identity import EntityRegistry
from osintgpt.graph.identity_tools import IdentityTools
from osintgpt.graph.request_context import RequestContext
from osintgpt.graph.windowing import encode
from osintgpt.llm.generation import GenerationResponse
from tests.conftest import FAKE_KEY
from tests.llm.test_generation_contract import provider_fixture
from tests.llm.anthropic_fixtures import message_stream, model_metadata
from test_interactive_identity import prior_entity, proposal, PAST, NOW


def lookup(name, args, number=0):
    return {'id': f'call-{number}', 'name': name, 'arguments': args}


def search(query, number=0):
    return lookup('find_entity_candidates', {'queries': [query], 'entity_type': '', 'offset': 0, 'limit': 10}, number)


def scripted(kind, turns, *, continuation_size=0, input_limit=None):
    provider, _ = provider_fixture(kind)
    provider.client.close()
    requests = []
    accepted = []
    api = httpx2 if kind == 'anthropic' else httpx
    def respond(request):
        if request.method == 'GET':
            return api.Response(200, json=model_metadata())
        data = json.loads(request.content)
        requests.append(data)
        if input_limit and len(request.content) > input_limit:
            assert len(requests) < 100, 'Offline capacity recovery must make progress.'
            return api.Response(400, json={'error': {'message': 'Input too large', 'type': 'invalid_request_error',
                                                    'code': 'context_length_exceeded'}})
        number = len(accepted)
        accepted.append(data)
        turn = turns(number, data) if callable(turns) else turns[min(number, len(turns) - 1)]
        calls = turn if isinstance(turn, list) else []
        text = turn if isinstance(turn, str) else json.dumps(turn) if not calls else ''
        if kind == 'responses':
            output = ([{'type': 'reasoning', 'id': f'rs{number}', 'summary': [], 'encrypted_content': 'x' * continuation_size}]
                      + [{'type': 'function_call', 'id': f'fc{number}-{i}', 'call_id': call['id'],
                          'name': call['name'], 'arguments': json.dumps(call['arguments'])} for i, call in enumerate(calls)] if calls else
                      [{'type': 'message', 'id': 'msg', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}])
            body = {'id': f'resp{number}', 'object': 'response', 'created_at': 0, 'model': 'offline',
                    'status': 'completed', 'output': output,
                    'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}
        elif kind == 'anthropic':
            body = {'id': 'msg', 'type': 'message', 'role': 'assistant', 'model': 'offline',
                    'content': [{'type': 'tool_use', 'id': c['id'], 'name': c['name'], 'input': c['arguments']} for c in calls]
                    if calls else [{'type': 'text', 'text': text}], 'stop_sequence': None,
                    'stop_reason': 'tool_use' if calls else 'end_turn', 'usage': {'input_tokens': 10, 'output_tokens': 5}}
            return api.Response(200, headers={'content-type': 'text/event-stream'}, content=message_stream(body))
        else:
            if not calls and not isinstance(turn, str):
                calls = [lookup('submit_extraction', turn, number)]
            message = {'role': 'assistant', 'content': None if calls else text}
            if calls:
                message['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': {
                    'name': c['name'], 'arguments': json.dumps(c['arguments'])}} for c in calls]
            body = {'id': 'chat', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
                    'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        return api.Response(200, json=body)
    client = api.Client(transport=api.MockTransport(respond))
    provider.client = Anthropic(api_key=FAKE_KEY, http_client=client) if kind == 'anthropic' else OpenAI(api_key=FAKE_KEY, http_client=client)
    return provider, requests


@pytest.mark.parametrize('kind', ['responses', 'gemini', 'anthropic'])
def test_eight_simultaneous_evidence_reads_return_actual_quotations(kind):
    entity = prior_entity()
    reads = [lookup('read_entity_context', {'entity_id': entity.key, 'offset': 0, 'limit': 10, 'quote_offset': 0}, i)
             for i in range(8)]
    provider, requests = scripted(kind, [[search('Orion')], reads, proposal(entity.key)])
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity])
        assert result.ok, result.problem
        assert result.edges[0].source_id == entity.key
        third = json.dumps(requests[2], ensure_ascii=False)
        assert third.count(PAST) >= 8
        assert 'No evidence fits' not in third
        assert result.accounting.requests_attempted == len(requests) == 3
    finally:
        provider.client.close()


def test_large_latest_continuation_is_not_rejected_by_old_application_budget():
    provider, requests = scripted('responses', [[search('Orion')], {'entities': [], 'edges': []}], continuation_size=44000)
    try:
        result = extract_document(provider, 'new.md', 'Source text.')
        assert result.ok, result.problem
        assert len(requests) == 2 and len(json.dumps(requests[1])) > 32768
        assert requests[1]['input'][1]['encrypted_content'] == 'x' * 44000
        assert 'max_output_tokens' not in requests[1]
    finally:
        provider.client.close()


@pytest.mark.parametrize('kind', ['responses', 'gemini', 'anthropic'])
def test_restarts_preserve_tools_and_continue_more_than_eight_useful_rounds(kind):
    def turn(index, request):
        return [search(f'Unique {index}', index)] if index < 12 else {'entities': [], 'edges': []}
    provider, requests = scripted(kind, turn, continuation_size=21000)
    try:
        result = extract_document(provider, 'source.md', 'Text.', options=ExtractionOptions(context_budget=128))
        assert result.ok, result.problem
        assert len(requests) == 13
        assert result.accounting.context_restarts > 0
        assert any('lookup_state' in json.dumps(request) for request in requests[1:])
        if kind == 'responses':
            assert any(len(r['input']) == 1 and 'lookup_state' in r['input'][0]['content'] for r in requests[1:])
    finally:
        provider.client.close()


def test_more_than_500_useful_rounds_have_no_record_call_ceiling():
    class Model:
        supports_tools = supports_structured_output = True
        def __init__(self): self.calls = 0
        def generate_response(self, system, user, *, max_output_tokens=None, schema=None, tools=None, history=None):
            self.calls += 1
            return GenerationResponse('', status='completed', tool_calls=(search(str(self.calls), self.calls),)) if self.calls <= 501 else GenerationResponse('{"entities":[],"edges":[]}', status='completed')
    model = Model()
    result = extract_document(model, 'source.md', 'Text.')
    assert result.ok, result.problem
    assert model.calls == result.accounting.requests_attempted == 502


def test_repeated_calls_with_new_ids_make_no_progress_and_finish():
    def turn(index, request):
        return [search('Same query', index)] if index < 2 else {'entities': [], 'edges': []}
    provider, requests = scripted('responses', turn)
    try:
        result = extract_document(provider, 'source.md', 'Text.')
        assert result.ok and len(requests) == 3
        assert result.accounting.no_progress_stops == 1
        assert [tool['name'] for tool in requests[-1]['tools']] == ['read_saved_lookup']
    finally:
        provider.client.close()


def test_oversized_query_batches_and_results_are_not_arbitrarily_rejected():
    entity = prior_entity()
    tools = IdentityTools(EntityRegistry([entity]))
    result = tools.run('find_entity_candidates', {'queries': ['Orion'] * 40, 'limit': 100, 'offset': 0})
    assert result['status'] == 'ok' and len(result['results']) == 40
    assert all(r['candidates'][0]['description'] == entity.description for r in result['results'])
    evidence = tools.run('read_entity_context', {'entity_id': entity.key, 'offset': 0, 'limit': 100})
    assert evidence['evidence'][0]['quote'] == PAST


def test_measurement_uses_native_payload_not_internal_response_bookkeeping():
    provider, _ = scripted('responses', [{'entities': [], 'edges': []}])
    try:
        context = RequestContext(provider, ExtractionOptions())
        history = [{'response': {'continuation': [], 'tool_calls': [], 'usage': {'internal': 'x' * 50000}}, 'results': {}}]
        expected = provider.graph_request('system', 'user', schema=None, tools=None, history=history)
        assert context.measure('system', 'user', history=history) == len(encode(expected).encode())
        assert context.measure('system', 'user', history=history) < 1000
    finally:
        provider.client.close()


def test_oversized_opaque_exchange_restarts_with_exact_pending_evidence():
    entity = prior_entity()
    turns = [[search('Orion')], [lookup('read_entity_context', {'entity_id': entity.key, 'offset': 0, 'limit': 10})], proposal(entity.key)]
    provider, requests = scripted('responses', turns, continuation_size=44000)
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity],
                                  options=ExtractionOptions(model_input_units=40000))
        assert result.ok, result.problem
        assert result.edges[0].source_id == entity.key
        assert result.accounting.context_restarts == 2
        assert len(requests[-1]['input']) == 1  # A new valid request, with no orphan tool result.
        source = json.loads(requests[-1]['input'][0]['content'])
        assert source['saved_lookup_results'][0]['evidence'][0]['quote'] == PAST
        assert source['saved_lookup_results'][0]['evidence'][0]['location']['source_excerpt'] == PAST
        assert result.accounting.requests_attempted == len(requests) == 3
    finally:
        provider.client.close()


def test_long_multilingual_evidence_pages_authorize_only_after_complete_delivery():
    from osintgpt.graph.identity_session import LookupState
    state = LookupState(IdentityTools(EntityRegistry([])))
    original = {'status': 'ok', 'entity_id': 'old', 'evidence': [{'quote': '证据 sin traducción. ' * 3000}]}
    state.entries.append({'name': 'read_entity_context', 'arguments': {}, 'result': original})
    offset, texts = 0, []
    while True:
        page = state.read({'index': 0, 'offset': offset, 'limit': 997})
        state.acknowledge(page, 0)
        texts.append(page['text'])
        offset = page['next_offset']
        if page['complete']:
            break
        assert not state.tools.evidence
    assert json.loads(''.join(texts))['result'] == original
    assert state.tools.evidence['old'] == {original['evidence'][0]['quote']}


@pytest.mark.parametrize('known_capacity', [True, False])
def test_long_evidence_crosses_native_restarts_without_losing_original_locations(known_capacity):
    from test_extraction_quality import Script
    quote = PAST + ' 证据 sin traducción.' * 2500
    entity = extract_document(Script({'entities': [{'id': 'old', 'name': 'Orion', 'type': 'organization',
        'evidence': [quote]}], 'edges': []}), 'past.md', quote, window_chars=len(quote) + 1,
        options=ExtractionOptions(context_budget=200000)).entities[0]
    pages = []
    def turn(index, request):
        if index == 0:
            return [search('Orion')]
        if index == 1:
            return [lookup('read_entity_context', {'entity_id': entity.key, 'offset': 0, 'limit': 10})]
        seed = json.loads(request['input'][0]['content'])
        delivered = seed.get('saved_lookup_results', [])
        delivered += [json.loads(item['output']) for item in request['input'] if item.get('type') == 'function_call_output']
        page = next((item for item in reversed(delivered) if item.get('index') == 1 and 'text' in item), None)
        if page:
            pages.append(page['text'])
            if page['complete']:
                return proposal(entity.key, candidate_quote=quote)
        return [lookup('read_saved_lookup', {'index': 1, 'offset': page['next_offset'] if page else 0, 'limit': 50000}, index)]
    provider, requests = scripted('responses', turn, continuation_size=44000 if known_capacity else 0,
                                  input_limit=None if known_capacity else 40000)
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity],
                                  options=ExtractionOptions(model_input_units=40000 if known_capacity else None))
        assert result.ok and result.edges, result.problem or result.rejected
        assert result.edges[0].source_id == entity.key
        recovered = json.loads(''.join(pages))['result']['evidence'][0]
        assert recovered['quote'] == recovered['location']['source_excerpt'] == quote
        assert recovered['location']['start'] == 0 and recovered['location']['end'] == len(quote)
        assert len(pages) > 1 and result.accounting.context_restarts > 0
        assert result.accounting.requests_failed == 0 if known_capacity else result.accounting.requests_failed > 0
    finally:
        provider.client.close()
