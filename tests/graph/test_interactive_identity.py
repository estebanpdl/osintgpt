'''Native SDK tool sequences, evidence-based reuse, and durable lookup recovery.'''

import json
from dataclasses import replace

import httpx
import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI

from osintgpt.graph import extract_document, GraphStore
from osintgpt.graph.checkpoints import Checkpoints
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.identity_tools import IdentityTools
from osintgpt.graph.identity import EntityRegistry
from osintgpt.graph.windowing import ExtractionOptions
from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import text_spans
from osintgpt.ingestion.snapshots import TextSnapshot
from tests.conftest import FAKE_KEY
from tests.llm.test_generation_contract import provider_fixture
from tests.llm.anthropic_fixtures import message_stream, model_metadata
from test_extraction_quality import Script

PAST = 'Orion, founded by Mira, builds Prism.'
NOW = PAST + ' Orion supports Lyra.'


def prior_entity():
    data = {'entities': [{'id': 'one', 'name': 'Orion', 'type': 'organization', 'evidence': [PAST],
                          'description': 'Long historical description. ' * 300}], 'edges': []}
    return extract_document(Script(data), 'past.md', PAST).entities[0]


def proposal(identity, *, candidate_quote=PAST, evidence=PAST):
    def row(local, name):
        return {'id': local, 'name': name, 'type': 'organization', 'description': '', 'evidence': [name],
                'existing_id': '', 'aliases': [], 'identifiers': [],
                'resolution': {'action': 'create', 'method': 'new', 'reason': 'New source mention.',
                               'evidence': [], 'candidate_evidence': [], 'type_compatible': False, 'type_reason': ''}}
    first, second = row('a', 'Orion'), row('b', 'Lyra')
    first['existing_id'] = identity
    first['resolution'] = {'action': 'reuse', 'method': 'context', 'reason': 'Same named founder and specific product.',
                           'evidence': [evidence], 'candidate_evidence': [candidate_quote],
                           'type_compatible': True, 'type_reason': 'Both passages identify an organization.'}
    return {'entities': [first, second], 'edges': [{'source': 'a', 'target': 'b', 'relation': 'supports',
            'evidence': ['Orion supports Lyra.'], 'relation_polarity': 'positive', 'assertion': 'stated',
            'attributed_to': '', 'event_time': '', 'statement_time': ''}]}


def native_lookup(kind, identity, *, final=None):
    provider, _ = provider_fixture(kind)
    provider.client.close()
    requests = []
    api = httpx2 if kind == 'anthropic' else httpx
    turns = [
        {'id': 'lookup1', 'name': 'find_entity_candidates',
         'arguments': {'queries': ['Orion'], 'entity_type': '', 'offset': 0, 'limit': 10}},
        {'id': 'lookup2', 'name': 'read_entity_context',
         'arguments': {'entity_id': identity, 'offset': 0, 'limit': 10}},
    ]
    def respond(request):
        if request.method == 'GET':
            return api.Response(200, json=model_metadata())
        data = json.loads(request.content)
        requests.append(data)
        index = len(requests) - 1
        call = turns[index] if index < 2 else None
        text = json.dumps(final or proposal(identity)) if call is None else ''
        if kind == 'responses':
            output = ([{'type': 'reasoning', 'id': f'rs{index}', 'summary': []},
                       {'type': 'function_call', 'id': f'fc{index}', 'call_id': call['id'],
                        'name': call['name'], 'arguments': json.dumps(call['arguments'])}] if call else
                      [{'type': 'message', 'id': 'msg', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}])
            body = {'id': f'resp{index}', 'object': 'response', 'created_at': 0, 'model': 'offline',
                    'status': 'completed', 'output': output,
                    'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}
        elif kind == 'anthropic':
            body = {'id': 'msg', 'type': 'message', 'role': 'assistant', 'model': 'offline',
                    'content': [{'type': 'tool_use', 'id': call['id'], 'name': call['name'], 'input': call['arguments']}]
                    if call else [{'type': 'text', 'text': text}], 'stop_reason': 'tool_use' if call else 'end_turn',
                    'stop_sequence': None, 'usage': {'input_tokens': 10, 'output_tokens': 5}}
            return api.Response(200, headers={'content-type': 'text/event-stream'}, content=message_stream(body))
        else:
            message = {'role': 'assistant', 'content': text or None}
            if call is None and any(tool.get('function', {}).get('name') == 'submit_extraction' for tool in data.get('tools', [])):
                call = {'id': 'submit', 'name': 'submit_extraction', 'arguments': json.loads(text)}
                message['content'] = None
            if call:
                message['tool_calls'] = [{'id': call['id'], 'type': 'function', 'function': {
                    'name': call['name'], 'arguments': json.dumps(call['arguments'])}}]
            body = {'id': 'chat', 'object': 'chat.completion', 'created': 0, 'model': 'offline',
                    'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if call else 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        return api.Response(200, json=body)
    client = api.Client(transport=api.MockTransport(respond))
    provider.client = (Anthropic(api_key=FAKE_KEY, http_client=client) if kind == 'anthropic' else
                       OpenAI(api_key=FAKE_KEY, http_client=client))
    return provider, requests


@pytest.mark.parametrize('kind', ['responses', 'gemini', 'anthropic', 'ollama'])
def test_native_search_read_and_contextual_reuse_are_accounted(kind):
    entity = prior_entity()
    provider, requests = native_lookup(kind, entity.key)
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity])
        assert result.ok, result.problem
        assert len(result.edges) == 1 and result.edges[0].source_id == entity.key
        decision = result.entities[0].provenance['resolution']
        assert decision['action'] == 'reuse' and decision['method'] == 'context'
        assert len(result.entities[0].provenance['mentions']) == 2
        assert result.accounting.requests_attempted == result.accounting.requests_succeeded == len(requests) == 3
        assert result.accounting.entity_lookups == 2 and provider.recorder.total_tokens == 45
        first = requests[0]
        user = first['input'][0]['content'] if kind == 'responses' else first['messages'][-1]['content']
        assert json.loads(user)['known_entities'] == []  # discovered beyond the initial bounded context
        if kind == 'responses':
            assert any(item.get('type') == 'reasoning' for item in requests[1]['input'])
            assert any(item.get('type') == 'function_call_output' for item in requests[1]['input'])
            assert all('max_output_tokens' not in item and item['store'] is False for item in requests)
        elif kind == 'anthropic':
            assert requests[1]['messages'][-1]['content'][0]['type'] == 'tool_result'
            assert all(item['max_tokens'] == 64000 and item['stream'] for item in requests)
        else:
            assert requests[1]['messages'][-1]['role'] == 'tool'
            assert all('max_tokens' not in item for item in requests)
            assert all('response_format' not in item and item['tool_choice'] == 'required' for item in requests)
    finally:
        provider.client.close()


def test_interrupted_lookup_replays_response_and_saved_local_results(tmp_path):
    entity = prior_entity()
    provider, requests = native_lookup('responses', entity.key)
    doc = Document('new.md', NOW, spans=text_spans(NOW))
    snapshot = TextSnapshot('new.md', 'hash', 'version', (doc,))
    with GraphStore(tmp_path / 'graph.sqlite') as graph:
        graph.add([entity], [])
        journal = Checkpoints(graph)
        journal.prepare('new.md', 'version', 'recipe', {})
        def stop(counts):
            if counts.requests_succeeded == 2:
                raise GraphCancelled('Pause after second response is saved.')
        try:
            with pytest.raises(GraphCancelled):
                extract_document(provider, doc.ref, NOW, snapshot=snapshot, document=doc,
                                 known_entities=[entity], checkpoints=journal, on_progress=stop)
            before = len(requests)
            resumed = extract_document(provider, doc.ref, NOW, snapshot=snapshot, document=doc,
                                       known_entities=[entity], checkpoints=journal)
            assert resumed.ok, resumed.problem
            assert before == 2 and len(requests) == 3
            assert resumed.accounting.responses_reused == 2
            assert resumed.accounting.entity_lookups_reused == 1
            assert resumed.edges[0].source_id == entity.key
        finally:
            provider.client.close()


def test_fabricated_candidate_evidence_cannot_merge_but_preserves_the_claim():
    entity = prior_entity()
    provider, _ = native_lookup('gemini', entity.key, final=proposal(entity.key, candidate_quote='Invented identity evidence.'))
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity])
        assert result.ok and len(result.edges) == 1
        assert result.edges[0].source_id != entity.key
        assert result.entities[0].provenance['resolution']['action'] == 'unresolved'
    finally:
        provider.client.close()


def test_tools_are_read_only_and_pagination_discloses_missing_results():
    entity = prior_entity()
    entities = [replace(entity, key=f'ent_{i}', description='') for i in range(40)]
    lookup = IdentityTools(EntityRegistry(entities), 4096)
    result = lookup.run('find_entity_candidates', {'queries': ['Orion'], 'limit': 2, 'offset': 0})['results'][0]
    assert not result['complete'] and result['total'] == 40 and result['next_offset'] == 2
    page = lookup.run('find_entity_candidates', {'queries': ['Orion'], 'limit': 2, 'offset': 2})['results'][0]
    assert {item['id'] for item in result['candidates']}.isdisjoint(item['id'] for item in page['candidates'])
    assert lookup.run('execute_sql', {'sql': 'DELETE FROM entities'})['status'] == 'invalid_arguments'
    assert lookup.run('read_entity_context', {'entity_id': 'invented'})['status'] == 'invalid_arguments'


def test_long_candidate_quotation_pages_without_losing_source_offsets():
    text = 'Orion ' + 'contexto ' * 1500 + 'founded by Mira.'
    entity = extract_document(Script({'entities': [{'id': 'a', 'name': 'Orion', 'evidence': [text]}],
                                     'edges': []}), 'long.md', text).entities[0]
    tools = IdentityTools(EntityRegistry([entity]), 1600)
    tools.allowed.add(entity.key)
    first = tools.run('read_entity_context', {'entity_id': entity.key, 'offset': 0, 'limit': 1})
    assert first['evidence'] and first['next_offset'] == 0 and first['next_quote_offset'] > 0
    second = tools.run('read_entity_context', {'entity_id': entity.key, 'offset': first['next_offset'],
                                             'quote_offset': first['next_quote_offset'], 'limit': 1})
    for page in (first, second):
        row = page['evidence'][0]
        assert row['quote'] == text[row['location']['start']:row['location']['end']]
        assert row['partial'] and not page['complete']
    assert second['evidence'][0]['location']['start'] == first['evidence'][0]['location']['end']


@pytest.mark.parametrize('kind,compatible,expected', [('company', True, 'reuse'), ('campaign', False, 'unresolved')])
def test_type_decision_distinguishes_broad_categories_from_different_referents(kind, compatible, expected):
    entity = prior_entity()
    data = proposal(entity.key)
    data['entities'][0]['type'] = kind
    data['entities'][0]['resolution'].update(type_compatible=compatible,
        type_reason='Compatible organization category.' if compatible else 'A campaign is a different referent.')
    provider, _ = native_lookup('responses', entity.key, final=data)
    try:
        result = extract_document(provider, 'new.md', NOW, known_entities=[entity])
        assert result.ok and len(result.edges) == 1
        assert result.entities[0].provenance['resolution']['action'] == expected
        assert (result.edges[0].source_id == entity.key) is compatible
    finally:
        provider.client.close()
