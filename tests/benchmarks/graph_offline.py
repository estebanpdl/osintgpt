'''Local native-SDK transports and synthetic embeddings; no provider network requests.'''

import json

import httpx
from openai import OpenAI

from osintgpt.llm.openai_responses import OpenAIResponsesGeneration
from osintgpt.llm.usage import Usage
from graph_passages import FixtureEmbedder


class RecordedFixtureEmbedder(FixtureEmbedder):
    provider = 'offline-fixture'

    def __init__(self, recorder):
        self.recorder = recorder

    def embed(self, texts, **kwargs):
        texts = list(texts)
        self.recorder.record(Usage(self.provider, self.model, sum(len(text) for text in texts), billable=False))
        return super().embed(texts, **kwargs)


def fixture_proposal(item, payload):
    if 'source' not in item:
        return {'entities': [], 'edges': []}
    entities = []
    for side in ('source', 'target'):
        name = item[side]
        known = [entity['id'] for entity in payload['known_entities']
                 if entity['name'] == name and entity['document_ref'] == payload['document_ref']]
        entities.append({'id': side, 'name': name, 'type': '', 'description': '', 'aliases': [], 'identifiers': [],
                         'existing_id': known[0] if len(known) == 1 and not item.get(side + '_id') else '', 'evidence': [name]})
    return {'entities': entities, 'edges': [{'source': 'source', 'target': 'target', 'relation': item['relation'],
        'evidence': [item['text']], 'relation_polarity': item.get('relation_polarity', 'positive'), 'assertion': item.get('assertion', 'stated'),
        **{key: item.get(key, '') for key in ('attributed_to', 'event_time', 'statement_time')}}]}


def with_resolution(proposed, request):
    '''Keep scripted fixtures compatible with the requested extraction contract.'''
    schema = request.get('text', {}).get('format', {}).get('schema', {})
    if 'resolution' in schema.get('properties', {}).get('entities', {}).get('items', {}).get('properties', {}):
        for entity in proposed.get('entities', []):
            entity['resolution'] = {'action': 'create', 'method': 'new', 'reason': 'Scripted sourced mention.',
                'evidence': [], 'candidate_evidence': [], 'type_compatible': False, 'type_reason': ''}
    return proposed


def generation(fixture, recorder, *, role, model):
    documents = {item['ref']: item for item in fixture['documents']}
    def handle(request):
        body = json.loads(request.content)
        user = body['input'][0]['content']
        if role == 'extraction':
            payload = json.loads(user)
            proposed = fixture_proposal(documents[payload['document_ref']], payload)
            text = json.dumps(with_resolution(proposed, body), ensure_ascii=False)
        else:
            text = 'Scripted offline answer [1]. Interpretation requires a real model and review.'
        return httpx.Response(200, json={'id': 'offline', 'object': 'response', 'created_at': 0,
            'model': body['model'], 'status': 'completed', 'incomplete_details': None,
            'output': [{'id': 'message', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}],
            'usage': {'input_tokens': len(request.content) // 4, 'output_tokens': len(text) // 4,
                      'total_tokens': len(request.content) // 4 + len(text) // 4}})
    provider = OpenAIResponsesGeneration(model, 'offline-fixture-key', recorder=recorder)
    provider.client.close()
    provider.client = OpenAI(api_key='offline-fixture-key', base_url='https://offline.invalid/v1',
                            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    return provider
