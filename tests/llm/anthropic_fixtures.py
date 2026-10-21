'''Native Anthropic SSE fixtures shared by graph and adapter tests.'''

import json


def message_stream(body, *, complete=True):
    message = {**body, 'content': [], 'stop_reason': None,
               'usage': {**body['usage'], 'output_tokens': 0}}
    events = [('message_start', {'message': message})]
    for index, block in enumerate(body['content']):
        if block['type'] == 'tool_use':
            events.extend([
                ('content_block_start', {'index': index, 'content_block': {**block, 'input': {}}}),
                ('content_block_delta', {'index': index, 'delta': {'type': 'input_json_delta',
                                                                 'partial_json': json.dumps(block['input'])}}),
                ('content_block_stop', {'index': index}),
            ])
            continue
        events.extend([
            ('content_block_start', {'index': index, 'content_block': {**block, 'text': ''}}),
            ('content_block_delta', {'index': index, 'delta': {'type': 'text_delta', 'text': block['text']}}),
            ('content_block_stop', {'index': index}),
        ])
    events.append(('message_delta', {'delta': {'stop_reason': body['stop_reason'], 'stop_sequence': None},
                                     'usage': {'output_tokens': body['usage']['output_tokens']}}))
    if complete:
        events.append(('message_stop', {}))
    return ''.join(f'event: {kind}\ndata: {json.dumps(dict(type=kind, **data))}\n\n' for kind, data in events).encode()


def model_metadata(model='offline', output=64000, context=200000):
    return {'id': model, 'type': 'model', 'display_name': model, 'created_at': '2026-01-01T00:00:00Z',
            'max_tokens': output, 'max_input_tokens': context}
