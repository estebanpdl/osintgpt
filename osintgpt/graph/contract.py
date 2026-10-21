'''Portable extraction schemas and local validation of their structural subset.'''

import json
import re
from dataclasses import replace

POLARITIES = {'positive', 'negative', 'unknown'}
ASSERTIONS = {'stated', 'alleged', 'uncertain', 'hypothetical', 'unknown'}


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def _array(items):
    return {'type': 'array', 'items': items}


_TEXT = {'type': 'string'}
_QUOTES = _array(_TEXT)
EXTRACTION_SCHEMA = _object({
    'entities': _array(_object({
        'id': _TEXT, 'name': _TEXT, 'type': _TEXT, 'description': _TEXT,
        'evidence': _QUOTES, 'existing_id': _TEXT,
        'aliases': _array(_object({'name': _TEXT, 'evidence': _QUOTES})),
        'identifiers': _array(_object({'issuer': _TEXT, 'value': _TEXT, 'evidence': _QUOTES})),
    })),
    'edges': _array(_object({
        'source': _TEXT, 'target': _TEXT, 'relation': _TEXT, 'evidence': _QUOTES,
        'relation_polarity': {'type': 'string', 'enum': sorted(POLARITIES)},
        'assertion': {'type': 'string', 'enum': sorted(ASSERTIONS)},
        'attributed_to': _TEXT, 'event_time': _TEXT, 'statement_time': _TEXT,
    })),
})
REPAIR_SCHEMA = _object({'corrections': _array(_object({
    'index': {'type': 'integer'}, 'evidence': _QUOTES,
}))})


def legacy_field_request(system, schema, tools):
    '''Reconstruct only the previous field spelling for an exact saved-request lookup.'''
    system = system.replace('"relation_polarity":', '"polarity":').replace(
        'Relation polarity and assertion', 'Polarity and assertion').replace(
        'relation_polarity, and qualifications', 'polarity, and qualifications')
    def previous_schema(value):
        return json.loads(json.dumps(value).replace('"relation_polarity"', '"polarity"'))
    previous_tools = ([{**tool, 'parameters': previous_schema(tool['parameters'])} for tool in tools]
                      if tools is not None else None)
    return system, previous_schema(schema), previous_tools


def legacy_repair_payload(user, purpose):
    """Reconstruct only programmatic repair candidates, never source text or history."""
    if not purpose.startswith('repair'):
        return user
    from .windowing import encode
    payload = json.loads(user)
    candidates = payload.get('candidates', [])
    changed = False
    for item in candidates:
        row = item.get('candidate', {})
        if 'relation_polarity' in row and 'polarity' not in row:
            item['candidate'] = {('polarity' if key == 'relation_polarity' else key): value for key, value in row.items()}
            changed = True
    return encode(payload) if changed else user


def normalize_saved_extraction(response):
    '''Adapt a recovered response in memory; its original audit and request hash remain intact.'''
    from .records import normalize_relation_polarity
    def normalized(data):
        if isinstance(data, dict) and isinstance(data.get('edges'), list):
            return {**data, 'edges': [normalize_relation_polarity(row) if isinstance(row, dict) else row
                                     for row in data['edges']]}
        return data
    text = response.text
    try:
        fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', text)
        data = json.loads(fenced[1] if fenced else text)
    except (TypeError, ValueError):
        pass  # Preserve malformed output for the existing validation path.
    else:
        updated = normalized(data)
        if updated != data:
            text = json.dumps(updated, ensure_ascii=False)
    calls = tuple({**call, 'arguments': normalized(call['arguments'])}
                  if isinstance(call, dict) and call.get('name') == 'submit_extraction' and 'arguments' in call
                  else call for call in response.tool_calls)
    return replace(response, text=text, tool_calls=calls)


def validate_structure(value, schema, path='response'):
    expected = {'object': dict, 'array': list, 'string': str, 'integer': int, 'boolean': bool}[schema['type']]
    if type(value) is not expected:
        raise ValueError(f'{path} must be {schema["type"]}.')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path} is not an allowed value.')
    if isinstance(value, dict):
        properties = schema['properties']
        if set(value) != set(properties):
            raise ValueError(f'{path} fields do not match the requested schema.')
        for name, child in properties.items():
            validate_structure(value[name], child, f'{path}.{name}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_structure(item, schema['items'], f'{path}[{index}]')
