'''Provider message continuations for durable generation with read-only tools.'''

import json


def generation_arguments(kind, system, user, *, schema=None, tools=None, history=None):
    '''One message/schema serializer for both graph planning and provider sending.'''
    if kind == 'responses':
        result = {'store': False, 'instructions': system, 'input': responses_input(user, history)}
        if schema is not None:
            result['text'] = {'format': {'type': 'json_schema', 'name': 'graph_response', 'strict': True, 'schema': schema}}
        if tools:
            result['tools'] = [{'type': 'function', **tool} for tool in tools]
        if tools or history:
            result['include'] = ['reasoning.encrypted_content']
        return result
    if kind == 'anthropic':
        result = {'system': system, 'messages': anthropic_messages(user, history)}
        if schema is not None:
            result['output_config'] = {'format': {'type': 'json_schema', 'schema': schema}}
        if tools:
            result['tools'] = [{'name': t['name'], 'description': t['description'], 'input_schema': t['parameters']} for t in tools]
        return result
    result = {'messages': chat_messages(system, user, history)}
    if schema is not None:
        result['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'graph_response', 'strict': True, 'schema': schema}}
    if tools:
        result['tools'] = [{'type': 'function', 'function': t} for t in tools]
        if any(t['name'] == 'submit_extraction' for t in tools):
            result['tool_choice'] = 'required'
    return result


def arguments(raw):
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or '{}')
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def responses_input(user, history):
    items = [{'role': 'user', 'content': user}]
    for exchange in history or []:
        items.extend(exchange['response']['continuation'])
        items.extend({'type': 'function_call_output', 'call_id': call['id'],
                      'output': exchange['results'][call['id']]}
                     for call in exchange['response']['tool_calls'])
    return items


def chat_messages(system, user, history):
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    for exchange in history or []:
        messages.extend(exchange['response']['continuation'])
        messages.extend({'role': 'tool', 'tool_call_id': call['id'],
                         'content': exchange['results'][call['id']]}
                        for call in exchange['response']['tool_calls'])
    return messages


def anthropic_messages(user, history):
    messages = [{'role': 'user', 'content': user}]
    for exchange in history or []:
        messages.append({'role': 'assistant', 'content': exchange['response']['continuation']})
        messages.append({'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': call['id'], 'content': exchange['results'][call['id']]}
            for call in exchange['response']['tool_calls']]})
    return messages
