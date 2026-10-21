'''Completion metadata for callers that must reject partial structured responses.'''

from dataclasses import dataclass, replace

from .usage import Usage


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    status: str = ''
    stop_reason: str = ''
    refused: bool = False
    usage: Usage | None = None
    tool_calls: tuple = ()
    continuation: tuple = ()

    @property
    def has_completion_status(self):
        return bool(self.status or self.stop_reason or self.refused)

    def require_complete(self, *, allow_tools=False):
        if self.refused:
            raise ValueError('Generation provider refused the structured response.')
        if self.stop_reason in {'length', 'max_tokens', 'max_output_tokens', 'model_context_window_exceeded'}:
            raise ValueError(f'Generation response was truncated ({self.stop_reason}); '
                             'the provider did not finish the structured response. Use smaller source windows '
                             'or a model with more output capacity.')
        if self.status and self.status != 'completed':
            raise ValueError(f'Generation response is not complete ({self.status}: {self.stop_reason or "unknown"}).')
        permitted = {'stop', 'end_turn'} | ({'tool_calls', 'tool_use'} if allow_tools and self.tool_calls else set())
        if self.stop_reason and self.stop_reason not in permitted:
            raise ValueError(f'Generation response stopped without completion ({self.stop_reason}).')
        if self.tool_calls and not allow_tools:
            raise ValueError('Generation requested tools instead of completing the structured response.')
        if self.tool_calls:
            ids = set()
            for call in self.tool_calls:
                if (not isinstance(call, dict) or not isinstance(call.get('id'), str) or not call['id']
                        or call['id'] in ids or not isinstance(call.get('name'), str) or not call['name']
                        or not isinstance(call.get('arguments'), dict)):
                    raise ValueError('Generation returned malformed or duplicate tool calls.')
                ids.add(call['id'])
        return self.text


def response_from(data):
    usage = data.get('usage')
    return GenerationResponse(**{**data, 'usage': Usage(**usage) if usage is not None else None,
                                 'tool_calls': tuple(data.get('tool_calls', ())),
                                 'continuation': tuple(data.get('continuation', ()))})


def deliver_response(provider, response, usage, on_response=None):
    response = replace(response, usage=usage)
    try:
        if on_response is not None:
            on_response(response)
    finally:
        provider._record(usage)
    return response


def send_request(client, operation, request=None):
    if request is None:
        return operation(client)
    # Graph generation can take substantially longer than an embedding batch.
    # Retry elapsed time controls new attempts, not an active response; streaming
    # uses this as an idle/read timeout rather than a total-generation deadline.
    controlled = client.with_options(max_retries=0, timeout=600)
    return request(lambda: operation(controlled))
