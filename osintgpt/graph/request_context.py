'''Provider-shaped input measurement; source sizing is never a conversation ceiling.'''

from .windowing import encode, units


class ProviderCapacityError(ValueError):
    pass


class RequestContext:
    def __init__(self, provider, options):
        self.provider, self.options = provider, options
        self.capacity = options.model_input_units

    def arguments(self, system, user, schema=None, tools=None, history=None):
        build = getattr(self.provider, 'graph_request', None)
        return (build(system, user, schema=schema, tools=tools, history=history) if callable(build)
                else {'system': system, 'user': user, 'schema': schema, 'tools': tools, 'history': history})

    def measure(self, system, user, schema=None, tools=None, history=None):
        return units(encode(self.arguments(system, user, schema, tools, history)))

    def fits(self, system, user, schema=None, tools=None, history=None):
        return self.capacity is None or self.measure(system, user, schema, tools, history) <= self.capacity

    def check(self, system, user, schema=None, tools=None, history=None):
        measured = self.measure(system, user, schema, tools, history)
        if self.capacity is not None and measured > self.capacity:
            raise ProviderCapacityError(f'Provider input allowance cannot fit this request: {measured} conservative UTF-8 '
                                        f'input units, {self.capacity} available. Saved work is preserved; '
                                        'automatic source/context partitioning is required, not a larger local byte ceiling.')
        return measured

    def should_restart(self, system, user, schema, tools, history):
        # A soft scheduling target derived from source sizing, never a rejection limit.
        target = max(self.options.context_budget, self.measure(system, user, schema, tools))
        if self.capacity is not None:
            target = min(target, self.capacity)
        measured = self.measure(system, user, schema, tools, history)
        return bool(history) and ((self.capacity is not None and measured > self.capacity)
                                  or (len(history) > 1 and measured > target))


def context_failure(error):
    '''Only classified context failures permit a smaller automatic request.'''
    while error is not None:
        code = getattr(error, 'code', '')
        body = getattr(error, 'body', {}) or {}
        if isinstance(body, dict):
            nested = body.get('error')
            code = body.get('code', code) or (nested.get('code', '') if isinstance(nested, dict) else '')
        if code in {'context_length_exceeded', 'context_window_exceeded', 'input_too_long'}:
            return True
        error = error.__cause__
    return False
