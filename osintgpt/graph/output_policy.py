'''Graph output policy, separate from source sizing and quota reservations.'''

from dataclasses import asdict, dataclass, replace
import json


@dataclass(frozen=True)
class OutputPolicy:
    mode: str = 'provider_default'
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    resolved_model: str = ''

    def __post_init__(self):
        if self.mode not in {'provider_default', 'model_maximum'}:
            raise ValueError('Unknown graph output policy.')
        for value in (self.max_tokens, self.max_input_tokens):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError('Model token limits must be positive integers.')
        if self.mode == 'model_maximum' and (self.max_tokens is None) != (self.max_input_tokens is None):
            raise ValueError('Anthropic requires both model input and output limits.')

    @property
    def resolved(self):
        return self.mode != 'model_maximum' or self.max_tokens is not None

    def options(self, options):
        if self.max_input_tokens is None:
            return options
        # Reserve the provider output allowance against model capacity, never
        # against the application's independent source-input budget.
        room = self.max_input_tokens - self.max_tokens
        if room <= 0:
            raise ValueError('Model metadata leaves insufficient context for graph input and output.')
        return replace(options, model_input_units=room)

    def reservation(self, input_units):
        # With provider-default output, use an explicit pacing estimate, not
        # an enforced generation limit or a claimed spending upper bound.
        return input_units + (self.max_tokens or input_units) + 512

    def label(self):
        if self.mode == 'provider_default':
            return 'Provider default; no application output-token ceiling.'
        if not self.resolved:
            return 'Anthropic model output allowance will be resolved when the build starts.'
        return f'Anthropic model output allowance: {self.max_tokens:,} tokens (automatic).'


def identity(generator):
    getter = getattr(generator, 'graph_options', None)
    if getter:
        return getter()
    from .recipes import fingerprint
    return {'implementation': f'{type(generator).__module__}.{type(generator).__qualname__}',
            'model': getattr(generator, 'model', ''),
            'endpoint': fingerprint(str(getattr(getattr(generator, 'client', None), 'base_url', ''))),
            'structured_output': bool(getattr(generator, 'supports_structured_output', False)),
            'provider': getattr(generator, 'provider', '')}


def policy_for(generator):
    saved = getattr(generator, '_graph_output_policy', None)
    if saved is not None and getattr(generator, '_graph_policy_identity', None) == identity(generator):
        return saved
    return OutputPolicy(mode='model_maximum' if identity(generator).get('provider') == 'anthropic'
                        else 'provider_default')


def prepare_policy(generator, entries=(), *, resolve=False):
    '''Reuse persisted exact-model metadata offline; only a build may fetch it.'''
    policy = policy_for(generator)
    wanted = identity(generator)
    if not policy.resolved:
        for entry in entries:
            recipe = entry.get('recipe_data', {})
            recipe = json.loads(recipe) if isinstance(recipe, str) else recipe
            if recipe.get('provider') == wanted and recipe.get('output_policy'):
                candidate = OutputPolicy(**recipe['output_policy'])
                if candidate.mode == policy.mode and candidate.resolved:
                    policy = candidate
                    break
    if not policy.resolved and resolve:
        from .calling import generation_provider
        provider = generation_provider(generator)
        loader = getattr(provider, 'graph_model_limits', None)
        if not callable(loader):
            raise ValueError('Cannot resolve Anthropic graph model limits before extraction.')
        policy = OutputPolicy(mode='model_maximum', **loader())
        if not policy.resolved:
            raise ValueError('Anthropic model metadata is missing input/output limits; no extraction was sent.')
    generator._graph_output_policy = policy
    generator._graph_policy_identity = wanted
    return policy


def policy_data(generator):
    return asdict(policy_for(generator))
