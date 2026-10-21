'''Lazy graph provider construction and credential-free configuration metadata.'''

from copy import deepcopy
from dataclasses import dataclass

from osintgpt.llm.registry import GENERATION_BACKENDS, backend_spec, resolve_base_url, resolve_generation_model
from osintgpt.projects.settings import graph_model_settings

from .recipes import fingerprint


@dataclass(frozen=True)
class GraphModelSelection:
    provider: str
    model: str
    mode: str
    implementation: str
    endpoint: str

    def identity(self):
        return {'provider': self.provider, 'implementation': self.implementation,
                'model': self.model, 'endpoint': self.endpoint}


def resolve_graph_model(effective, config, *, provider=None, model=None):
    '''Resolve the graph pair once, excluding selection mode from extraction identity.'''
    for value, label in ((provider, 'provider'), (model, 'model')):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'Graph {label} override must be non-empty text.')
    configured = graph_model_settings(effective)
    mode = configured['graph_model_mode']
    use_settings = mode == 'inherit'
    if provider is not None:
        name, chosen, use_settings = provider, model, False
    elif use_settings:
        name, chosen = effective.generation_provider, model or effective.generation_model
    else:
        name, chosen = configured['graph_generation_provider'], model or configured['graph_generation_model']
        if not name:
            raise ValueError('Independent graph selection requires graph_generation_provider.')
    spec = backend_spec(name, GENERATION_BACKENDS, 'generation')
    resolved = resolve_generation_model(spec, config, name, chosen, use_settings=use_settings)
    return GraphModelSelection(name, resolved, 'override' if provider is not None or model is not None else mode,
                               spec.kind, fingerprint(resolve_base_url(spec, config) or ''))


class LazyGraphGenerator:
    def __init__(self, factory, effective, config, recorder, *, provider=None, model=None) -> None:
        self.factory = factory
        self.effective = deepcopy(effective)
        self.config = deepcopy(config)
        self.recorder = recorder
        self.overrides = {'provider': provider, 'model': model}
        self._selection = None
        self._built = None

    @property
    def selection(self):
        if self._selection is None:
            self._selection = resolve_graph_model(self.effective, self.config, **self.overrides)
        return self._selection

    @property
    def provider(self):
        if self._built is None:
            self._built = self.factory(
                self.selection.provider,
                self.config,
                model=self.selection.model,
                recorder=self.recorder
            )

        return self._built

    @property
    def model(self) -> str:
        return self.selection.model

    def graph_options(self):
        return self.selection.identity()

    def graph_request(self, system, user, **options):
        '''Serialize a preview through the same adapter helper without creating a client.'''
        from osintgpt.llm.generation_tools import generation_arguments
        kind = {'openai-responses': 'responses', 'anthropic': 'anthropic'}.get(self.selection.implementation, 'chat')
        return generation_arguments(kind, system, user, **options)

    @property
    def graph_submission_tool(self):
        return self.selection.provider in {'gemini', 'ollama'}

    def generate(self, system: str, user: str, *, max_output_tokens=None) -> str:
        return self.generate_response(system, user, max_output_tokens=max_output_tokens).text

    def generate_response(self, system, user, *, max_output_tokens=None, schema=None, on_response=None, request=None,
                          graph_output=None, tools=None, history=None):
        from osintgpt.graph.calling import bounded_generate
        return bounded_generate(self.provider, system, user, max_output_tokens, schema, on_response, request,
                                graph_output, tools, history)

    @property
    def supports_structured_output(self):
        return getattr(self.provider, 'supports_structured_output', False)

    @property
    def supports_output_limit(self):
        from osintgpt.graph.calling import supports_output_limit
        return supports_output_limit(self.provider)


    @property
    def generation_provider(self):
        return self.provider

    @property
    def structured_output_hint(self):
        return self.selection.provider in {'openai', 'anthropic', 'gemini', 'ollama'}

    @property
    def interactive_lookup_hint(self):
        return self.selection.provider in {'openai', 'anthropic', 'gemini', 'ollama'}
