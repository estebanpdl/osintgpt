'''Resolve and route graph models without borrowing chat configuration or constructing clients early.'''

from dataclasses import replace
from types import SimpleNamespace

import pytest

from osintgpt.config import Settings
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.graph.provider import LazyGraphGenerator, resolve_graph_model
from osintgpt.graph.recipes import extraction_recipe, fingerprint
from osintgpt.llm import build_generation_provider
from osintgpt.llm.registry import BackendSpec, GENERATION_BACKENDS, OPENAI_COMPAT
from osintgpt.projects import ProjectSettings


def choices(**changes):
    return replace(ProjectSettings(generation_provider='openai', generation_model='chat-model',
                                   graph_generation_provider='anthropic', graph_generation_model='graph-model'), **changes)


@pytest.mark.parametrize('mode,provider,model', [('inherit', 'openai', 'chat-model'),
                                               ('independent', 'anthropic', 'graph-model')])
def test_explicit_mode_resolves_its_own_pair(mode, provider, model):
    selected = resolve_graph_model(choices(graph_model_mode=mode), Settings(openai_gpt_model='environment-chat'))
    assert (selected.provider, selected.model, selected.mode) == (provider, model, mode)


def test_invocation_model_override_keeps_selected_provider():
    selected = resolve_graph_model(choices(), Settings(), model='one-run-model')
    assert (selected.provider, selected.model, selected.mode) == ('anthropic', 'one-run-model', 'override')


def test_invocation_pair_can_fix_invalid_persistent_configuration():
    selected = resolve_graph_model(choices(graph_generation_provider='not-registered'), Settings(),
                                    provider='ollama', model='local-extractor')
    assert selected.provider == 'ollama' and selected.model == 'local-extractor'


def test_provider_only_override_requires_model_instead_of_borrowing_chat():
    with pytest.raises(ValueError, match='no model given for the ollama'):
        resolve_graph_model(choices(), Settings(openai_gpt_model='chat-model'), provider='ollama')


def test_independent_missing_fields_do_not_fall_back_to_chat():
    with pytest.raises(ValueError, match='requires graph_generation_provider'):
        resolve_graph_model(choices(graph_generation_provider=''), Settings())
    with pytest.raises(ValueError, match='no model given for the anthropic'):
        resolve_graph_model(choices(graph_generation_model=''), Settings(openai_gpt_model='chat-model'))


@pytest.mark.parametrize('override', [{'model': ''}, {'provider': ''}, {'model': 123}])
def test_invalid_overrides_are_rejected_without_client(override):
    with pytest.raises(ValueError, match='override must be non-empty text'):
        resolve_graph_model(choices(), Settings(), **override)


def test_registry_default_and_custom_endpoint_match_real_provider(monkeypatch):
    monkeypatch.setitem(GENERATION_BACKENDS, 'custom', BackendSpec(OPENAI_COMPAT,
                        base_url='http://graph.invalid/v1', default_model='default-extractor', local=True))
    generator = LazyGraphGenerator(build_generation_provider, choices(), Settings(), None, provider='custom')
    assert generator.graph_options()['model'] == 'default-extractor'
    assert generator.graph_options()['endpoint'] == fingerprint('http://graph.invalid/v1')
    assert generator._built is None
    provider = generator.provider
    try:
        assert provider.model == 'default-extractor'
        assert str(provider.client.base_url) == 'http://graph.invalid/v1/'
    finally:
        provider.client.close()


def test_missing_graph_credentials_cannot_use_available_chat_credentials():
    generator = LazyGraphGenerator(build_generation_provider, choices(), Settings(openai_api_key='test-chat-key'), None)
    assert generator.graph_options()['provider'] == 'anthropic'
    with pytest.raises(MissingEnvironmentVariableError, match='ANTHROPIC_API_KEY'):
        generator.provider
    assert generator._built is None


def test_inheritance_retains_existing_recipe_identity():
    effective = ProjectSettings(generation_provider='openai', generation_model='chat-model')
    generator = LazyGraphGenerator(None, effective, Settings(), None)
    expected = {'provider': 'openai', 'implementation': GENERATION_BACKENDS['openai'].kind,
                'model': 'chat-model', 'endpoint': fingerprint('')}
    assert generator.graph_options() == expected
    pinned = LazyGraphGenerator(None, replace(effective, graph_model_mode='independent',
                                graph_generation_provider='openai', graph_generation_model='chat-model'), Settings(), None)
    assert extraction_recipe(generator) == extraction_recipe(pinned)


def test_chat_changes_and_selection_labels_do_not_change_independent_recipe():
    first = LazyGraphGenerator(None, choices(), Settings(openai_gpt_model='first-chat'), None)
    second = LazyGraphGenerator(None, choices(generation_provider='ollama', generation_model='other-chat'),
                                Settings(openai_gpt_model='second-chat'), None)
    assert extraction_recipe(first) == extraction_recipe(second)
    third = LazyGraphGenerator(None, choices(), Settings(), None, provider='anthropic', model='graph-model')
    assert extraction_recipe(third) == extraction_recipe(first)


def test_lazy_generator_snapshots_mutable_caller_settings():
    effective = SimpleNamespace(generation_provider='openai', generation_model='start-model')
    created = []
    generator = LazyGraphGenerator(lambda *a, **k: created.append((a, k)) or SimpleNamespace(),
                                   effective, Settings(openai_gpt_model='fallback'), None)
    effective.generation_model = 'changed-before-start'
    assert generator.model == 'start-model' and not created
    generator.provider
    assert created[0][1]['model'] == 'start-model'


def test_unknown_saved_model_is_not_treated_as_reusable():
    generator = LazyGraphGenerator(None, ProjectSettings(), Settings(), None)
    with pytest.raises(ValueError, match='no model given'):
        generator.graph_options()
