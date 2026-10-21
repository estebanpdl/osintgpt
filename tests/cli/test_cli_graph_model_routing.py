'''Existing graph commands consume independent project settings and stored provider credentials.'''

import json
from types import SimpleNamespace

from osintgpt.config import Settings
from osintgpt.credentials import save_credentials
from osintgpt.graph.provider import resolve_graph_model
from osintgpt.llm.generation import GenerationResponse
from osintgpt.ingestion import Corpus
from test_cli_graph import create_project, home, invoke, runner


def test_cli_build_and_preview_select_graph_pair_and_stored_credentials(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True, generation_model='chat-A',
        graph_model_mode='independent', graph_generation_provider='anthropic', graph_generation_model='graph-B')
    project.save()
    (project.paths.root / 'source.md').write_text('Alpha supports Beta.')
    Corpus.load(project.paths.sources).register('source.md')
    save_credentials(home, {'anthropic_api_key': 'offline-stored-key'})
    selected = []
    def factory(provider, config, *, model, recorder):
        selected.append((provider, model, config.anthropic_api_key))
        return SimpleNamespace(model=model, supports_structured_output=True,
            graph_model_limits=lambda: {'max_tokens': 64000, 'max_input_tokens': 200000, 'resolved_model': model},
            generate_response=lambda *a, **k: GenerationResponse('{"entities":[],"edges":[]}', status='completed'))
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', factory)
    preview = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--dry-run', '--json')
    assert preview.exit_code == 0, preview.output
    data = json.loads(preview.stdout)
    assert data['provider']['provider'] == 'anthropic' and data['provider']['model'] == 'graph-B'
    assert not selected and 'offline-stored-key' not in preview.stdout
    result = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--json')
    assert result.exit_code == 0, result.output
    assert selected == [('anthropic', 'graph-B', 'offline-stored-key')]
    project.with_settings(generation_model='chat-C', generation_provider='ollama').save()
    selected.clear()
    repeated = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--incremental', '--json')
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.stdout)['calls'] == 0 and not selected


def test_existing_config_commands_persist_pair_and_explicit_inherit(runner, home):
    from osintgpt.projects import Project
    project = create_project(runner, home)
    for key, value in [('graph_generation_provider', 'anthropic'), ('graph_generation_model', 'graph-B'),
                       ('graph_model_mode', 'independent')]:
        result = invoke(runner, home, 'config', 'set', key, value, '--project', project.slug, '--json')
        assert result.exit_code == 0, result.output
    loaded = Project.load(project.paths.root)
    assert resolve_graph_model(loaded.effective_settings(), Settings()).model == 'graph-B'
    result = invoke(runner, home, 'config', 'set', 'graph_model_mode', 'inherit', '--project', project.slug, '--json')
    assert result.exit_code == 0, result.output
    loaded = Project.load(project.paths.root)
    assert resolve_graph_model(loaded.effective_settings(), Settings()).model == 'test-chat'


def test_missing_graph_model_fails_before_client_without_chat_fallback(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True, graph_model_mode='independent',
                                                       graph_generation_provider='anthropic')
    project.save()
    def forbidden(*a, **k):
        raise AssertionError('Invalid configuration must not construct a provider')
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)
    result = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--dry-run', '--json')
    assert result.exit_code == 1
    assert 'no model given for the anthropic' in json.loads(result.stdout)['error']
    assert not (project.paths.root / 'graph.sqlite').exists()
