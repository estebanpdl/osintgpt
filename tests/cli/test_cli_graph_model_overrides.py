'''Graph invocation overrides are visible, keyless in previews, and never persisted.'''

import json
from types import SimpleNamespace

import pytest

from osintgpt.ingestion import Corpus
from osintgpt.projects import Project
from test_cli_graph import create_project, home, invoke, runner


@pytest.mark.parametrize('flags,provider,model', [
    (['--model', 'one-run'], 'anthropic', 'one-run'),
    (['--provider', 'ollama', '--model', 'local-model'], 'ollama', 'local-model'),
    ([], 'anthropic', 'saved-model'),
])
def test_override_preview_and_build_match_and_do_not_persist(runner, home, monkeypatch, flags, provider, model):
    project = create_project(runner, home).with_settings(graph_enabled=True, graph_model_mode='independent',
        graph_generation_provider='anthropic', graph_generation_model='saved-model')
    project.save()
    (project.paths.root / 'a.md').write_text('Alpha supports Beta.')
    Corpus.load(project.paths.sources).register('a.md')
    before = project.paths.config.read_bytes()
    calls = []
    def factory(name, config, *, model, recorder):
        calls.append((name, model))
        return SimpleNamespace(model=model, generate=lambda *a, **k: '{"entities":[],"edges":[]}',
            graph_model_limits=lambda: {'max_tokens': 64000, 'max_input_tokens': 200000, 'resolved_model': model})
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', factory)
    args = ['graph', 'build', '--project', project.slug, *flags]
    preview = invoke(runner, home, *args, '--dry-run', '--json')
    assert preview.exit_code == 0, preview.output
    selected = json.loads(preview.stdout)['graph_model']
    assert (selected['provider'], selected['model']) == (provider, model)
    assert selected['mode'] == ('override' if flags else 'independent')
    assert not calls and not (project.paths.root / 'graph.sqlite').exists()
    human = invoke(runner, home, *args, '--dry-run')
    estimate = '$0.000000' if provider == 'ollama' else 'unknown'
    assert human.exit_code == 0 and model in human.stdout and f'Budget-based cost estimate: {estimate}' in human.stdout
    built = invoke(runner, home, *args, '--json')
    assert built.exit_code == 0, built.output
    assert calls == [(provider, model)]
    assert json.loads(built.stdout)['usage']['graph_model'] == selected
    assert project.paths.config.read_bytes() == before
    assert Project.load(project.paths.root).settings.generation_model == 'test-chat'
    status = invoke(runner, home, 'graph', 'status', '--project', project.slug, '--json')
    assert json.loads(status.stdout)['extraction_models']['sources'][0]['identity']['model'] == model


def test_provider_only_override_and_invalid_override_fail_before_client(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', lambda *a, **k: pytest.fail('No provider allowed'))
    for flags, error in [(['--provider', 'ollama'], 'no model given'), (['--model', ' '], 'non-empty text'),
                         (['--provider', 'unknown', '--model', 'x'], 'unknown')]:
        result = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--dry-run', '--json', *flags)
        assert result.exit_code == 1
        assert error in json.loads(result.stdout)['error']


def test_help_explains_override_and_inheritance_scope(runner, home):
    result = invoke(runner, home, 'graph', 'build', '--help')
    assert result.exit_code == 0
    assert '--provider' in result.stdout and '--model' in result.stdout and 'invocation only' in result.stdout
    result = invoke(runner, home, 'config', '--help')
    assert result.exit_code == 0 and 'graph_model_mode' in result.stdout and 'inherit' in result.stdout
