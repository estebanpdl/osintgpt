'''Operator model controls persist shared settings without requests and invalidate stale previews.'''

import json
from threading import Event
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from osintgpt.app.session import VIEW
from osintgpt.cli import app as cli
from osintgpt.graph.jobs import GraphJobs
from osintgpt.ingestion import Corpus
from osintgpt.projects import Project, ProjectSettings, save_user_defaults
from test_app_renders import home, project, run
from test_graph_view import button, widget


def no_providers(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('Settings and previews must not construct a provider')
    monkeypatch.setattr('osintgpt.app.session.build_generation_provider', forbidden)
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)


def test_app_saves_shared_graph_pair_and_cli_resolves_it(project, home, monkeypatch):
    no_providers(monkeypatch)
    project = project.with_settings(generation_model='chat-A')
    project.save()
    app = run(project, **{VIEW: 'Settings'})
    widget(app.checkbox, 'Use Chat/RAG settings').uncheck().run()
    widget(app.selectbox, 'Graph extraction provider').select('ollama').run()
    widget(app.text_input, 'Graph extraction model').set_value('graph-B').run()
    button(app, 'Save').click().run()
    assert not app.exception
    saved = Project.load(project.paths.root).settings
    assert (saved.graph_model_mode, saved.graph_generation_provider, saved.graph_generation_model) == ('independent', 'ollama', 'graph-B')
    assert saved.generation_model == 'chat-A'
    assert any('credential: not required' in row.value for row in app.caption)
    result = CliRunner().invoke(cli, ['--home', str(home), 'graph', 'build', '--project', project.slug, '--dry-run', '--json'])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)['graph_model']['model'] == 'graph-B'
    assert not (project.paths.root / 'graph.sqlite').exists()


def test_provider_change_does_not_carry_model_and_inherit_overrides_defaults(project, home, monkeypatch):
    no_providers(monkeypatch)
    save_user_defaults(home, ProjectSettings(graph_generation_provider='anthropic', graph_generation_model='default-B'))
    project = project.with_settings(generation_model='chat-A')
    project.save()
    app = run(project, **{VIEW: 'Settings'})
    assert not widget(app.checkbox, 'Use Chat/RAG settings').value
    assert widget(app.text_input, 'Graph extraction model').value == 'default-B'
    widget(app.selectbox, 'Graph extraction provider').select('ollama').run()
    assert widget(app.text_input, 'Graph extraction model').value == '' and button(app, 'Save').disabled
    widget(app.checkbox, 'Use Chat/RAG settings').check().run()
    button(app, 'Save').click().run()
    assert not app.exception
    saved = Project.load(project.paths.root)
    assert saved.settings.graph_model_mode == 'inherit'
    fresh = run(saved, **{VIEW: 'Settings'})
    assert widget(fresh.checkbox, 'Use Chat/RAG settings').value
    assert any('openai / chat-A (inherited' in row.value for row in fresh.caption)


def test_effective_model_change_discards_cached_preview(project, monkeypatch):
    no_providers(monkeypatch)
    project = project.with_settings(generation_model='chat-A', graph_generation_provider='ollama', graph_generation_model='graph-B')
    project.save()
    app = run(project, **{VIEW: 'Graph'})
    button(app, 'Preview build workload').click().run()
    assert any('Preview:' in row.value for row in app.info)
    project.with_settings(generation_model='chat-C').save()
    app.run()
    assert any('Preview:' in row.value for row in app.info)
    project.with_settings(graph_generation_model='graph-D').save()
    app.run()
    assert not app.exception
    assert not any('Preview:' in row.value for row in app.info)
    assert any('Preview again' in row.value for row in app.info)
    assert any('graph-D' in row.value for row in app.caption)


def test_running_job_keeps_its_model_when_next_build_selection_changes(project, monkeypatch):
    project = project.with_settings(graph_enabled=True, generation_model='chat-A',
        graph_generation_provider='ollama', graph_generation_model='graph-B')
    project.save()
    (project.paths.root / 'a.md').write_text('Alpha funds Beta.')
    Corpus.load(project.paths.sources).register('a.md')
    entered, release, jobs = Event(), Event(), GraphJobs()
    calls = []
    def factory(provider, config, *, model, recorder):
        calls.append((provider, model))
        def generate(*a, **k):
            entered.set()
            assert release.wait(60)
            return '{"entities":[],"edges":[]}'
        return SimpleNamespace(model=model, generate=generate)
    monkeypatch.setattr('osintgpt.app.views.graph_controls.JOBS', jobs)
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', factory)
    app = run(project, **{VIEW: 'Graph'})
    button(app, 'Build graph').click().run()
    try:
        assert entered.wait(10)
        project.with_settings(graph_generation_model='graph-D').save()
        app.run()
        assert not app.exception
        captions = [row.value for row in app.caption]
        assert any('Next build model: ollama / graph-D' in value for value in captions)
        assert any('Running build model: ollama / graph-B' in value for value in captions)
        assert calls == [('ollama', 'graph-B')]
    finally:
        release.set()
        jobs.jobs[jobs.key(project)].thread.join(10)


@pytest.mark.parametrize('change', ['inherited_chat', 'user_default', 'endpoint'])
def test_previews_follow_inheritance_defaults_and_endpoint_changes(project, home, monkeypatch, change):
    no_providers(monkeypatch)
    choices = {'generation_provider': 'ollama', 'generation_model': 'chat-A'}
    if change == 'user_default':
        save_user_defaults(home, ProjectSettings(graph_generation_provider='ollama', graph_generation_model='default-A'))
    project = project.with_settings(**choices)
    project.save()
    app = run(project, **{VIEW: 'Graph'})
    button(app, 'Preview build workload').click().run()
    assert any('Preview:' in row.value for row in app.info)
    if change == 'inherited_chat':
        project.with_settings(generation_model='chat-B').save()
    elif change == 'user_default':
        save_user_defaults(home, ProjectSettings(graph_generation_provider='ollama', graph_generation_model='default-B'))
    else:
        monkeypatch.setenv('OLLAMA_BASE_URL', 'http://localhost:11435')
    app.run()
    assert not app.exception
    assert not any('Preview:' in row.value for row in app.info)
    assert any('Preview again' in row.value for row in app.info)
