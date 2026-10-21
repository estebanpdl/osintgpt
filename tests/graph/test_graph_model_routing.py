'''Graph model choices control extraction, repair, accounting, and compatible checkpoint reuse.'''

import json
from types import SimpleNamespace

import pytest

from osintgpt.config import Settings
from osintgpt.graph import build_graph, graph_for, preflight_graph, saved_graph_status
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.provider import LazyGraphGenerator
from osintgpt.graph.run_settings import GraphRunSettings
from osintgpt.llm.usage import UsageRecorder
from osintgpt.projects import Project, ProjectSettings, save_user_defaults
from test_build_status import EMPTY, Generator, project
from test_extraction_contract import StructuredScript, complete_proposal
from test_generation_recovery import scripted_native


def lazy(project, *outcomes, forbidden=False):
    constructed = []
    def factory(provider, config, *, model, recorder):
        assert not forbidden, 'Compatible work must not construct a provider'
        constructed.append((provider, model))
        return StructuredScript(*(outcomes or (json.loads(EMPTY),)))
    return LazyGraphGenerator(factory, project.effective_settings(), Settings(), UsageRecorder()), constructed


def independent(project):
    return project.with_settings(generation_provider='openai', generation_model='chat-A', graph_model_mode='independent',
                                 graph_generation_provider='gemini', graph_generation_model='graph-B')


def test_native_extraction_and_repair_both_use_graph_model_and_recorder(project):
    project = independent(project).with_settings(graph_generation_model='gpt-4o-mini')
    (project.paths.root / 'material' / 'a.md').write_text('Alpha funds Beta.')
    native, requests = scripted_native(complete_proposal('Invented.'),
                                      {'corrections': [{'index': 0, 'evidence': ['Alpha funds Beta.']}]})
    recorder = UsageRecorder()
    created = []
    def factory(provider, config, *, model, recorder):
        created.append((provider, model))
        native.model, native.recorder = model, recorder
        return native
    wrapper = LazyGraphGenerator(factory, project.effective_settings(), Settings(), recorder)
    try:
        report = build_graph(project, wrapper)
        assert report.status == 'complete' and report.accounting.repaired_claims == 1
        assert created == [('gemini', 'gpt-4o-mini')]
        assert [request['model'] for request in requests] == ['gpt-4o-mini', 'gpt-4o-mini']
        assert any(tool['function']['name'] == 'submit_extraction' for tool in requests[0]['tools'])
        assert requests[1]['response_format']['type'] == 'json_schema'
        assert all('max_tokens' not in request and 'max_output_tokens' not in request for request in requests)
        assert recorder.calls == 2 and all(record.model == 'gpt-4o-mini' for record in recorder.records)
        assert report.accounting.requests_attempted == report.usage['calls'] == 2
        with graph_for(project) as graph:
            recipe = json.loads(graph.state.entries()['material/a.md']['recipe_data'])
        assert recipe['provider'] == wrapper.graph_options()
    finally:
        native.client.close()


def test_pin_same_model_then_change_chat_reuses_completed_graph_keylessly(project):
    initial = project.with_settings(generation_model='model-A')
    generator, constructed = lazy(initial)
    assert build_graph(initial, generator).calls == 1
    pinned = initial.with_settings(graph_model_mode='independent', graph_generation_provider='openai',
                                   graph_generation_model='model-A', generation_provider='anthropic', generation_model='chat-C')
    generator, constructed = lazy(pinned, forbidden=True)
    preview = preflight_graph(pinned, generator)
    assert preview['counts']['documents_unchanged'] == 1
    assert preview['provider']['provider'] == 'openai' and preview['provider']['model'] == 'model-A'
    result = build_graph(pinned, generator, incremental=True)
    assert result.calls == 0 and result.skipped == 1 and not constructed


@pytest.mark.parametrize('inherit', [False, True])
def test_actual_graph_selection_change_previews_refresh_and_keeps_failed_publication(project, inherit):
    (project.paths.root / 'material' / 'a.md').write_text('Alpha funds Beta.')
    initial = project.with_settings(generation_model='chat-A') if inherit else independent(project)
    generator, _ = lazy(initial, complete_proposal('Alpha funds Beta.'))
    assert build_graph(initial, generator).edges == 1
    changed = initial.with_settings(**({'generation_model': 'chat-D'} if inherit else {'graph_generation_model': 'graph-D'}))
    generator, _ = lazy(changed, RuntimeError('Injected provider failure'))
    preview = preflight_graph(changed, generator)
    assert preview['counts']['documents_to_extract'] == 1
    assert 'provider' in preview['details'][0]['reason']
    result = build_graph(changed, generator, incremental=True)
    assert result.failed and result.status == 'complete_with_gaps'
    with graph_for(changed) as graph:
        assert graph.edge_count == 1
        assert json.loads(graph.state.entries()['material/a.md']['recipe_data'])['provider']['model'] in ('chat-A', 'graph-B')


@pytest.mark.parametrize('stop_at', ['window_saved', 'response_saved'])
@pytest.mark.parametrize('change_graph', [False, True])
def test_resume_routes_by_graph_identity_instead_of_chat(project, stop_at, change_graph):
    initial = independent(project)
    (project.paths.root / 'material' / 'a.md').write_text('Long source content. ' * 4000)
    generator, _ = lazy(initial)
    def stop(event):
        if event.kind == stop_at:
            raise GraphCancelled('Injected stop')
    with pytest.raises(GraphCancelled):
        build_graph(initial, generator, on_event=stop)
    before = saved_graph_status(initial)
    assert before['checkpoints']['windows_saved' if stop_at == 'window_saved' else 'responses_saved'] == 1
    changed = initial.with_settings(generation_model='chat-C',
                                   graph_generation_model='graph-D' if change_graph else 'graph-B')
    generator, constructed = lazy(changed)
    result = build_graph(changed, generator)
    assert result.status == 'complete'
    reused = result.accounting.windows_reused + result.accounting.responses_reused
    assert reused == (0 if change_graph else 1)
    assert result.accounting.checkpoints_discarded == int(change_graph)
    assert all(model == ('graph-D' if change_graph else 'graph-B') for _, model in constructed)


def test_app_factory_captures_graph_selection_and_does_not_change_chat_runtime(tmp_path, monkeypatch):
    from osintgpt.app.session import runtime_for
    from osintgpt.app.views.graph_controls import generator_factory
    project = Project.create('Separate roles', home=tmp_path,
                             settings=ProjectSettings(generation_model='chat-A', graph_generation_provider='anthropic',
                                                      graph_generation_model='graph-B', cost_ceiling_usd=0.5))
    created = []
    def factory(provider, config, *, model, recorder=None):
        created.append((provider, model, recorder))
        return SimpleNamespace(model=model)
    monkeypatch.setattr('osintgpt.app.session.build_generation_provider', factory)
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', factory)
    runtime = runtime_for(project, tmp_path)
    graph_factory = generator_factory(runtime, tmp_path, GraphRunSettings(cost_ceiling_usd=0.25))
    project.with_settings(graph_generation_model='later-model').save()
    save_user_defaults(tmp_path, ProjectSettings(graph_generation_provider='ollama', graph_generation_model='later-default'))
    runtime.generator
    generator = graph_factory()
    generator.provider
    assert [(provider, model) for provider, model, _ in created] == [('openai', 'chat-A'), ('anthropic', 'graph-B')]
    assert generator.recorder.cost_ceiling_usd == 0.25
    assert project.settings.generation_model == 'chat-A'


def test_preflight_does_not_claim_a_cost_bound_for_provider_default_output(project, monkeypatch):
    initial = independent(project)
    generator, constructed = lazy(initial, forbidden=True)
    priced = []
    monkeypatch.setattr('osintgpt.llm.usage.estimate_cost', lambda model, *a, **k: priced.append(model) or 0.01)
    report = preflight_graph(initial, generator)
    assert report['budget_cost_estimate_usd'] is None
    assert report['output_policy']['mode'] == 'provider_default'
    assert not priced and not constructed
