'''Remaining model-separation cases reuse the existing recovery and legacy fixtures.'''

import pytest

from osintgpt.graph import build_graph, preflight_graph, saved_graph_status, verify_evidence
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.export_snapshot import full_graph_package
from osintgpt.graph.migration import migrate_graph
from test_build_status import project
from test_graph_model_routing import lazy
from test_graph_migration import legacy, populate


@pytest.mark.parametrize('saved_event,counter', [('window_saved', 'windows_reused'), ('response_saved', 'responses_reused')])
def test_pinning_inherited_identity_preserves_unfinished_work(project, saved_event, counter):
    initial = project.with_settings(generation_model='same-model')
    (project.paths.root / 'material' / 'a.md').write_text('Long content. ' * 9000)
    generator, _ = lazy(initial)
    def stop(event):
        if event.kind == saved_event:
            raise GraphCancelled('Checkpoint saved')
    with pytest.raises(GraphCancelled):
        build_graph(initial, generator, on_event=stop)
    pinned = initial.with_settings(generation_model='different-chat', graph_model_mode='independent',
                                   graph_generation_provider='openai', graph_generation_model='same-model')
    generator, _ = lazy(pinned)
    preview = preflight_graph(pinned, generator)
    assert preview['details'][0]['checkpoint']['compatible']
    report = build_graph(pinned, generator)
    assert report.status == 'complete' and getattr(report.accounting, counter) == 1
    assert report.accounting.checkpoints_discarded == 0


def test_invalid_independent_model_does_not_block_disabled_legacy_inspection(legacy, monkeypatch):
    project, path = legacy
    populate(path)
    project = project.with_settings(graph_enabled=False, graph_model_mode='independent',
                                   graph_generation_provider='unavailable-provider', graph_generation_model='')
    monkeypatch.setattr('osintgpt.graph.provider.resolve_graph_model', lambda *a, **k: pytest.fail('Read operations must not resolve a provider'))
    assert migrate_graph(project)['provider_calls'] == 0
    generator, created = lazy(project, forbidden=True)
    assert build_graph(project, generator).refused and not created
    assert saved_graph_status(project)['extraction_models']['unknown_sources'] == 1
    assert full_graph_package(project)['metadata']['extraction_models']['unknown_sources'] == 1
    assert verify_evidence(project).total == 1
