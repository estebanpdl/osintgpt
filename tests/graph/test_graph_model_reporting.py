'''Preflight and export model metadata describe actual saved work and source publications.'''

import json

import pytest

from osintgpt.graph import ExtractionOptions, build_graph, graph_for, preflight_graph, query_graph, saved_graph_status
from osintgpt.graph.export_snapshot import full_graph_package, query_export_context, selection_package
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.records import Entity
from test_build_status import Generator, project
from test_extraction_contract import complete_proposal


def model(name, *outcomes):
    generator = Generator(*outcomes)
    generator.graph_options = lambda: {'provider': 'fixture', 'model': name, 'endpoint': 'endpoint-hash'}
    return generator


def test_preflight_reports_both_source_change_and_model_change(project):
    build_graph(project, model('A'))
    path = project.paths.root / 'material' / 'a.md'
    path.write_text('Changed content.')
    before = (project.paths.root / 'graph.sqlite').read_bytes()
    preview = preflight_graph(project, model('B'))
    change = preview['details'][0]['changes']
    assert change['source_changed'] and change['model_changed']
    assert change['previous_model']['model'] == 'A' and change['next_model']['model'] == 'B'
    assert preview['counts']['documents_model_changed'] == 1
    assert 'Source text' in preview['details'][0]['reason'] and 'A -> fixture / B' in preview['details'][0]['reason']
    assert (project.paths.root / 'graph.sqlite').read_bytes() == before


@pytest.mark.parametrize('change_model', [False, True])
def test_preflight_counts_saved_work_that_will_be_lost(project, change_model):
    (project.paths.root / 'material' / 'a.md').write_text('Long content. ' * 9000)
    def stop(event):
        if event.kind == 'window_saved':
            raise GraphCancelled('stop after first saved window')
    with pytest.raises(GraphCancelled):
        build_graph(project, model('A'), on_event=stop)
    preview = preflight_graph(project, model('B' if change_model else 'A'))
    counts, checkpoint = preview['counts'], preview['details'][0]['checkpoint']
    assert counts['checkpoints_incompatible'] == int(change_model)
    assert counts['windows_incompatible'] == int(change_model)
    assert checkpoint['compatible'] is not change_model
    assert checkpoint['previous_model']['model'] == 'A'
    assert checkpoint['model_changed'] is change_model


def test_actual_publication_models_survive_partial_refresh_and_settings_changes(project):
    for name in ('a.md', 'b.md'):
        (project.paths.root / 'material' / name).write_text('Alpha funds Beta.')
    proposal = json.dumps(complete_proposal('Alpha funds Beta.'))
    assert build_graph(project, model('A', proposal)).status == 'complete'
    (project.paths.root / 'material' / 'b.md').unlink()
    assert build_graph(project, model('B', proposal)).status == 'complete_with_gaps'
    configured = project.with_settings(graph_generation_provider='ollama', graph_generation_model='not-built')
    status = saved_graph_status(configured)
    models = status['extraction_models']
    assert models['mixed'] and models['unknown_sources'] == 0
    assert {row['ref']: row['identity']['model'] for row in models['sources']} == {'material/a.md': 'B', 'material/b.md': 'A'}
    assert status['run_model']['model'] == 'B'
    assert status['previous_runs'][0]['usage']['graph_model']['model'] == 'A'
    assert full_graph_package(configured)['metadata']['extraction_models'] == models
    # Use the actual published claims as the displayed selection. Repeated names
    # in these independent extraction fixtures deliberately have different IDs.
    published = full_graph_package(project)
    result = {'claims': published['edges'], 'graph': published['metadata']['graph']}
    context = query_export_context(project, result)
    with graph_for(project) as graph:
        graph.connection.execute("UPDATE graph_documents SET recipe_data='{}'")
        graph.connection.commit()
    package = selection_package(project, result, context=context)
    assert package['metadata']['extraction_models']['mixed']
    assert {row['identity']['model'] for row in package['metadata']['extraction_models']['sources']} == {'A', 'B'}
    selected = [row for row in result['claims'] if row['source_ref'] == 'material/a.md']
    limited = selection_package(project, result, context=context, claims=selected)
    assert not limited['metadata']['extraction_models']['mixed']
    assert len(limited['metadata']['extraction_models']['sources']) == 1


def test_unknown_historical_models_are_not_replaced_by_current_settings(project):
    build_graph(project, model('A'))
    with graph_for(project) as graph:
        graph.connection.execute("UPDATE graph_documents SET recipe_data='{}'")
        graph.add([Entity('orphan', 'Legacy entity')], [])
        graph.connection.commit()
    status = saved_graph_status(project.with_settings(generation_model='latest-chat'))
    assert status['extraction_models']['unknown_sources'] == 1
    assert status['extraction_models']['unattributed_entities'] == 1
    assert status['extraction_models']['sources'][0]['identity'] == {}
