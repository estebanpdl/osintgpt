'''Rich exports preserve displayed evidence, selection limits, status, and import identity.'''

import json
from copy import deepcopy

import pytest

from osintgpt.graph import Edge, Entity, graph_for, query_graph, to_json, to_cypherl
from osintgpt.graph.export import serialize_package
from osintgpt.graph.export_snapshot import full_graph_package, query_export_context, selection_package
from osintgpt.ingestion.snapshots import load_snapshot


@pytest.fixture
def published(graph_case):
    case = graph_case
    first = case.publish('a.md', 'X afirma: Alpha no financió Beta en 2024.',
                         quotes=['Alpha no financió Beta', 'en 2024'], relation_polarity='negative',
                         assertion='alleged', attributed_to='X', event_time='2024')
    second = case.publish('b.md', 'Beta owns Gamma.', source='Beta', target='Gamma', relation='owns')
    with graph_for(case.project) as graph:
        for edge in (first, second):
            snapshot = load_snapshot(case.project.paths.root / edge.ref, ref=edge.ref, cache_dir=case.project.paths.extracts)
            graph.state.completed(edge.ref, snapshot.fingerprint, snapshot.version)
        graph.connection.execute("UPDATE graph_build SET started_at='2026-10-07T01:00:00Z', finished_at='2026-10-07T02:00:00Z'")
        graph.connection.commit()
    return case, first, second


def test_full_export_matches_retrieval_and_is_deterministic(published):
    case, first, _ = published
    path = case.project.paths.root / 'graph.sqlite'
    original = path.read_bytes()
    package = full_graph_package(case.project)
    row = next(row for row in package['edges'] if row['assertion_id'] == first.assertion_id)
    retrieved = query_graph(case.project, 'Alpha', depth=1)['claims'][0]
    assert row == {key: value for key, value in retrieved.items() if key != 'traversal'}
    assert package['metadata']['graph']['build_finished_at'] == '2026-10-07T02:00:00Z'
    assert package['metadata']['graph']['stale'] is False
    assert len(package['metadata']['source_versions']) == 2
    assert serialize_package(package) == serialize_package(full_graph_package(case.project))
    assert path.read_bytes() == original
    with graph_for(case.project) as graph:
        assert json.loads(to_json(graph, project=case.project)) == json.loads(json.dumps(package))
        cypher = to_cypherl(graph, project=case.project)
    assert 'OsintGPTExport' in cypher and 'project_id: ' in cypher
    edge = next(line for line in cypher.splitlines() if first.assertion_id in line and line.startswith('MATCH'))
    merge, update = edge.split(' SET ', 1)
    assert 'verification' not in merge and 'r.verification = ' in update
    assert 'r.source_version = ' in update and 'negative' in merge and '2024' in merge


def test_stale_incomplete_and_missing_sources_are_exportable(published):
    case, _, _ = published
    (case.project.paths.root / 'a.md').write_text('This source no longer contains the assertion.')
    (case.project.paths.root / 'b.md').unlink()
    with graph_for(case.project) as graph:
        graph.connection.execute("UPDATE graph_build SET status='complete_with_gaps'")
        graph.connection.commit()
    package = full_graph_package(case.project.with_settings(graph_enabled=False))
    state = package['metadata']['graph']
    assert state['stale'] and state['incomplete'] and not state['enabled']
    assert {row['verification']['current'] for row in package['edges']} == {'not_found', 'unreadable'}
    assert all(row['verification']['snapshot'] == 'found' for row in package['edges'])


def test_selected_path_is_frozen_and_filtered_path_is_not_complete(published):
    case, first, second = published
    result = query_graph(case.project, 'Alpha', 'Gamma', direction='outgoing')
    context = query_export_context(case.project, result, request={'refs': ['a.md', 'b.md'], 'direction': 'outgoing'})
    package = selection_package(case.project, result, context=context)
    assert package['edges'] == result['path']
    assert package['metadata']['scope'] == 'path' and package['metadata']['path_complete']
    filtered = selection_package(case.project, result, context=context, claims=[result['path'][1]], filters={'relations': ['owns']})
    assert [row['assertion_id'] for row in filtered['edges']] == [second.assertion_id]
    assert {row['key'] for row in filtered['entities']} == {'beta', 'gamma'}
    assert filtered['metadata']['scope'] == 'filtered' and not filtered['metadata']['path_complete']
    assert filtered['metadata']['query']['path_assertion_ids'] == [first.assertion_id, second.assertion_id]
    assert filtered['metadata']['query']['request']['refs'] == ['a.md', 'b.md']
    before = serialize_package(package)
    result['graph']['stale'] = True
    (case.project.paths.root / 'graph.sqlite').unlink()
    assert serialize_package(package) == before
    assert selection_package(case.project, result, context=context)['edges'] == package['edges']


def test_selection_rejects_modified_claims_and_wrong_project(published):
    case, _, _ = published
    result = query_graph(case.project, 'Alpha')
    context = query_export_context(case.project, result)
    forged = deepcopy(result['claims'])
    forged[0]['relation_polarity'] = 'positive'
    with pytest.raises(ValueError, match='unchanged'):
        selection_package(case.project, result, context=context, claims=forged)
    context['project']['id'] = 'someone-else'
    with pytest.raises(ValueError, match='different project'):
        selection_package(case.project, result, context=context)


def test_source_filtered_export_has_no_unselected_records(published):
    case, _, _ = published
    with graph_for(case.project) as graph:
        package = json.loads(to_json(graph, refs=['a.md'], project=case.project))
    assert {row['ref'] for row in package['edges']} == {'a.md'}
    assert {row['name'] for row in package['entities']} == {'Alpha', 'Beta'}
    assert {row['ref'] for row in package['metadata']['source_versions']} == {'a.md'}


def test_omitted_path_keeps_ids_and_limits_without_fabricating_edges(graph_case):
    case = graph_case
    first = case.publish('a.md', 'Alpha ' + '文' * 5000 + ' Beta.')
    second = case.publish('b.md', 'Beta owns Gamma.', source='Beta', target='Gamma', relation='owns')
    result = query_graph(case.project, 'Alpha', 'Gamma', max_bytes=4096)
    package = selection_package(case.project, result, context=query_export_context(case.project, result))
    assert package['edges'] == [] and package['entities'] == []
    assert package['metadata']['truncated'] and not package['metadata']['path_complete']
    assert package['metadata']['query']['path_assertion_ids'] == [first.assertion_id, second.assertion_id]
    assert package['metadata']['query']['response_bytes_limit'] == 4096


def test_import_retains_identity_details_and_unstored_endpoints(graph_case):
    case = graph_case
    with graph_for(case.project) as graph:
        graph.add([Entity('alpha', 'Alpha', aliases=('Άλφα',), description='Quoted "name"')],
                  [Edge('Alpha', 'Beta', 'owns', 'missing.md', 'Alpha owns Beta.', event_time='2024')])
    package = full_graph_package(case.project)
    assert {row['key'] for row in package['entities']} == {'alpha', 'beta'}
    cypher = serialize_package(package, '.cypherl')
    entity = next(line for line in cypher.splitlines() if 'key: "alpha"' in line and line.startswith('MERGE'))
    details = json.loads(json.loads(entity.split('n.details = ', 1)[1].rstrip(';')))
    assert details['aliases'] == ['Άλφα'] and details['description'] == 'Quoted "name"'
    assert 'event_time: "2024"' in cypher
