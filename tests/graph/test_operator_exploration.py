'''Operator counts and scope messages remain distinct from bounded model retrieval.'''

from copy import deepcopy

from osintgpt.graph import discover_entities, graph_for, query_graph
from osintgpt.graph.export_snapshot import query_export_context, selection_package
from osintgpt.graph.inspection import operator_query_result


def test_direct_counts_distinguish_assertions_neighbors_and_source_scope(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha supports Beta.')
    case.publish('two.md', 'Alpha denies supporting Beta.', relation_polarity='negative')
    case.publish('three.md', 'Gamma supports Alpha.', source='Gamma', target='Alpha')
    case.publish('self.md', 'Alpha describes Alpha.', target='Alpha')
    case.publish('indirect.md', 'Beta supports Delta.', source='Beta', target='Delta')
    row = discover_entities(case.project, 'Alpha', refs=['one.md', 'two.md'])['candidates'][0]
    assert row['relationships'] == {'all_sources': 4, 'selected_sources': 2,
                                    'distinct_neighbors_all_sources': 2, 'distinct_neighbors_selected_sources': 1}
    empty = discover_entities(case.project, 'Alpha', refs=[])['candidates'][0]['relationships']
    assert empty['selected_sources'] == empty['distinct_neighbors_selected_sources'] == 0


def test_neighbor_counts_follow_reviewed_identity_mappings(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha supports Beta.', target_id='beta_one')
    case.publish('two.md', 'Alpha supports Beta.', target_id='beta_two')
    with graph_for(case.project) as graph:
        graph.connection.execute('INSERT INTO entity_links VALUES (?, ?, ?)', ('beta_two', 'beta_one', 'review'))
        graph.connection.commit()
        assert graph.relationship_counts('alpha')['distinct_neighbors_all_sources'] == 1
        assert graph.relationship_counts('beta_two')['all_sources'] == 2


def test_depth_boundary_is_operator_scope_not_a_truncation_warning(graph_case):
    edge = graph_case.publish('one.md', 'Alpha supports Beta.')
    original = query_graph(graph_case.project, 'Alpha', depth=1)
    saved = deepcopy(original)
    assert original['truncation_reasons'] == ['depth_boundary']
    result = operator_query_result(original)
    assert original == saved
    assert not result['truncated'] and result['truncation_reasons'] == []
    assert result['inspection']['discovery_complete']
    assert result['inspection']['assertions_found'] == 1
    assert result['claims'][0]['assertion_id'] == edge.assertion_id
    context = query_export_context(graph_case.project, result)
    assert not selection_package(graph_case.project, result, context=context)['metadata']['truncated']


def test_neighborhood_counts_separate_direct_and_indirect_assertions(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha supports Beta.')
    case.publish('two.md', 'Alpha denies supporting Beta.', relation_polarity='negative')
    case.publish('three.md', 'Beta supports Gamma.', source='Beta', target='Gamma')
    result = operator_query_result(query_graph(case.project, 'Alpha', depth=2, max_bytes=65536))
    summary = result['inspection']
    assert summary['assertions_found'] == summary['assertions_returned'] == 3
    assert summary['direct_assertions_returned'] == 2
    assert result['selected']['entity']['relationships']['all_sources'] == 2


def test_work_limit_keeps_unknown_total_and_real_warning(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha supports Beta.')
    case.publish('two.md', 'Alpha supports Gamma.', target='Gamma')
    result = operator_query_result(query_graph(case.project, 'Alpha', limit=1))
    assert result['truncated'] and result['truncation_reasons'] == ['result_limit']
    assert not result['inspection']['discovery_complete']
    assert result['inspection']['assertions_found'] == result['inspection']['assertions_returned'] == 1


def test_response_limit_does_not_erase_the_found_count(graph_case):
    graph_case.publish('one.md', 'Alpha ' + '文' * 3000 + ' Beta.')
    result = operator_query_result(query_graph(graph_case.project, 'Alpha', max_bytes=4096))
    assert result['truncated'] and result['truncation_reasons'] == ['response_bytes']
    assert result['inspection']['discovery_complete']
    assert result['inspection']['assertions_found'] == 1
    assert result['inspection']['assertions_returned'] == 0


def test_omitted_path_and_errors_cannot_be_presented_as_complete_results(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha ' + '文' * 3000 + ' Beta.')
    case.publish('two.md', 'Beta supports Gamma.', source='Beta', target='Gamma')
    result = operator_query_result(query_graph(case.project, 'Alpha', 'Gamma', max_bytes=4096))
    assert result['path_found'] and not result['path_complete'] and result['connected'] is None
    assert result['inspection']['assertions_found'] == 2
    for status in ('unavailable', 'response_too_large', 'unbuilt'):
        failed = operator_query_result({'status': status, 'claims': []})
        assert failed['inspection']['assertions_found'] is None
        assert not failed['inspection']['discovery_complete']
