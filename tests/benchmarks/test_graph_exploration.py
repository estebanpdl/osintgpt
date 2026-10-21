'''Cross-layer closeout cases use an independent topology oracle and real source evidence.'''

import json

import pytest

from osintgpt.graph import query_graph
from osintgpt.graph.inspection import query_messages
from osintgpt.graph.operator_exploration import OperatorExploration
from graph_exploration import expected, finish, fixture, run


@pytest.fixture
def network(tmp_path):
    return fixture(tmp_path, identities=35, parallel=3)


@pytest.mark.parametrize('direction', ['incoming', 'outgoing', 'connectivity'])
@pytest.mark.parametrize('refs', [None, ['source-0.md', 'source-1.md'], []])
def test_scopes_counts_pages_filters_exports_agree_with_independent_oracle(network, direction, refs):
    project, edges = network
    query = OperatorExploration(project, 'root', direction=direction, refs=refs, depth=10)
    try:
        finish(query, 7)
        ids = [row['assertion_id'] for row in query.rows]
        assert len(ids) == len(set(ids)) and set(ids) == expected(edges, depth=10, direction=direction, refs=refs)
        direct = [edge for edge in edges if 'root' in (edge.source_id, edge.target_id)]
        scoped = [edge for edge in direct if refs is None or edge.ref in refs]
        assert query.result['selected']['entity']['relationships']['all_sources'] == len(direct)
        assert query.result['selected']['entity']['relationships']['selected_sources'] == len(scoped)
        for relation in ('links', 'reports', 'cycle', 'path step'):
            filters = {'relations': [relation], 'verification': 'found'}
            matched = query.matching(**filters)
            wanted = {edge.assertion_id for edge in edges if edge.assertion_id in ids and edge.relation == relation}
            assert {row['assertion_id'] for row in matched} == wanted
            package = query.prepare_export(filters=filters)
            assert {edge['assertion_id'] for edge in package['edges']} == wanted
            assert package['metadata']['selection']['matching_total'] == len(wanted)
            assert all('polarity' not in edge and 'relation_polarity' in edge for edge in package['edges'])
        assert query_messages(query.result)[1] == []
    finally:
        query.close()


def test_ten_step_path_leaf_disconnected_and_paused_path_remain_distinct(network):
    project, _ = network
    for target, depth, count in [('p10', 10, 10), ('p12', 12, 12), ('isolated', 12, 0)]:
        query = OperatorExploration(project, 'root', target, direction='outgoing', depth=depth)
        try:
            query.advance(batch_size=3); query.pause()
            assert query.result['connected'] is None and not query.result['path_complete'] and query.rows == []
            query.resume(); finish(query, 11)
            assert len(query.rows) == count and query.result['path_complete'] == bool(count)
            assert query_messages(query.result)[1] == []
        finally:
            query.close()
    leaf = OperatorExploration(project, 'p11', direction='outgoing', depth=1)
    finish(leaf, 2)
    assert len(leaf.rows) == 1 and query_messages(leaf.result)[1] == []
    leaf.close()
    bounded = query_graph(project, 'root')
    assert bounded['truncated'] and len(bounded['claims']) <= 60
    assert len(json.dumps(bounded, ensure_ascii=False).encode('utf8')) <= 65536


def test_measurement_harness_reports_data_without_performance_pass_thresholds():
    result = run(identities=35, parallel=3, batch_size=13, trace=True)
    assert result['provider_calls'] == 0 and not result['operator_projects_modified']
    assert result['fixture']['longest_quote_bytes'] > 65536
    metrics = result['measurements']
    assert metrics['discovered_assertions'] == result['fixture']['assertions'] > 60
    assert metrics['exports']['table_page']['assertions'] == 30
    assert metrics['exports']['diagram']['assertions'] == 75
    assert metrics['memory']['python_allocations']['peak_bytes'] > 0


def test_stage_measurements_report_reuse_and_selected_work(network):
    from contextlib import ExitStack
    from graph_performance import Stages, measure_stages
    project, edges = network
    with ExitStack() as stack:
        result = measure_stages(project, edges, 'root', 12, Stages(stack))
    assert result['discovery']['inclusive_stages']['corpus_mappings']['calls'] == 1
    page = result['exports']['table_page']
    assert page['preparation']['inclusive_stages']['assertion_details']['calls'] == 30
    assert page['preparation']['inclusive_stages']['identity_batch']['calls'] == 1
    assert set(page['formats']) == {'.json', '.cypherl'}
