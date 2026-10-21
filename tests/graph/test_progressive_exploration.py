'''Operator traversal crosses former total-work ceilings while model retrieval stays bounded.'''

from threading import Thread

import pytest

from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.graph.exploration import ExplorationCursor
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.retrieval import query_graph
from osintgpt.graph.store import GraphStore


def populate(project, count, *, chain=False):
    entities = [Entity(str(i), f'Identity {i}') for i in range(count + 1)]
    edges = [Edge(f'Identity {i if chain else 0}', f'Identity {i + 1}', 'links', 'synthetic.md', 'source text',
                  source_id=str(i if chain else 0), target_id=str(i + 1)) for i in range(count)]
    with graph_for(project) as graph:
        graph.add(entities, edges)
    return edges


def finish(query, batch_size=50):
    while query.state == 'running':
        query.advance(batch_size=batch_size)
    assert query.state == 'complete', query.result


def test_more_than_2000_edges_256_nodes_and_60_results_pause_resume_no_loss(graph_case):
    edges = populate(graph_case.project, 2105)
    with graph_for(graph_case.project) as graph:
        graph.add([], [Edge('Identity 0', 'Identity 0', 'self', 'synthetic.md', 'source text', source_id='0', target_id='0')])
    query = OperatorExploration(graph_case.project, '0')
    query.advance(batch_size=37)
    before = [row['assertion_id'] for row in query.rows]
    query.pause(); query.advance()
    assert query.state == 'paused' and len(query.rows) == len(before)
    assert query.result['inspection']['discovery_complete'] is False
    assert query.full_result()['truncated']
    query.resume(); finish(query, 137)
    ids = [row['assertion_id'] for row in query.rows]
    assert len(ids) == len(set(ids)) == 2106
    assert ids[:len(before)] == before
    assert set(edge.assertion_id for edge in edges) <= set(ids)
    assert query.result['nodes_visited'] == 2106
    assert query.result['inspection']['discovery_complete'] and not query.result['truncated']
    pages = [row['assertion_id'] for offset in range(0, len(ids), 31) for row in query.page(offset=offset, page_size=31)['rows']]
    assert pages == ids
    bounded = query_graph(graph_case.project, '0', limit=60)
    assert bounded['truncated'] and len(bounded['claims']) <= 60
    with pytest.raises(ValueError):
        query_graph(graph_case.project, '0', depth=10)
    query.close()


def test_depth_expansion_keeps_frontier_and_complete_long_path(graph_case):
    populate(graph_case.project, 12, chain=True)
    query = OperatorExploration(graph_case.project, '0', direction='outgoing')
    finish(query)
    assert len(query.rows) == 1 and not query.cursor.exhausted
    query.expand(10); finish(query, 3)
    assert len(query.rows) == 10
    assert max(row['traversal']['depth'] for row in query.rows) == 10
    query.expand(10000); finish(query)
    assert len(query.rows) == 12 and query.cursor.exhausted
    query.close()
    path = OperatorExploration(graph_case.project, '0', '12', depth=12, direction='outgoing')
    path.advance(batch_size=3); path.pause()
    assert path.rows == [] and path.result['connected'] is None and not path.result['path_complete']
    path.resume(); finish(path, 3)
    assert len(path.rows) == 12 and path.result['path_complete']
    assert [row['traversal']['from_id'] for row in path.rows] == list(map(str, range(12)))
    assert len(path.page(page_size=2)['rows']) == 2
    assert len(path.full_result()['path']) == 12
    path.close()


def test_unfinished_no_path_is_not_absence_and_can_expand(graph_case):
    populate(graph_case.project, 8, chain=True)
    query = OperatorExploration(graph_case.project, '0', '8', depth=2, direction='outgoing')
    query.advance(batch_size=1)
    assert query.result['status'] == 'exploring' and query.result['connected'] is None
    finish(query)
    assert query.result['status'] == 'no_connection' and query.result['connected'] is False
    query.expand(8); finish(query)
    assert query.result['connected'] and len(query.rows) == 8
    query.close()


def test_changed_graph_invalidates_even_with_same_counts(graph_case):
    edges = populate(graph_case.project, 4)
    query = OperatorExploration(graph_case.project, '0')
    query.advance(batch_size=2); query.pause()
    with graph_for(graph_case.project) as graph:
        graph.connection.execute('UPDATE edges SET relation = ? WHERE assertion_id = ?', ('revised', edges[0].assertion_id))
        graph.connection.commit()
    query.resume()
    assert query.state == 'invalidated' and not query.check_current()
    assert query.result['truncated']


def test_cursor_directions_sources_cycles_and_mid_batch_cancellation():
    with GraphStore(':memory:') as graph:
        graph.add([Entity(x, x) for x in 'abcd'], [
            Edge('a', 'b', 'r', 'one', 'q', source_id='a', target_id='b'),
            Edge('b', 'a', 'r', 'two', 'q', source_id='b', target_id='a'),
            Edge('b', 'c', 'r', 'one', 'q', source_id='b', target_id='c'),
            Edge('c', 'a', 'r', 'one', 'q', source_id='c', target_id='a'),
            Edge('d', 'a', 'r', 'two', 'q', source_id='d', target_id='a')])
        for direction, refs, expected in [('incoming', ['two'], 2), ('outgoing', ['one'], 3), ('connectivity', [], 0)]:
            cursor = ExplorationCursor('a', direction=direction, refs=refs, depth=10)
            rows = []
            while not cursor.complete:
                rows += cursor.advance(graph, batch_size=2)
            assert len(rows) == expected and len({row.edge.assertion_id for row in rows}) == expected
        cursor = ExplorationCursor('a', depth=10)
        calls = []
        def cancel():
            calls.append(1)
            return len(calls) > 2
        rows = cursor.advance(graph, cancelled=cancel)
        assert not cursor.complete
        while not cursor.complete:
            rows += cursor.advance(graph, batch_size=2)
        assert len(rows) == len({row.edge.assertion_id for row in rows}) == 5


def test_rerun_threads_can_resume_and_read_transactions_do_not_block_writes(graph_case):
    populate(graph_case.project, 4)
    query = OperatorExploration(graph_case.project, '0')
    thread = Thread(target=query.advance, kwargs={'batch_size': 2}); thread.start(); thread.join()
    assert query.batches == 1 and not query._graph.connection.in_transaction
    finish(query); query.close()


def test_complete_path_can_exceed_former_60_assertion_limit(graph_case):
    populate(graph_case.project, 65, chain=True)
    query = OperatorExploration(graph_case.project, '0', '65', depth=65, direction='outgoing')
    finish(query, 17)
    assert query.result['path_complete'] and len(query.full_result()['path']) == 65
    assert len(query.page(offset=60, page_size=30)['rows']) == 5
    query.close()
    same = OperatorExploration(graph_case.project, '0', '0')
    assert same.result['path_complete'] and same.rows == []
    same.close()


def test_identity_review_invalidates_continuation_and_new_walk_uses_canonical_members(graph_case):
    from osintgpt.graph.identity_review import reconcile_identities
    first = graph_case.publish('one.md', 'Alpha funds Beta.')
    second = graph_case.publish('two.md', 'Alias funds Gamma.', source='Alias', target='Gamma')
    query = OperatorExploration(graph_case.project, 'alpha')
    query.advance(batch_size=1); query.pause()
    reconcile_identities(graph_case.project, 'alias', 'alpha', reason='Synthetic operator review.', apply=True)
    assert not query.check_current() and query.state == 'invalidated'
    fresh = OperatorExploration(graph_case.project, 'alias')
    finish(fresh, 1)
    assert fresh.result['selected']['entity']['id'] == 'alpha'
    assert {row['assertion_id'] for row in fresh.rows} == {first.assertion_id, second.assertion_id}
    fresh.close()


@pytest.mark.parametrize('depth', [0, -1, True, 1.5])
def test_invalid_depth(graph_case, depth):
    with pytest.raises(ValueError):
        OperatorExploration(graph_case.project, '0', depth=depth)
