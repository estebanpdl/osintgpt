'''Operation counts and semantic parity for local graph performance changes.'''

from copy import deepcopy
from dataclasses import asdict

import pytest

from osintgpt.graph import Entity, graph_for
from osintgpt.graph.export import serialize_package
from osintgpt.graph.export_snapshot import query_export_context, selection_package
from osintgpt.graph.identity_review import reconcile_identities
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.verification_cache import VerificationChanged
from osintgpt.ingestion import Corpus
from test_operator_filter_snapshot import many
from test_progressive_exploration import finish


def test_mapping_enumeration_reused_across_batches_details_and_exports(graph_case, monkeypatch):
    edge, _ = many(graph_case, 205)
    calls = []
    original = Corpus.mappings
    def mappings(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(Corpus, 'mappings', mappings)
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query, 9)
    assert len(calls) == 1
    query.inspect_assertion(edge.assertion_id)
    query.page(offset=30)
    query.prepare_export(scope='table_page', assertion_ids=[edge.assertion_id])
    assert len(calls) == 1
    query.close()


@pytest.mark.parametrize('scope', ['table_page', 'diagram', 'all_matching'])
def test_export_selection_matches_full_preparation_without_unselected_details(graph_case, monkeypatch, scope):
    many(graph_case, 205)
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query, 37)
    full = query.full_result()
    ids = [row['assertion_id'] for row in query.rows]
    selected_ids = ids if scope == 'all_matching' else ids[30:60]
    selected = [row for row in full['claims'] if row['assertion_id'] in selected_ids]
    context = query_export_context(graph_case.project, full, request={'depth': 1})
    reference = selection_package(graph_case.project, full, context=context, claims=selected,
                                  filters={'sources': None, 'direction': 'connectivity'})
    calls = []
    original = query.details
    def details(key):
        calls.append(key)
        return original(key)
    monkeypatch.setattr(query, 'details', details)
    monkeypatch.setattr(query, 'full_result', lambda: pytest.fail('Unselected full-query preparation'))
    package = query.prepare_export(scope=scope, assertion_ids=selected_ids, request={'depth': 1})
    assert calls == selected_ids
    assert package['edges'] == reference['edges'] and package['entities'] == reference['entities']
    assert package['metadata']['query'] == reference['metadata']['query']
    assert package['metadata']['source_versions'] == reference['metadata']['source_versions']
    assert package['metadata']['extraction_models'] == reference['metadata']['extraction_models']
    assert package['metadata']['path_complete'] == reference['metadata']['path_complete']
    for suffix in ('.json', '.cypherl'):
        # Supply identical scope metadata to compare actual interchange serialization.
        reference['metadata'] = deepcopy(package['metadata'])
        assert serialize_package(package, suffix) == serialize_package(reference, suffix)
    query.close()


def test_returned_details_and_exports_cannot_mutate_cached_evidence(graph_case):
    edge, _ = many(graph_case)
    query = OperatorExploration(graph_case.project, 'alpha')
    query.advance(batch_size=3)
    first = query.rows[0]['assertion_id']
    before = query.details(first)
    detail = query.details(first)
    detail['locations'].clear()
    detail['provenance'].clear()
    detail['verification']['current'] = 'forged'
    query.pause()
    query.resume()
    finish(query, 7)
    assert query.details(first) == before
    package = query.prepare_export(scope='table_page', assertion_ids=[first])
    package['edges'][0]['locations'].clear()
    package['metadata']['query']['selected'].clear()
    package['entities'][0]['name'] = 'forged'
    assert query.details(first) == before
    assert query.result['selected']
    query.expand(2)
    finish(query)
    assert query.details(edge.assertion_id)['provenance']
    query.close()


def test_bulk_identity_lookup_matches_existing_resolution_and_uses_indexes(graph_case):
    graph_case.publish('a.md', 'Alpha links Beta.')
    graph_case.publish('b.md', 'Other Alpha links Gamma.', source='Other Alpha', target='Gamma')
    reconcile_identities(graph_case.project, 'other alpha', 'alpha', reason='Synthetic review.', apply=True)
    with graph_for(graph_case.project) as graph:
        graph.add([Entity(f'unrelated-{i}', f'Unrelated {i}') for i in range(200)], [])
        wanted = ['alpha', 'other alpha', 'beta', 'missing', 'alpha']
        expected = {}
        for key in wanted:
            candidates, mode = graph.entity_candidates(key, limit=1)
            if mode == 'id':
                expected[candidates[0].key] = asdict(candidates[0])
        statements = []
        graph.connection.set_trace_callback(statements.append)
        actual = {entity.key: asdict(entity) for entity in graph.entities_by_ids(wanted)}
        graph.connection.set_trace_callback(None)
        assert actual == expected
        selects = [sql for sql in statements if sql.lstrip().upper().startswith(('SELECT', 'WITH'))]
        assert len(selects) <= 3
        plans = [dict(row)['detail'] for sql in selects for row in graph.connection.execute('EXPLAIN QUERY PLAN '+sql)]
        assert not any('SCAN e ' in line or line == 'SCAN entities' for line in plans)
        assert any('SEARCH e USING INDEX' in line for line in plans)
        assert graph.entities_by_ids([]) == []


def test_export_does_not_use_candidate_search_for_known_endpoints(graph_case, monkeypatch):
    many(graph_case)
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    from osintgpt.graph.store import GraphStore
    monkeypatch.setattr(GraphStore, 'entity_candidates', lambda *a, **k: pytest.fail('Candidate search during export'))
    package = query.prepare_export(scope='table_page', assertion_ids=[query.rows[0]['assertion_id']])
    assert len(package['entities']) == 2
    query.close()


def test_bulk_identity_lookup_has_no_sql_parameter_or_selection_ceiling(graph_case):
    with graph_for(graph_case.project) as graph:
        entities = [Entity(f'id-{i}', f'Identity {i}') for i in range(2105)]
        graph.add(entities, [])
        assert {entity.key for entity in graph.entities_by_ids(row.key for row in entities)} == {
            entity.key for entity in entities}


def test_empty_export_never_expands_assertions_or_loads_identities(graph_case, monkeypatch):
    many(graph_case)
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    from osintgpt.graph.store import GraphStore
    monkeypatch.setattr(query, 'details', lambda *a: pytest.fail('Details for an empty selection'))
    monkeypatch.setattr(GraphStore, 'entities_by_ids', lambda *a: pytest.fail('Identities for an empty selection'))
    package = query.prepare_export(scope='table_page', assertion_ids=[])
    assert package['edges'] == package['entities'] == []
    assert package['metadata']['selection']['matching_total'] == 75
    query.close()


@pytest.mark.parametrize('action', ['merge', 'undo'])
def test_identity_review_invalidates_prepared_query(graph_case, action):
    graph_case.publish('a.md', 'Alpha links Beta.')
    graph_case.publish('b.md', 'Other Alpha links Gamma.', source='Other Alpha', target='Gamma')
    applied = None
    if action == 'undo':
        applied = reconcile_identities(graph_case.project, 'other alpha', 'alpha', reason='Synthetic review.', apply=True)
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    frozen = query.prepare_export()
    if applied:
        reconcile_identities(graph_case.project, undo=applied['id'], apply=True)
    else:
        reconcile_identities(graph_case.project, 'other alpha', 'alpha', reason='Synthetic review.', apply=True)
    with pytest.raises(VerificationChanged):
        query.prepare_export()
    assert query.state == 'invalidated' and frozen['edges']
