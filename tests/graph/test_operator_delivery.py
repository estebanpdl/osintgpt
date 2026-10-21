'''Operator delivery keeps large evidence reachable without widening model contracts.'''

import json

import pytest

from osintgpt.graph import Entity, discover_entities, graph_for, query_graph
from osintgpt.graph.export_snapshot import query_export_context, selection_package
from osintgpt.graph.inspection import inspect_assertion, inspect_entities, inspect_graph, query_messages
from osintgpt.graph.retrieval import payload_size


def large_quote():
    return 'Alpha funds Beta. ' + 'Long supporting material. ' * 3000 + ' Final evidence: financiación.'


def test_large_assertion_is_reachable_and_complete_on_demand(graph_case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Operator inspection must make no generation calls.')
    monkeypatch.setattr('osintgpt.graph.provider.LazyGraphGenerator.__init__', forbidden)
    edge = graph_case.publish('large.md', large_quote(), relation_polarity='negative', assertion='alleged')
    bounded = query_graph(graph_case.project, 'Alpha', max_bytes=65536)
    assert bounded['claims'] == [] and 'response_bytes' in bounded['truncation_reasons']
    assert payload_size(bounded) <= 65536

    query = inspect_graph(graph_case.project, 'Alpha')
    result = query.result
    assert result['inspection']['assertions_found'] == result['inspection']['assertions_returned'] == 1
    assert not result['truncated'] and 'response_bytes' not in result['limits']
    assert query_messages(result)[1] == []
    row = query.page()['rows'][0]
    assert row['assertion_id'] == edge.assertion_id
    assert (row['relation_polarity'], row['assertion']) == ('negative', 'alleged')
    assert not {'provenance', 'evidence', 'evidence_quotes', 'locations'} & row.keys()
    assert len(row['quotation_preview']) == 240
    assert payload_size(result) < 65536
    full = query.details(edge.assertion_id)
    assert full['evidence_quotes'] == [large_quote()]
    assert full['provenance'] == edge.provenance
    assert full['verification']['current'] == 'found'
    assert payload_size(full) > 65536
    selected = inspect_assertion(graph_case.project, edge.assertion_id)
    assert selected['claims'][0]['evidence_quotes'] == [large_quote()]
    assert not selected['truncated']
    assert query.matching(text='FINANCIACIÓN', verification='found') == [row]
    assert query.matching(text='absent phrase') == []
    full['locations'].clear()
    assert query.details(edge.assertion_id)['locations']  # Returned details cannot alter the retained snapshot.


def test_pages_do_not_drop_or_duplicate_large_rows(graph_case):
    edges = [graph_case.publish(f'source-{i}.md', f'Alpha funds Organization {i}. ' + 'Evidence. ' * 1500,
                               target=f'Organization {i}') for i in range(7)]
    query = inspect_graph(graph_case.project, 'Alpha')
    offset, ids = 0, []
    while offset is not None:
        page = query.page(offset=offset, page_size=2)
        assert page['total_returned'] == 7
        ids.extend(row['assertion_id'] for row in page['rows'])
        offset = page['next_offset']
    assert len(ids) == len(set(ids)) == 7
    assert set(ids) == {edge.assertion_id for edge in edges}
    assert query.page(offset=7)['rows'] == []
    assert query.page(offset=100)['next_offset'] is None
    assert payload_size(query.full_result()) > 65536
    for options in ({'offset': -1}, {'page_size': 0}, {'offset': True}):
        with pytest.raises(ValueError):
            query.page(**options)


def test_large_path_stays_complete_and_keeps_arrow_direction(graph_case):
    first = graph_case.publish('first.md', large_quote())
    second = graph_case.publish('second.md', 'Gamma owns Beta.', source='Gamma', target='Beta')
    bounded = query_graph(graph_case.project, 'Alpha', 'Gamma', max_bytes=65536)
    assert bounded['path'] == [] and not bounded['path_complete']
    query = inspect_graph(graph_case.project, 'Alpha', 'Gamma')
    assert query.result['connected'] and query.result['path_complete']
    assert query.result['path_assertion_ids'] == [first.assertion_id, second.assertion_id]
    assert query.rows[1]['source'] == 'Gamma' and query.rows[1]['target'] == 'Beta'
    assert query.details(second.assertion_id)['traversal']['against_direction']
    assert query.page(page_size=1)['next_offset'] == 1
    assert not query_messages(query.result)[1]
    directed = inspect_graph(graph_case.project, 'Alpha', 'Gamma', direction='outgoing')
    assert not directed.result['connected']


def test_existing_source_and_work_limits_still_apply(graph_case):
    graph_case.publish('one.md', 'Alpha funds Beta.')
    graph_case.publish('two.md', 'Alpha funds Gamma.', target='Gamma')
    scoped = inspect_graph(graph_case.project, 'Alpha', refs=['two.md'])
    assert [row['ref'] for row in scoped.rows] == ['two.md']
    assert not inspect_graph(graph_case.project, 'Alpha', refs=[]).rows
    limited = inspect_graph(graph_case.project, 'Alpha', limit=1)
    assert len(limited.rows) == 1 and 'result_limit' in limited.result['truncation_reasons']
    assert not limited.result['inspection']['discovery_complete']
    assert 'maximum assertion count' in query_messages(limited.result)[1][0]
    assert inspect_graph(graph_case.project, 'Missing').result['status'] == 'entity_not_found'
    for options in ({'depth': 5}, {'limit': 61}, {'max_nodes': 257}, {'max_edges': 2001}):
        with pytest.raises(ValueError):
            inspect_graph(graph_case.project, 'Alpha', **options)
    for max_bytes in (None, 65537):
        with pytest.raises(ValueError):
            query_graph(graph_case.project, 'Alpha', max_bytes=max_bytes)


def test_export_hydrates_full_saved_evidence_not_previews(graph_case):
    edge = graph_case.publish('large.md', large_quote())
    graph_case.publish('other.md', 'Alpha owns Gamma.', target='Gamma', relation='owns')
    query = inspect_graph(graph_case.project, 'Alpha')
    context = query_export_context(graph_case.project, query.result)
    # Subsequent edits cannot replace evidence in a previously captured operator query.
    (graph_case.project.paths.root / 'large.md').write_text('Different current source.', encoding='utf-8')
    selected = query.matching(text='financiación')
    package = selection_package(graph_case.project, query.full_result(), context=context,
                                claims=[query.details(row['assertion_id']) for row in selected])
    assert package['edges'][0]['evidence_quotes'] == [large_quote()]
    assert package['edges'][0]['provenance'] == edge.provenance
    assert package['metadata']['verification_basis'] == 'displayed_query'
    assert package['edges'][0]['verification']['current'] == 'found'
    assert 'quotation_preview' not in json.dumps(package)
    assert package['metadata']['scope'] == 'filtered'


def test_operator_identity_discovery_is_not_pruned_by_bytes(graph_case):
    graph_case.publish('base.md', 'Alpha funds Beta.')
    origins = [{'document_ref': f'{i}-' + 'long-source-name-' * 120, 'source_version': str(i)} for i in range(8)]
    with graph_for(graph_case.project) as graph:
        graph.add([Entity(f'org-{i}', f'Organization {i}', provenance={'identity_sources': origins})
                   for i in range(30)], [])
    operator = inspect_entities(graph_case.project, 'Organization')
    model = discover_entities(graph_case.project, 'Organization', limit=30, max_bytes=65536)
    assert len(operator['candidates']) == 30 and not operator['truncated']
    assert payload_size(operator) > 65536
    assert len(model['candidates']) < 30 and model['truncated']
