'''Identity selection, assertion semantics, and bounded indexed retrieval.'''

from dataclasses import replace
import sqlite3

import pytest

from osintgpt.graph import Entity, GraphStore, discover_entities, graph_for, query_graph
from osintgpt.graph.retrieval import payload_size, retrieval_status


def test_aliases_and_namesakes_require_identity_selection(graph_case):
    case = graph_case
    first = case.publish('one.md', 'Álex funds Beta.', source='Álex', source_id='person_one')
    case.publish('two.md', 'Álex funds Gamma.', source='Álex', target='Gamma', source_id='person_two')
    with graph_for(case.project) as graph:
        graph.add([Entity('person_one', 'Álex', aliases=('أليكس', '亚历克斯'), description='Researcher')], [])
    ambiguous = query_graph(case.project, 'ÁLEX')
    assert ambiguous['status'] == 'ambiguous' and 'claims' not in ambiguous
    assert {row['id'] for row in ambiguous['candidates']} == {'person_one', 'person_two'}
    assert discover_entities(case.project, 'أليكس')['candidates'][0]['id'] == 'person_one'
    assert discover_entities(case.project, '亚历克斯')['match'] == 'exact'
    assert query_graph(case.project, 'person_one')['claims'][0]['assertion_id'] == first.assertion_id
    assert discover_entities(case.project, "' OR 1=1 --")['status'] == 'entity_not_found'


def test_identity_candidate_cap_cannot_make_ambiguity_disappear(graph_case):
    graph_case.publish('a.md', 'Alpha funds Beta.')
    with graph_for(graph_case.project) as graph:
        graph.add([Entity(f'person_{i}', 'Same') for i in range(25)], [])
    result = discover_entities(graph_case.project, 'Same', limit=1)
    assert result['status'] == 'ambiguous'
    assert len(result['candidates']) == 1 and result['truncated']
    result = query_graph(graph_case.project, 'Same')
    assert len(result['candidates']) == 20 and result['truncation_reasons'] == ['candidate_limit']


def test_direction_and_qualifiers_survive_reverse_traversal(graph_case):
    case = graph_case
    first = case.publish('a.md', 'According to Witness, Alpha allegedly funds Beta in 2024.', assertion='alleged',
                         relation_polarity='positive', attributed_to='Witness', event_time='2024')
    case.publish('b.md', 'Gamma does not fund Beta.', source='Gamma', relation_polarity='negative')
    assert query_graph(case.project, 'Beta', 'Alpha', direction='outgoing')['status'] == 'no_connection'
    path = query_graph(case.project, 'Beta', 'Alpha', direction='incoming')['path']
    assert len(path) == 1 and path[0]['assertion_id'] == first.assertion_id
    assert path[0]['source'] == 'Alpha' and path[0]['target'] == 'Beta'
    assert path[0]['assertion'] == 'alleged' and path[0]['event_time'] == '2024'
    assert path[0]['traversal'] == {'from_id': 'beta', 'to_id': 'alpha', 'against_direction': True, 'depth': 1}
    rows = query_graph(case.project, 'Beta', direction='incoming')['claims']
    assert {row['relation_polarity'] for row in rows} == {'positive', 'negative'}
    assert query_graph(case.project, 'Beta', direction='outgoing')['status'] == 'no_claims'
    same = query_graph(case.project, 'Beta', 'Beta')
    assert same['connected'] and same['path_complete'] and same['path'] == []


def test_all_support_and_conflicting_assertions_are_preserved(graph_case):
    case = graph_case
    quotes = ['Alpha funds Beta.', 'The agreement began in 2024.']
    first = case.publish('a.md', '# Funding\n\n' + '\n\n'.join(quotes), quotes=quotes, event_time='2024')
    second = case.publish('b.md', 'Alpha does not fund Beta.', relation_polarity='negative')
    result = query_graph(case.project, 'Alpha')
    assert {row['assertion_id'] for row in result['claims']} == {first.assertion_id, second.assertion_id}
    row = result['claims'][0]
    assert row['evidence_quotes'] == quotes
    assert len(row['locations']) == 2
    assert row['locations'][1]['matches'][0]['spans'][0]['section'] == 'Funding'
    assert row['verification']['snapshot'] == row['verification']['current'] == 'found'
    assert row['source_version'] == first.provenance['source_version']
    assert first.assertion_id in row['citation'] and first.ref in row['citation']


def test_assertion_ids_distinguish_versions_qualifications_and_occurrences(graph_case):
    edge = graph_case.publish('a.md', 'Alpha funds Beta.')
    assert replace(edge).assertion_id == edge.assertion_id
    changed = [replace(edge, relation_polarity='negative'), replace(edge, assertion='alleged'),
               replace(edge, evidence_quotes=('Another quote.',)),
               replace(edge, provenance={**edge.provenance, 'source_version': 'other'}),
               replace(edge, provenance={**edge.provenance, 'supporting': [{'matches': [{'start': 42}]}]})]
    assert len({edge.assertion_id, *(item.assertion_id for item in changed)}) == 6
    with graph_for(graph_case.project) as graph:
        graph.add([Entity('alpha', 'Renamed Alpha')], [])
        assert graph.assertion(edge.assertion_id).assertion_id == edge.assertion_id


def test_traversal_uses_indexed_adjacency_and_enforces_work_limits(graph_case, monkeypatch):
    case = graph_case
    for index in range(6):
        case.publish(f'{index}.md', f'N{index} funds N{index + 1}.', source=f'N{index}', target=f'N{index + 1}')

    def forbidden(*args, **kwargs):
        raise AssertionError('A query must not load the full graph')

    monkeypatch.setattr(GraphStore, 'edges', forbidden)
    monkeypatch.setattr(GraphStore, 'entities', forbidden)
    result = query_graph(case.project, 'N0', 'N3', direction='outgoing')
    assert result['connected'] and len(result['path']) == 3
    for limits, reason in [({'depth': 1}, 'depth_boundary'), ({'max_edges': 1}, 'edge_limit'),
                           ({'max_nodes': 1}, 'node_limit'), ({'limit': 1}, 'result_limit')]:
        result = query_graph(case.project, 'N0', 'N3', direction='outgoing', **limits)
        assert not result['connected'] and result['truncated']
        assert reason in result['truncation_reasons'] and 'not the same' in result['note']
    result = query_graph(case.project, 'N0', direction='outgoing', depth=4, limit=1)
    assert len(result['claims']) == 1 and 'result_limit' in result['truncation_reasons']
    assert query_graph(case.project, 'N0', refs=['5.md'])['claims'] == []


def test_response_budget_never_turns_a_partial_path_into_a_complete_one(graph_case):
    case = graph_case
    case.publish('a.md', 'Alpha ' + '文' * 5000 + ' Beta.')
    case.publish('b.md', 'Beta funds Gamma.', source='Beta', target='Gamma')
    result = query_graph(case.project, 'Alpha', 'Gamma', max_bytes=4096)
    assert payload_size(result) <= 4096
    assert result['truncated'] and result['path'] == []
    assert result['path_found'] and result['connected'] is None and not result['path_complete']
    assert len(result['path_assertion_ids']) == 2


@pytest.mark.parametrize('arguments', [{'limit': True}, {'depth': 5}, {'max_nodes': 257},
                                     {'max_edges': 2001}, {'direction': 'sideways'},
                                     {'max_bytes': 100}, {'refs': 'a.md'}, {'entity': 3}])
def test_bad_arguments_are_rejected_before_storage(graph_case, arguments):
    with pytest.raises(ValueError):
        query_graph(graph_case.project, **{'entity': 'Alpha', **arguments})


def test_disabled_unbuilt_and_unavailable_states_do_not_create_a_graph(graph_case):
    case = graph_case
    path = case.project.paths.root / 'graph.sqlite'
    assert retrieval_status(case.project.with_settings(graph_enabled=False))['status'] == 'disabled'
    assert query_graph(case.project, 'Alpha')['status'] == 'unbuilt'
    assert not path.exists()
    path.write_text('corrupt')
    assert query_graph(case.project, 'Alpha')['status'] == 'unavailable'
    path.unlink()
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=5')
    assert query_graph(case.project, 'Alpha')['status'] == 'unavailable'


@pytest.mark.parametrize('status', ['building', 'interrupted', 'failed', 'complete_with_gaps'])
def test_incomplete_publication_remains_queryable_and_labeled(graph_case, status):
    case = graph_case
    case.publish('a.md', 'Alpha funds Beta.')
    with graph_for(case.project) as graph:
        graph.connection.execute('UPDATE graph_build SET status=? WHERE id=1', (status,))
        graph.connection.commit()
    result = query_graph(case.project, 'Alpha')
    assert result['claims'] and result['graph']['incomplete']
    assert 'incomplete' in result['note']


def test_source_change_is_reported_as_stale_with_original_version(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.')
    with graph_for(case.project) as graph:
        graph.state.completed('a.md', edge.provenance['source_fingerprint'], edge.provenance['source_version'])
        graph.connection.execute("UPDATE graph_build SET started_at='2026-01-01' WHERE id=1")
        graph.connection.commit()
    assert not query_graph(case.project, 'Alpha')['graph']['stale']
    (case.project.paths.root / 'a.md').write_text('Alpha no longer funds Beta.')
    result = query_graph(case.project, 'Alpha')
    assert result['graph']['stale'] and 'stale' in result['note']
    assert result['claims'][0]['verification']['snapshot'] == 'found'
    assert result['claims'][0]['verification']['current'] == 'not_found'
