'''Provider-free coverage, bounded pair inspection, and multilingual filtering.'''

from osintgpt.graph import graph_for
from osintgpt.graph.inspection import overview, inspect_assertion, filter_claims, rejected_proposals
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot


def test_coverage_includes_zero_assertion_sources_and_distinguishes_gaps(graph_case):
    case = graph_case
    case.publish('a.md', 'Alpha funds Beta.')
    empty = case.project.paths.root / 'empty.md'
    empty.write_text('No relationships in this source.')
    Corpus.load(case.project.paths.sources).register('empty.md')
    snapshot = load_snapshot(empty, ref='empty.md', cache_dir=case.project.paths.extracts)
    with graph_for(case.project) as graph:
        graph.state.completed('empty.md', snapshot.fingerprint, snapshot.version, quality_gaps=True)
    result = overview(case.project)
    assert result['coverage'] == {'registered': 2, 'current': 1, 'with_gaps': 1}
    assert next(row for row in result['sources'] if row['source'] == 'empty.md')['published']
    empty.unlink()
    result = overview(case.project)
    assert next(row for row in result['sources'] if row['source'] == 'empty.md')['state'] == 'unavailable'


def test_inspector_keeps_all_quotes_qualifications_and_competing_directions(graph_case):
    case = graph_case
    chosen = case.publish('a.md', 'Según X, Alpha no financió Beta. En 2024.',
                          quotes=['Alpha no financió Beta.', 'En 2024.'], relation_polarity='negative',
                          assertion='alleged', attributed_to='X', event_time='2024')
    case.publish('b.md', 'Beta funds Alpha.', source='Beta', target='Alpha')
    case.publish('c.md', 'Alpha funds Beta.')
    result = inspect_assertion(case.project, chosen.assertion_id, limit=2)
    assert result['truncated'] and len(result['claims']) == 2
    row = result['claims'][0]
    assert row['assertion_id'] == chosen.assertion_id and row['evidence_quotes'] == list(chosen.evidence_quotes)
    assert (row['relation_polarity'], row['attributed_to'], row['event_time']) == ('negative', 'X', '2024')
    assert row['verification']['snapshot'] == 'found'
    assert filter_claims(result['claims'], text='NO FINANCIÓ', verification='found') == [row]
    assert inspect_assertion(case.project, 'nonexistent')['status'] == 'assertion_not_found'
    assert rejected_proposals(case.project) == []


def test_chart_keeps_opposite_arrows_and_all_explicitly_selected_assertions(graph_case):
    from osintgpt.app.views.graph_chart import diagram_data, selected_assertion
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.')
    case.publish('b.md', 'Beta funds Alpha.', source='Beta', target='Alpha')
    claims = inspect_assertion(case.project, edge.assertion_id)['claims']
    data = diagram_data(claims, starting_id=edge.source_id)
    assert len({row['assertion_id'] for row in data['edges']}) == 2
    assert {(item['source'], item['target']) for item in data['edges']} == {('Alpha', 'Beta'), ('Beta', 'Alpha')}
    assert selected_assertion({'selected': {'scope': data['scope'], 'assertion_id': edge.assertion_id}}, data) == edge.assertion_id
    assert len(diagram_data(claims * 40, starting_id=edge.source_id)['edges']) == len(claims) * 40
