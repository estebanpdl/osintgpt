'''Read-only build activity and coverage do not masquerade as entity-search outcomes.'''

import json

import pytest

from osintgpt.graph import Entity, graph_for, query_graph, saved_graph_status
from osintgpt.graph.accounting import BuildCounts
from osintgpt.graph.inspection import inspect_entities, inspect_graph
from osintgpt.graph.retrieval import retrieval_status
from osintgpt.graph.state import build_lock
from osintgpt.graph.status_messages import build_messages, count_label, query_outcome
from osintgpt.ingestion.snapshots import load_snapshot
from test_inspection_cli import invoke


@pytest.mark.parametrize('status,available,phrase,active', [
    ('building', True, 'is running', True),
    ('building', False, 'is running', True),
    ('interrupted', True, 'was interrupted', False),
    ('failed', True, 'failed', False),
    ('failed', False, 'failed', False),
    ('complete_with_gaps', True, 'finished with gaps', False),
    ('complete', True, 'finished', False),
    ('stale', True, 'finished', False),
    ('never_built', False, 'No graph build has completed', False),
])
def test_checked_status_has_distinct_activity_and_coverage(status, available, phrase, active):
    messages = build_messages({'status': status, 'available': available})
    assert phrase in messages['activity'] and messages['active'] is active
    coverage = ' '.join(messages['coverage'])
    assert ('No published graph is available yet.' in coverage) is (not available)
    assert ('coverage may be incomplete' in coverage) is (
        available and status in ('building', 'interrupted', 'failed', 'complete_with_gaps'))
    assert ('stale' in coverage) is (status == 'stale')
    assert 'The build is incomplete' not in str(messages)
    assert '0 source' not in coverage


def test_starting_finishing_and_finished_jobs_are_distinct():
    saved = {'status': 'complete', 'available': True}
    starting = build_messages(saved, job={'active': True, 'status': 'starting'})
    finishing = build_messages(saved, job={'active': True, 'status': 'building'})
    finished = build_messages(saved, job={'active': False, 'status': 'complete'})
    assert starting['active'] and 'is starting' in starting['activity']
    assert finishing['active'] and 'is finishing' in finishing['activity']
    assert not finished['active'] and 'finished' in finished['activity']


def test_stale_coverage_and_known_failure_counts_are_independent():
    saved = {'status': 'complete_with_gaps', 'available': True, 'stale': True,
             'counts': {'documents_failed': 2}}
    messages = build_messages(saved)
    assert not messages['active']
    assert any('2 source document(s) failed in the last build.' == note for note in messages['coverage'])
    assert any('explicit build' in note for note in messages['coverage'])
    for value in (None, -1, True, '2'):
        saved['counts']['documents_failed'] = value
        assert not any('source document(s)' in note for note in build_messages(saved)['coverage'])
        assert count_label(saved['counts'], 'documents_failed') == 'unknown'
    assert count_label({'documents_failed': 0}, 'documents_failed') == '0'
    assert count_label({}, 'documents_failed') == 'unknown'


def test_live_lock_and_released_lock_produce_different_messages_without_writes(graph_case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Status and inspection must not construct a provider.')
    monkeypatch.setattr('osintgpt.graph.provider.LazyGraphGenerator.__init__', forbidden)
    graph_case.publish('source.md', 'Alpha supports Beta.')
    project = graph_case.project
    snapshot = load_snapshot(project.paths.root / 'source.md', ref='source.md', cache_dir=project.paths.extracts)
    path = project.paths.root / 'graph.sqlite'
    with build_lock(path):
        with graph_for(project) as graph:
            graph.state.completed('source.md', snapshot.fingerprint, snapshot.version)
            graph.state.begin(BuildCounts())
        running = retrieval_status(project)
        assert running['build_active'] and running['build_message'] == 'A graph build is running.'
        assert query_graph(project, 'Alpha')['claims']
    before = path.read_bytes()
    interrupted = retrieval_status(project)
    assert not interrupted['build_active'] and interrupted['build_status'] == 'interrupted'
    assert interrupted['build_message'] == 'The last graph build was interrupted.'
    assert path.read_bytes() == before
    with graph_for(project) as graph:
        assert graph.connection.execute('SELECT status FROM graph_build').fetchone()[0] == 'building'


def test_no_match_isolation_and_disabled_retrieval_keep_separate_outcomes(graph_case):
    graph_case.publish('source.md', 'Alpha supports Beta.')
    with graph_for(graph_case.project) as graph:
        graph.add([Entity('isolated', 'Separate identity')], [])
        graph.state.finish('complete_with_gaps')
    missing = inspect_entities(graph_case.project, 'Absent identity')
    isolated = inspect_graph(graph_case.project, 'isolated').result
    assert query_outcome(missing) == 'No matching identity was found in the published graph.'
    assert query_outcome(isolated) == 'This identity has no accepted relationships in the published graph.'
    for result in (missing, isolated):
        assert result['graph']['incomplete'] and not result['graph']['build_active']
        assert result['graph']['build_message'] == 'The last graph build finished with gaps.'
        assert 'build' not in query_outcome(result) and 'truth' not in query_outcome(result)
    project = graph_case.project.with_settings(graph_enabled=False)
    assert query_graph(project, 'Alpha')['status'] == 'disabled'
    assert inspect_graph(project, 'Alpha').rows
    assert not inspect_graph(project, 'Alpha').result['enabled']


def test_unused_and_unpublished_failure_do_not_claim_prior_published_data(graph_case):
    project = graph_case.project
    unbuilt = inspect_entities(project, 'Alpha')
    assert query_outcome(unbuilt) == 'No published graph is available to search.'
    assert not (project.paths.root / 'graph.sqlite').exists()
    with graph_for(project) as graph:
        graph.state.begin(BuildCounts(documents_failed=1))
        graph.state.finish('failed')
    failed = retrieval_status(project)
    assert failed['build_message'] == 'The last graph build failed.'
    assert not failed['available'] and not failed['build_active']
    assert 'Published claims remain available' not in failed['note']


def test_cli_prints_coverage_and_search_outcome_on_separate_lines(graph_case):
    graph_case.publish('source.md', 'Alpha supports Beta.')
    with graph_for(graph_case.project) as graph:
        graph.state.finish('complete_with_gaps')
    human = invoke(graph_case, 'entities', 'Absent')
    assert human.exit_code == 0, human.output
    lines = human.output.splitlines()
    assert 'The last graph build finished with gaps.' in lines
    assert 'No matching identity was found in the published graph.' in lines
    assert 'The build is incomplete' not in human.output
    machine = json.loads(invoke(graph_case, 'entities', 'Absent', '--json').output)
    assert machine['graph']['coverage_notes'] and machine['status'] == 'entity_not_found'
    assert not machine['graph']['build_active']
