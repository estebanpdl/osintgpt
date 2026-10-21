'''Graph build state and accounting without network or provider credentials.'''

import json
import sqlite3

import pytest

from osintgpt import Project
from osintgpt.graph import GraphStore, build_graph, graph_for, saved_graph_status
from osintgpt.graph.accounting import BuildCounts
from osintgpt.graph.state import SCHEMA_VERSION, build_lock
from osintgpt.ingestion import Corpus, Document
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder

EMPTY = json.dumps({'entities': [], 'edges': []})


class Generator:
    model = 'offline-graph'

    def __init__(self, *outcomes, recorder=None, observe=None):
        self.outcomes = outcomes or (EMPTY,)
        self.recorder = recorder
        self.observe = observe
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        if self.observe:
            self.observe()
        result = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(result, BaseException):
            raise result
        if self.recorder is not None:
            self.recorder.record(Usage('stub', self.model, 10, 5))
        return result


@pytest.fixture
def project(tmp_path):
    project = Project.create('Graph counts', home=tmp_path).with_settings(graph_enabled=True)
    project.save()
    material = project.paths.root / 'material'
    material.mkdir()
    (material / 'a.md').write_text('Alpha supports Beta.', encoding='utf-8')
    Corpus.load(project.paths.sources).register('material')
    return project


def test_status_of_unused_graph_is_read_only(project):
    assert saved_graph_status(project)['status'] == 'never_built'
    assert not (project.paths.root / 'graph.sqlite').exists()


def test_empty_extraction_completes_and_is_skipped_incrementally(project):
    first = build_graph(project, Generator())
    assert first.status == 'complete'
    assert first.edges == 0
    generator = Generator()
    second = build_graph(project, generator, incremental=True)
    assert second.skipped == 1
    assert second.calls == generator.calls == 0
    with graph_for(project) as graph:
        assert graph.is_built
        assert graph.refs() == ['material/a.md']
    assert saved_graph_status(project)['status'] == 'complete'


def test_no_source_documents_is_a_completed_empty_build(project):
    (project.paths.root / 'material' / 'a.md').unlink()
    result = build_graph(project, Generator())
    assert result.status == 'complete'
    assert result.calls == 0
    with graph_for(project) as graph:
        assert graph.is_built


def test_long_document_counts_each_window_and_request(project):
    (project.paths.root / 'material' / 'a.md').write_text('x' * 100000)
    generator = Generator()
    report = build_graph(project, generator)
    counts = report.accounting
    assert report.calls == generator.calls > 1
    assert counts.documents_total == counts.documents_completed == counts.records_loaded == 1
    assert counts.windows_total == counts.windows_completed == counts.requests_succeeded == generator.calls
    assert counts.requests_failed == 0
    assert report.usage['complete'] is False
    assert report.usage['total_tokens'] is None
    assert report.usage['missing_usage_requests'] == generator.calls
    assert report.usage['transport_attempts'] is None
    assert saved_graph_status(project)['counts'] == counts.to_dict()


def test_record_counts_are_separate_from_files(project, monkeypatch):
    monkeypatch.setattr('osintgpt.ingestion.snapshots.load_documents', lambda *args, **kwargs: [
        Document(ref='a#1', text='First'), Document(ref='a#2', text='Second')
    ])
    report = build_graph(project, Generator())
    assert report.accounting.documents_completed == 1
    assert report.accounting.records_loaded == 2


@pytest.mark.parametrize('reply', ['not json', '{', '[]', '{}',
                                  '{"entities": {}, "edges": []}',
                                  '{"entities": [5], "edges": []}',
                                  '[{"entities": [], "edges": []}]',
                                  '{"entities": [{"name": null}], "edges": []}'])
def test_invalid_output_fails_a_document_with_visible_accounting(project, reply):
    report = build_graph(project, Generator(reply))
    assert report.status == 'failed'
    assert len(report.failed) == 1
    counts = report.accounting
    assert counts.requests_succeeded == counts.invalid_responses == 2
    assert counts.windows_completed == 0
    assert not saved_graph_status(project)['available']


def test_failure_late_in_document_keeps_actual_request_counts(project):
    (project.paths.root / 'material' / 'a.md').write_text('x' * 100000)
    report = build_graph(project, Generator(EMPTY, EMPTY, RuntimeError('Unavailable')))
    counts = report.accounting
    assert report.status == 'failed'
    assert report.calls == 3
    assert counts.windows_total > report.calls
    assert counts.windows_completed == counts.requests_succeeded == 2
    assert counts.requests_failed == counts.documents_failed == 1


def test_unreadable_document_does_not_count_as_a_model_request(project, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError('Unreadable source')
    monkeypatch.setattr('osintgpt.ingestion.snapshots.load_documents', unreadable)
    report = build_graph(project, Generator())
    assert report.calls == report.accounting.windows_total == 0
    assert report.accounting.documents_failed == 1


def test_mixed_success_is_complete_with_gaps(project):
    (project.paths.root / 'material' / 'b.md').write_text('Another source')
    report = build_graph(project, Generator(EMPTY, RuntimeError('Unavailable')))
    assert report.status == 'complete_with_gaps'
    status = saved_graph_status(project)
    assert status['available']
    assert 'Unavailable' in status['problem']
    assert status['counts']['documents_completed'] == status['counts']['documents_failed'] == 1


def test_building_is_visible_while_provider_runs(project):
    observed = []
    build_graph(project, Generator(observe=lambda: observed.append(saved_graph_status(project))))
    assert observed[0]['status'] == 'building'
    assert observed[0]['counts']['requests_attempted'] == 1


def test_interrupt_is_persisted_and_propagated(project):
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(KeyboardInterrupt()))
    status = saved_graph_status(project)
    assert status['status'] == 'interrupted'
    assert status['counts']['requests_attempted'] == status['counts']['requests_interrupted'] == 1


def test_abandoned_build_is_reported_interrupted_without_changing_database(project):
    with graph_for(project) as graph:
        with build_lock(graph.path):
            graph.state.begin(BuildCounts(requests_attempted=1))
            assert saved_graph_status(project)['status'] == 'building'
        status = saved_graph_status(project)
        assert status['status'] == 'interrupted'
        assert status['counts']['requests_interrupted'] == 1
        assert graph.connection.execute('SELECT status FROM graph_build').fetchone()[0] == 'building'


def test_second_build_is_refused_without_overwriting_active_state(project):
    with graph_for(project) as graph, build_lock(project.paths.root / 'graph.sqlite'):
        graph.state.begin(BuildCounts())
        with pytest.raises(ValueError, match='already running'):
            build_graph(project, Generator())
        assert graph.state.read()['status'] == 'building'


def test_source_changes_are_reported_as_stale(project):
    build_graph(project, Generator())
    (project.paths.root / 'material' / 'a.md').write_text('Changed source')
    assert saved_graph_status(project)['status'] == 'stale'
    report = build_graph(project, Generator(), incremental=True)
    assert report.status == 'complete'
    assert report.calls == 1
    assert build_graph(project, Generator(), rebuild=True).status == 'complete'


def test_added_and_removed_sources_make_status_stale(project):
    build_graph(project, Generator())
    path = project.paths.root / 'material' / 'b.md'
    path.write_text('New source')
    assert saved_graph_status(project)['status'] == 'stale'
    assert build_graph(project, Generator(), incremental=True).status == 'complete'
    path.unlink()
    assert saved_graph_status(project)['status'] == 'stale'


def test_usage_is_for_this_run_and_unknown_pricing_is_explicit(project):
    recorder = UsageRecorder(records=[Usage('stub', 'prior-run', 1000, 1000)])
    report = build_graph(project, Generator(recorder=recorder))
    assert report.usage['input_tokens'] == 10
    assert report.usage['output_tokens'] == 5
    assert report.usage['recorded_responses'] == 1
    assert report.usage['missing_usage_requests'] == 0
    assert report.usage['unpriced_calls'] == 1
    assert report.usage['estimated_cost_usd'] is None
    assert report.usage['complete'] is False


def test_usage_with_verified_pricing_is_complete(project, monkeypatch):
    monkeypatch.setattr('osintgpt.llm.usage.estimate_cost', lambda *args, **kwargs: 0.1)
    report = build_graph(project, Generator(recorder=UsageRecorder()))
    assert report.usage['estimated_cost_usd'] == 0.2
    assert report.usage['complete'] is True


def test_cost_stop_records_response_and_interrupted_state(project):
    generator = Generator(recorder=UsageRecorder(cost_ceiling_usd=0.01))
    with pytest.raises(CostLimitReached):
        build_graph(project, generator)
    status = saved_graph_status(project)
    assert status['status'] == 'interrupted'
    assert status['counts']['requests_succeeded'] == 1
    assert status['counts']['windows_completed'] == 0
    assert status['usage']['total_tokens'] == 15


def test_schema_version_is_set_for_new_graphs(tmp_path):
    with GraphStore(tmp_path / 'graph.sqlite') as graph:
        assert graph.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION


def test_legacy_schema_is_refused_and_preserved(tmp_path):
    path = tmp_path / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE old_data (name TEXT)')
        connection.execute("INSERT INTO old_data VALUES ('keep me')")
    with pytest.raises(ValueError, match='graph migrate --dry-run'):
        GraphStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT name FROM old_data').fetchone()[0] == 'keep me'
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 0
