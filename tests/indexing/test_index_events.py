'''Progress is truthful about durability and isolated from other indexing runs.'''

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_checkpoints import Endpoint, project, provider
from osintgpt import index_project
from osintgpt.index_events import emit_index, observe_index
from osintgpt.index_status import saved_index_status, model_changes
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.ingestion.checkpoints import IndexJournal
from osintgpt.ingestion.preview import preview_file


def test_checkpoint_notifications_follow_commit_and_precede_usage(project):
    events = []
    def observe(event):
        events.append(event)
        if event.kind == 'checkpoint':
            with sqlite3.connect(project.paths.index_journal) as connection:
                assert connection.execute('SELECT completed FROM jobs').fetchone()[0] == event.data['completed']
    index_project(project, provider('openai', Endpoint()), on_event=observe)
    kinds = [event.kind for event in events]
    assert kinds.index('checkpoint') < kinds.index('usage') < kinds.index('published')
    checkpoints = [event.data for event in events if event.kind == 'checkpoint']
    assert [event['completed'] for event in checkpoints] == [100, 200, 250]
    assert [event['completed_records'] for event in checkpoints] == [100, 200, 250]
    assert checkpoints[0]['statistics']['kept'] == 250


def test_exclusion_counts_are_saved_and_previewed_without_an_extra_read(project, monkeypatch):
    path = project.paths.root / 'records.csv'
    path.write_text('id,message\na,Record 1 contains useful evidence.\nb,\nc,tiny\n', encoding='utf-8')
    Corpus.load(project.paths.sources).register(path, FieldMapping(content=('message',), min_chars=10))
    from osintgpt.ingestion import tabular
    original, reads = tabular._records, []
    def read(*args):
        reads.append(args)
        return original(*args)
    monkeypatch.setattr(tabular, '_records', read)
    report = index_project(project, provider('openai', Endpoint()))
    assert not report.failed
    assert len(reads) == 1
    counts = saved_index_status(project)['published'][0]['statistics']
    assert (counts['records'], counts['kept'], counts['without_content'], counts['too_short']) == (3, 1, 1, 1)
    preview = preview_file(path, FieldMapping(content=('message',), min_chars=10))
    assert preview.statistics == {key: counts[key] for key in ('records', 'kept', 'without_content', 'too_short')}


def test_read_only_status_creates_nothing_and_reads_legacy_journal(project):
    assert not project.paths.index_journal.exists()
    assert saved_index_status(project)['pending'] == []
    assert not project.paths.index_journal.exists()
    with sqlite3.connect(project.paths.index_journal) as connection:
        connection.execute('CREATE TABLE jobs (ref TEXT, total INTEGER, completed INTEGER, status TEXT)')
        connection.execute("INSERT INTO jobs VALUES ('old.csv', 250, 100, 'embedding')")
    status = saved_index_status(project)
    assert status['pending'][0]['completed'] == 100
    assert status['pending'][0]['details'] == {}
    with sqlite3.connect(project.paths.index_journal) as connection:
        assert 'details' not in {row[1] for row in connection.execute('PRAGMA table_info(jobs)')}


def test_status_can_read_checkpoints_while_writer_holds_project_lock(project):
    with IndexJournal(project.paths.index_journal) as journal:
        journal.prepare('a.csv', 'signature', 20, details={'provider': 'openai', 'model': 'text-embedding-3-small'})
        status = saved_index_status(project)
        assert status['pending'][0]['total'] == 20
        assert not status['problems']
        assert model_changes(status, 'gemini', 'gemini-embedding-001') == [('openai', 'text-embedding-3-small')]
        assert model_changes(status, 'openai', 'text-embedding-3-small') == []


def test_observer_failure_does_not_lose_vectors_or_repeat_requests(project):
    endpoint = Endpoint()
    def broken(event):
        raise RuntimeError('display unavailable')
    report = index_project(project, provider('openai', endpoint), on_event=broken)
    assert report.chunks == 250 and len(endpoint.calls) == 3
    assert saved_index_status(project)['pending'] == []


def test_observers_are_isolated_between_threads_and_removed_after_scope():
    def run(label):
        events = []
        with observe_index(events.append):
            emit_index('checkpoint', ref=label)
        emit_index('checkpoint', ref='outside')
        return events
    with ThreadPoolExecutor(2) as pool:
        first, second = list(pool.map(run, ['first', 'second']))
    assert [event.data['ref'] for event in first] == ['first']
    assert [event.data['ref'] for event in second] == ['second']


def test_corrupt_checkpoint_status_reports_unknown_instead_of_empty_success(project):
    project.paths.index_journal.write_bytes(b'not a database')
    assert saved_index_status(project)['problems']


def test_partly_embedded_record_is_not_counted_as_complete(project, monkeypatch):
    from osintgpt.ingestion import chunking
    monkeypatch.setattr(chunking, 'MAX_CHARS', 1000)
    path = project.paths.root / 'records.csv'
    path.write_text('id,message\n1,' + 'Record 1 contains evidence. ' * 200 + '\n', encoding='utf-8')
    events = []
    report = index_project(project, provider('openai', Endpoint(fail_at=2), batch_size=1), on_event=events.append)
    assert report.failed
    saved = [event.data for event in events if event.kind == 'checkpoint'][0]
    assert saved['completed'] == 1 and saved['total'] > 1
    assert saved['completed_records'] == 0
    assert saved_index_status(project)['pending'][0]['details']['completed_records'] == 0


def test_old_journal_migration_preserves_checkpoint_and_adds_display_metadata(project):
    # A step-1/2 journal has no details column. Opening it for indexing migrates
    # only display metadata, preserving the fingerprint and completed prefix.
    with sqlite3.connect(project.paths.index_journal) as connection:
        connection.execute('CREATE TABLE jobs (ref TEXT PRIMARY KEY, fingerprint TEXT, total INTEGER, completed INTEGER, status TEXT)')
        connection.execute("INSERT INTO jobs VALUES ('old.csv', 'same', 5, 2, 'embedding')")
    with IndexJournal(project.paths.index_journal) as journal:
        assert journal.prepare('old.csv', 'same', 5, details={'provider': 'openai'}) == 2
    saved = saved_index_status(project)['pending'][0]
    assert saved['completed'] == 2 and saved['details']['provider'] == 'openai'
