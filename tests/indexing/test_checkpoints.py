'''Failure and restart tests using the real paid-provider batching paths.'''

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
import pytest
from openai import RateLimitError

from osintgpt import Project, index_project
from osintgpt.ingestion import Corpus, FieldMapping, IndexState
from osintgpt.ingestion.checkpoints import IndexJournal
from osintgpt.llm.gemini import GeminiEmbedding
from osintgpt.llm.openai_compat import OpenAICompatEmbedding
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from osintgpt.vector_store import SQLiteVectorStore


class Endpoint:
    def __init__(self, fail_at=None, error=None):
        self.calls = []
        self.fail_at = fail_at
        self.error = error

    def response(self, inputs):
        self.calls.append(list(inputs))
        if len(self.calls) == self.fail_at:
            if self.error is not None:
                raise self.error
            raise RateLimitError(
                'temporary token limit', body={'code': 'rate_limit_exceeded'},
                response=httpx.Response(429, request=httpx.Request('POST', 'https://offline.invalid'))
            )
        return [[float(text.split()[1]), 1.0] for text in inputs]

    def create(self, *, input, model):
        return NS(
            data=[NS(index=i, embedding=v) for i, v in enumerate(self.response(input))],
            usage=NS(prompt_tokens=len(input))
        )

    def embed_content(self, *, contents, model, config):
        return NS(embeddings=[
            NS(values=v, statistics=NS(token_count=1)) for v in self.response(contents)
        ])


def provider(kind, endpoint, recorder=None, batch_size=100):
    # No live clients, credentials, network, or SDK retry sleeps.
    cls = GeminiEmbedding if kind == 'gemini' else OpenAICompatEmbedding
    instance = cls.__new__(cls)
    instance.model = 'gemini-embedding-001' if kind == 'gemini' else 'text-embedding-3-small'
    instance.batch_size = batch_size
    instance.recorder = recorder
    instance.provider = kind
    instance.billable = True
    instance.task_style = 'task_type'
    instance.client = NS(embeddings=endpoint, models=endpoint)
    return instance


@pytest.fixture
def project(tmp_path):
    instance = Project.create('Checkpoint case', home=tmp_path)
    path = instance.paths.root / 'records.csv'
    path.write_text(
        'id,message\n' + ''.join(f'{i},Record {i} contains evidence.\n' for i in range(250)),
        encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register(
        path, FieldMapping(content=('message',), identity='id')
    )
    return instance


def checkpoint_counts(project):
    with sqlite3.connect(project.paths.index_journal) as connection:
        return connection.execute('SELECT total, completed, status FROM jobs').fetchall()


def stored_vectors(project):
    with SQLiteVectorStore(project.paths.store) as store:
        return store.connection.execute(
            'SELECT sequence, text, vector FROM chunks ORDER BY sequence'
        ).fetchall()


@pytest.mark.parametrize('kind', ['openai', 'voyage', 'gemini'])
def test_resume_skips_each_saved_response_after_provider_failure(project, kind):
    first = Endpoint(fail_at=2)
    report = index_project(project, provider(kind, first))
    assert len(report.failed) == 1
    assert stored_vectors(project) == []
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    assert len(IndexState.load(project.paths.index_state)) == 0
    with sqlite3.connect(project.paths.index_journal) as connection:
        row = connection.execute(
            'SELECT document_ref, chunk FROM completed_chunks WHERE sequence = 99'
        ).fetchone()
    assert row[0].endswith('records.csv#99')
    assert json.loads(row[1])['text'] == 'Record 99 contains evidence.'

    second = Endpoint()
    report = index_project(Project.load(project.paths.root), provider(kind, second))
    assert not report.failed
    assert [len(batch) for batch in second.calls] == [100, 50]
    assert second.calls[0][0] == 'Record 100 contains evidence.'
    assert report.chunks == 250
    assert checkpoint_counts(project) == []
    assert len(stored_vectors(project)) == 250
    third = Endpoint()
    assert index_project(project, provider(kind, third)).unchanged == 1
    assert third.calls == []


def test_budget_crossing_response_is_saved_before_the_stop(project):
    first = Endpoint()
    recorder = UsageRecorder(cost_ceiling_usd=0.000003)
    with pytest.raises(CostLimitReached):
        index_project(project, provider('openai', first, recorder))
    assert [len(batch) for batch in first.calls] == [100, 100]
    assert recorder.calls == 2
    assert checkpoint_counts(project) == [(250, 200, 'embedding')]
    second = Endpoint()
    index_project(project, provider('openai', second, UsageRecorder()))
    assert [len(batch) for batch in second.calls] == [50]


def test_missing_usage_ceiling_also_saves_successful_vectors(project):
    first = Endpoint()
    create = first.create
    first.create = lambda **kwargs: NS(data=create(**kwargs).data)
    with pytest.raises(CostLimitReached, match='no usage'):
        index_project(project, provider('openai', first, UsageRecorder(cost_ceiling_usd=1)))
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]


def test_interruption_preserves_batches_and_releases_project_lock(project):
    first = Endpoint(fail_at=2, error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        index_project(project, provider('openai', first))
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    second = Endpoint()
    index_project(project, provider('openai', second))
    assert second.calls[0][0] == 'Record 100 contains evidence.'


def test_old_searchable_index_survives_failed_forced_replacement(project):
    index_project(project, provider('openai', Endpoint()))
    original = stored_vectors(project)
    replacement = provider('openai', Endpoint(fail_at=2))
    replacement.model = 'another-model'
    report = index_project(project, replacement, force=True, purge_other_models=True)
    assert report.failed
    assert stored_vectors(project) == original
    # The file hash is unchanged, but the unfinished replacement must resume.
    second = Endpoint()
    resumed = provider('openai', second)
    resumed.model = 'another-model'
    assert index_project(project, resumed).chunks == 250
    assert second.calls[0][0] == 'Record 100 contains evidence.'


def test_store_failure_retries_publication_without_embedding(project):
    with SQLiteVectorStore(project.paths.store) as store:
        store.upsert = lambda *args: (_ for _ in ()).throw(RuntimeError('disk unavailable'))
        assert index_project(project, provider('openai', Endpoint()), store=store).failed
    assert checkpoint_counts(project) == [(250, 250, 'ready')]
    second = Endpoint()
    assert index_project(project, provider('openai', second)).chunks == 250
    assert second.calls == []


def test_state_write_interruption_reuses_already_published_vectors(project, monkeypatch):
    save = IndexState.save
    monkeypatch.setattr(IndexState, 'save', lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        index_project(project, provider('openai', Endpoint()))
    assert len(stored_vectors(project)) == 250
    assert checkpoint_counts(project) == [(250, 250, 'ready')]
    monkeypatch.setattr(IndexState, 'save', save)
    second = Endpoint()
    assert index_project(project, provider('openai', second)).chunks == 250
    assert second.calls == []


@pytest.mark.parametrize('change', ['source', 'mapping', 'model', 'provider', 'force'])
def test_incompatible_pending_jobs_are_not_reused(project, change):
    index_project(project, provider('openai', Endpoint(fail_at=2)))
    if change == 'source':
        path = project.paths.root / 'records.csv'
        path.write_text(path.read_text() + '250,Record 250 contains evidence.\n')
    if change == 'mapping':
        Corpus.load(project.paths.sources).register(
            'records.csv', FieldMapping(content=('message',), identity='id', min_chars=10)
        )
    endpoint = Endpoint()
    embedder = provider('voyage' if change == 'provider' else 'openai', endpoint)
    if change == 'model':
        embedder.model = 'different-model'
    report = index_project(project, embedder, force=change == 'force')
    assert not report.failed
    assert endpoint.calls[0][0] == 'Record 0 contains evidence.'


def test_resume_can_change_batch_size(project):
    index_project(project, provider('openai', Endpoint(fail_at=2)))
    second = Endpoint()
    index_project(project, provider('openai', second, batch_size=75))
    assert [len(batch) for batch in second.calls] == [75, 75]
    assert second.calls[0][0] == 'Record 100 contains evidence.'


def test_another_process_cannot_index_the_same_project(project):
    with IndexJournal(project.paths.index_journal):
        attempt = subprocess.run(
            [sys.executable, '-c',
             'from pathlib import Path; import sys; '
             'from osintgpt.ingestion.checkpoints import IndexJournal; '
             'IndexJournal(Path(sys.argv[1]))', str(project.paths.index_journal)],
            capture_output=True, text=True, timeout=30
        )
    assert attempt.returncode != 0
    assert 'already being indexed' in attempt.stderr


def test_invalid_response_does_not_advance_checkpoint(project):
    endpoint = Endpoint()
    create = endpoint.create
    endpoint.create = lambda **kwargs: NS(data=create(**kwargs).data[:-1])
    assert index_project(project, provider('openai', endpoint)).failed
    assert checkpoint_counts(project) == [(250, 0, 'embedding')]
    assert stored_vectors(project) == []


def test_process_death_preserves_committed_batches(project):
    script = '''
import os, runpy, sys
from osintgpt import Project, index_project
helpers = runpy.run_path(sys.argv[1])
endpoint = helpers['Endpoint']()
respond = endpoint.response
def response(inputs):
    if endpoint.calls:
        os._exit(23)
    return respond(inputs)
endpoint.response = response
index_project(Project.load(sys.argv[2]), helpers['provider']('openai', endpoint))
'''
    stopped = subprocess.run(
        [sys.executable, '-c', script, str(Path(__file__).resolve()), str(project.paths.root)],
        capture_output=True, text=True, timeout=30
    )
    assert stopped.returncode == 23, stopped.stderr
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    second = Endpoint()
    assert index_project(project, provider('openai', second)).chunks == 250
    assert second.calls[0][0] == 'Record 100 contains evidence.'


def test_disk_failure_rolls_back_the_entire_batch(project):
    index_project(project, provider('openai', Endpoint(fail_at=2)))
    with sqlite3.connect(project.paths.index_journal) as connection:
        connection.execute('''
            CREATE TRIGGER fail_checkpoint BEFORE INSERT ON completed_chunks
            WHEN NEW.sequence = 150 BEGIN
                SELECT RAISE(ABORT, 'simulated disk failure');
            END
        ''')
    recorder = UsageRecorder()
    report = index_project(project, provider('openai', Endpoint(), recorder))
    assert report.failed
    assert recorder.calls == 1
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    with sqlite3.connect(project.paths.index_journal) as connection:
        assert connection.execute('SELECT COUNT(*) FROM completed_chunks').fetchone()[0] == 100
        connection.execute('DROP TRIGGER fail_checkpoint')
    second = Endpoint()
    assert index_project(project, provider('openai', second)).chunks == 250
    assert second.calls[0][0] == 'Record 100 contains evidence.'


def test_interrupted_state_write_preserves_previous_state(project, monkeypatch):
    from osintgpt.ingestion import indexing

    index_project(project, provider('openai', Endpoint()))
    original = project.paths.index_state.read_bytes()

    def partial_write(path, *args, **kwargs):
        path.write_text('broken = [')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(indexing, 'write_toml', partial_write)
    with pytest.raises(OSError):
        IndexState.load(project.paths.index_state).save()
    assert project.paths.index_state.read_bytes() == original
    assert len(IndexState.load(project.paths.index_state)) == 1


def test_source_change_before_publication_keeps_previous_index(project):
    index_project(project, provider('openai', Endpoint()))
    original = stored_vectors(project)
    path = project.paths.root / 'records.csv'
    endpoint = Endpoint()
    create = endpoint.create

    def response(**kwargs):
        result = create(**kwargs)
        if len(endpoint.calls) == 3:
            path.write_text(path.read_text() + '250,Record 250 contains evidence.\n')
        return result

    endpoint.create = response
    report = index_project(project, provider('openai', endpoint), force=True)
    assert 'Source changed' in report.failed[0].problem
    assert stored_vectors(project) == original
    second = Endpoint()
    assert index_project(project, provider('openai', second)).chunks == 251
    assert second.calls[0][0] == 'Record 0 contains evidence.'
