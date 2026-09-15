'''Pacing changes preserve step 1 checkpoints and step 2 completion fingerprints.'''

import sqlite3

import pytest

from test_checkpoints import Endpoint, project, provider
from osintgpt import index_project
from osintgpt.llm.embedding_pacing import EmbeddingPacer
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from osintgpt.rate_settings import RateLimits
from tests.llm.test_rate_ledger import Clock, ledger


def paced(endpoint, clock, *, batch_tokens=120, recorder=None):
    embedder = provider('openai', endpoint, recorder)
    embedder.pacer = EmbeddingPacer(
        'openai', embedder.model, 'https://offline.invalid', 'test',
        RateLimits(rpm=60, tpm=1000, batch_tokens=batch_tokens), ledger=ledger(clock),
        retry_policy=RetryPolicy(max_attempts=1)
    )
    return embedder


def test_resume_reuses_prefix_when_operator_changes_quota_and_batch_budget(project):
    clock = Clock()
    first = Endpoint(fail_at=2)
    assert index_project(project, paced(first, clock)).failed
    with sqlite3.connect(project.paths.index_journal) as connection:
        saved = connection.execute('SELECT completed FROM jobs').fetchone()[0]
    assert 0 < saved < 100
    second = Endpoint()
    assert index_project(project, paced(second, clock, batch_tokens=60)).chunks == 250
    assert second.calls[0][0] == f'Record {saved} contains evidence.'
    assert sum(map(len, second.calls)) == 250 - saved
    third = Endpoint()
    assert index_project(project, paced(third, clock, batch_tokens=240)).unchanged == 1
    assert third.calls == []


def test_cost_stop_does_not_reserve_or_send_the_next_batch(project):
    clock = Clock()
    endpoint = Endpoint()
    recorder = UsageRecorder(cost_ceiling_usd=0.00000001)
    embedder = paced(endpoint, clock, recorder=recorder)
    with pytest.raises(CostLimitReached):
        index_project(project, embedder)
    assert len(endpoint.calls) == recorder.calls == 1
    with embedder.pacer.ledger.transaction() as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1
    with sqlite3.connect(project.paths.index_journal) as connection:
        assert connection.execute('SELECT completed FROM jobs').fetchone()[0] == len(endpoint.calls[0])
