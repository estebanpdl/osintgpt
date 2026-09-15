'''Retry exhaustion, recovery and cost stops through durable indexing.'''

import pytest

from test_checkpoints import Endpoint, checkpoint_counts, project, provider, stored_vectors
from osintgpt import index_project
from osintgpt.llm.embedding_pacing import EmbeddingPacer
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from osintgpt.rate_settings import RateLimits
from tests.llm.test_embedding_errors import api_error
from tests.llm.test_rate_ledger import Clock, ledger


def retrying(kind, endpoint, project, clock, recorder=None):
    instance = provider(kind, endpoint, recorder)
    instance.pacer = EmbeddingPacer(
        kind, instance.model, 'https://offline.invalid', 'test',
        RateLimits(rpm=5000, tpm=2_000_000),
        ledger=ledger(clock, project.paths.root / 'test-rates.sqlite'),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    instance.pacer.retry_jitter = lambda low, high: 0
    return instance


class UnavailableAfterFirst(Endpoint):
    def response(self, inputs):
        if self.calls:
            self.calls.append(list(inputs))
            raise self.error
        return super().response(inputs)


@pytest.mark.parametrize('kind', ['openai', 'voyage', 'gemini'])
def test_exhaustion_keeps_successful_prefix_and_restart_embeds_only_remainder(project, kind):
    clock = Clock()
    endpoint = UnavailableAfterFirst(error=api_error(kind))
    recorder = UsageRecorder()
    report = index_project(project, retrying(kind, endpoint, project, clock, recorder))
    assert len(report.failed) == 1
    assert [len(batch) for batch in endpoint.calls] == [100, 100, 100, 100]
    assert all(batch[0] == 'Record 100 contains evidence.' for batch in endpoint.calls[1:])
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    assert stored_vectors(project) == []
    assert recorder.calls == 1

    resumed = Endpoint()
    report = index_project(project, retrying(kind, resumed, project, clock))
    assert not report.failed and report.chunks == 250
    assert [len(batch) for batch in resumed.calls] == [100, 50]
    assert resumed.calls[0][0] == 'Record 100 contains evidence.'
    assert len(stored_vectors(project)) == 250 and checkpoint_counts(project) == []
    completed = Endpoint()
    assert index_project(project, retrying(kind, completed, project, clock)).unchanged == 1
    assert completed.calls == []


@pytest.mark.parametrize('kind', ['openai', 'voyage', 'gemini'])
def test_transient_failure_recovers_in_same_run_and_records_success_once(project, kind):
    clock = Clock()
    endpoint = Endpoint(fail_at=2, error=api_error(kind))
    recorder = UsageRecorder()
    report = index_project(project, retrying(kind, endpoint, project, clock, recorder))
    assert not report.failed and report.chunks == 250
    assert [len(batch) for batch in endpoint.calls] == [100, 100, 100, 50]
    assert endpoint.calls[1] == endpoint.calls[2]
    assert recorder.calls == 3 and recorder.input_tokens == 250
    assert len(stored_vectors(project)) == 250 and checkpoint_counts(project) == []


def test_cost_stop_after_recovered_response_saves_checkpoint_before_next_attempt(project):
    clock = Clock()
    endpoint = Endpoint(fail_at=1, error=api_error())
    recorder = UsageRecorder(cost_ceiling_usd=0.00000001)
    embedder = retrying('openai', endpoint, project, clock, recorder)
    with pytest.raises(CostLimitReached):
        index_project(project, embedder)
    assert len(endpoint.calls) == 2 and recorder.calls == 1
    assert checkpoint_counts(project) == [(250, 100, 'embedding')]
    with embedder.pacer.ledger.transaction() as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 2


def test_cancelled_retry_keeps_prior_checkpoint_and_does_not_reserve_next_attempt(project):
    clock = Clock()
    endpoint = Endpoint(fail_at=2, error=api_error())
    embedder = retrying('openai', endpoint, project, clock)
    def sleep(seconds):
        if len(endpoint.calls) == 2:
            raise KeyboardInterrupt()
        clock.sleep(seconds)
    embedder.pacer.ledger.sleep = sleep
    with pytest.raises(KeyboardInterrupt):
        index_project(project, embedder)
    assert len(endpoint.calls) == 2 and checkpoint_counts(project) == [(250, 100, 'embedding')]
    with embedder.pacer.ledger.transaction() as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 2
