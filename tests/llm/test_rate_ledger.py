'''Quota timing and concurrency checks without network calls or real sleeps.'''

import sqlite3
import subprocess
import sys

import pytest

from osintgpt.llm.rate_headers import duration
from osintgpt.llm.rate_ledger import RateLedger, TokenBudgetExceeded
from osintgpt.rate_settings import RateLimits


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock():
    return Clock()


def ledger(clock, path=''):
    return RateLedger(path, clock=clock.time, sleep=clock.sleep)


def test_two_million_tpm_is_supported_without_a_one_million_ceiling(clock):
    quota = ledger(clock)
    settings = RateLimits(tpm=2_000_000)
    times = []
    for _ in range(3):
        quota.reserve('scope', 50_000, settings)
        times.append(clock.now)
    assert times == [1000, 1001.5, 1003]


def test_rpm_and_tpm_both_apply_to_unequal_batches(clock):
    quota = ledger(clock)
    settings = RateLimits(rpm=2, tpm=100)
    quota.reserve('scope', 70, settings)
    quota.reserve('scope', 70, settings)
    assert clock.now == 1060  # 140 tokens cannot fit inside any minute.
    quota.reserve('scope', 1, settings)
    assert clock.now == 1102  # Spaces after the preceding 70-token reservation.
    quota.reserve('scope', 1, settings)
    assert clock.now == 1132  # RPM is now the tighter constraint.


def test_another_instance_and_restart_share_capacity(clock, tmp_path):
    path = tmp_path / 'rates.sqlite'
    ledger(clock, path).reserve('scope', 5, RateLimits(rpm=1))
    ledger(clock, path).reserve('scope', 5, RateLimits(rpm=1))
    assert clock.now == 1060


def test_scopes_are_independent(clock):
    quota = ledger(clock)
    quota.reserve('one', 5, RateLimits(rpm=1))
    quota.reserve('two', 5, RateLimits(rpm=1))
    assert clock.now == 1000


def test_cancellation_while_waiting_reserves_nothing(clock):
    quota = ledger(clock)
    quota.reserve('scope', 5, RateLimits(rpm=1))
    def cancel(seconds):
        raise KeyboardInterrupt()
    quota.sleep = cancel
    with pytest.raises(KeyboardInterrupt):
        quota.reserve('scope', 5, RateLimits(rpm=1))
    with quota.transaction() as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1


def test_oversized_request_fails_without_waiting_forever(clock):
    quota = ledger(clock)
    with pytest.raises(TokenBudgetExceeded, match='101'):
        quota.reserve('scope', 101, RateLimits(tpm=100))
    assert not clock.sleeps


def test_response_headers_update_tiers_and_preserve_project_limit(clock):
    quota = ledger(clock)
    settings = RateLimits()
    ticket = quota.reserve('scope', 10, settings)
    quota.observe('scope', ticket, settings, headers={
        'X-RateLimit-Limit-Requests': '5000',
        'x-ratelimit-limit-tokens': '1000000',
        'x-ratelimit-limit-project-tokens': '200000'
    })
    assert quota.limits('scope', settings) == (5000, 200000)
    quota.observe('scope', ticket, settings, headers={'x-ratelimit-limit-tokens': '2000000'})
    assert quota.limits('scope', settings) == (5000, 200000)
    quota.observe('scope', ticket, settings, headers={'x-ratelimit-limit-project-tokens': '2000000'})
    assert quota.limits('scope', settings) == (5000, 2000000)
    assert quota.limits('scope', RateLimits(rpm=600, tpm=1500000)) == (600, 1500000)
    clock.now += 301
    assert quota.limits('scope', settings) == (0, 0)


def test_older_response_cannot_raise_a_newly_lowered_limit(clock):
    quota = ledger(clock)
    settings = RateLimits()
    older = quota.reserve('scope', 1, settings)
    newer = quota.reserve('scope', 1, settings)
    quota.observe('scope', newer, settings, headers={'x-ratelimit-limit-tokens': '1000'})
    quota.observe('scope', older, settings, headers={'x-ratelimit-limit-tokens': '2000'})
    assert quota.limits('scope', settings) == (0, 1000)


def test_remaining_capacity_and_reset_hints_delay_the_next_call(clock):
    quota = ledger(clock)
    settings = RateLimits()
    ticket = quota.reserve('scope', 10, settings)
    quota.observe('scope', ticket, settings, headers={
        'x-ratelimit-remaining-tokens': '0', 'x-ratelimit-reset-tokens': '1.589s'
    })
    quota.reserve('scope', 10, settings)
    assert clock.now == pytest.approx(1001.589)
    assert all(0 < seconds <= 1 for seconds in clock.sleeps)


def test_delayed_headers_cannot_restore_capacity_reserved_by_another_job(clock):
    quota = ledger(clock)
    settings = RateLimits()
    first = quota.reserve('scope', 1, settings)
    quota.observe('scope', first, settings, headers={
        'x-ratelimit-remaining-tokens': '20', 'x-ratelimit-reset-tokens': '10s'
    })
    second = quota.reserve('scope', 15, settings)
    quota.observe('scope', second, settings, headers={
        'x-ratelimit-remaining-tokens': '20', 'x-ratelimit-reset-tokens': '10s'
    })
    quota.reserve('scope', 10, settings)
    assert clock.now == 1010


def test_first_headers_also_account_for_requests_already_in_flight(clock):
    quota = ledger(clock)
    settings = RateLimits()
    first = quota.reserve('scope', 1, settings)
    quota.reserve('scope', 15, settings)
    quota.observe('scope', first, settings, headers={
        'x-ratelimit-remaining-tokens': '20', 'x-ratelimit-reset-tokens': '10s'
    })
    quota.reserve('scope', 10, settings)
    assert clock.now == 1010


def test_actual_usage_increases_reservation_without_refunding_estimates(clock):
    quota = ledger(clock)
    settings = RateLimits(tpm=100)
    ticket = quota.reserve('scope', 10, settings)
    quota.observe('scope', ticket, settings, tokens=90)
    quota.observe('scope', ticket, settings, tokens=5)
    quota.reserve('scope', 20, settings)
    assert clock.now == 1060


def test_expired_ticket_cannot_update_a_new_request(clock):
    quota = ledger(clock)
    settings = RateLimits()
    old = quota.reserve('scope', 10, settings)
    clock.now += 61
    new = quota.reserve('scope', 20, settings)
    assert old != new
    quota.observe('scope', old, settings, tokens=1000)
    with quota.transaction() as connection:
        assert connection.execute('SELECT tokens FROM embedding_requests WHERE id = ?', (new,)).fetchone()[0] == 20


@pytest.mark.parametrize('value, expected', [('1.589s', 1.589), ('200ms', .2), ('6m0s', 360), ('1h2m3.5s', 3723.5), ('bad', None), ('1s-junk', None), ('-1s', None)])
def test_reset_duration_parsing(value, expected):
    assert duration(value) == expected


def test_processes_cannot_overbook_one_quota(tmp_path):
    path = tmp_path / 'rates.sqlite'
    script = '''
import sys
from osintgpt.llm.rate_ledger import RateLedger
from osintgpt.rate_settings import RateLimits
def waiting(seconds):
    raise SystemExit(23)
RateLedger(sys.argv[1], clock=lambda: 1000, sleep=waiting).reserve('shared', 1, RateLimits(rpm=1))
'''
    workers = [subprocess.Popen(
        [sys.executable, '-c', script, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) for _ in range(4)]
    outcomes = []
    try:
        for worker in workers:
            _, error = worker.communicate(timeout=30)
            assert worker.returncode in (0, 23), error.decode()
            outcomes.append(worker.returncode)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
    assert outcomes.count(0) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM embedding_requests').fetchone()[0] == 1
