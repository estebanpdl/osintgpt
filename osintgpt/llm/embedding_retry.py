'''One bounded retry owner for paid embedding HTTP requests, before delivery.'''

import logging
import math
from dataclasses import dataclass
from osintgpt.index_events import emit_index

from .embedding_errors import classify, error_headers
from .rate_ledger import QuotaWaitExpired

log = logging.getLogger('osintgpt.indexing')
ATTEMPT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class RetryPolicy:
    # Includes the initial request. The time budget includes retry pacing and
    # HTTP time; an already-running request is governed by its SDK timeout.
    max_attempts: int = 6
    max_elapsed: float = 300.0
    initial_delay: float = 1.0
    max_backoff: float = 60.0

    def __post_init__(self):
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError('max_attempts must be a positive whole number')
        for field in ('max_elapsed', 'initial_delay', 'max_backoff'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field} must be a positive finite number')

    def delay(self, attempt, minimum, jitter):
        backoff = min(self.max_backoff, self.initial_delay * 2 ** min(attempt - 1, 30))
        # Positive jitter never reduces Retry-After/RetryInfo. The cap is for
        # fallback backoff, not permission to shorten a server-specified wait.
        return max(minimum, backoff) + jitter(0, min(1.0, backoff / 4))


def request_with_retries(pacer, batch, ticket, request, reserve):
    '''Retry only request(batch); parsing validation, sinks and costs stay out.

    reserve(deadline) re-sizes the unfinished prefix and obtains a new quota
    ticket. Every failed attempt remains charged to the local rate window.
    '''
    policy = pacer.retry_policy
    deadline = pacer.ledger.clock() + policy.max_elapsed
    attempt = 1
    while True:
        emit_index('request', attempt=attempt, maximum=policy.max_attempts, inputs=len(batch))
        try:
            response = request(batch)
        except Exception as error:
            failure = classify(error, pacer.provider, pacer.ledger.clock())
            delay = policy.delay(attempt, failure.delay, pacer.retry_jitter) if failure.retryable else 0
            # Persist the cooldown even when this invocation cannot retry, so
            # other jobs sharing the scope respect the provider's instruction.
            pacer.observe(ticket, headers=error_headers(error), cooldown=delay)
            if not failure.retryable:
                emit_index('request_stopped', reason='Provider or request configuration needs attention.', status=failure.status)
                raise
            if attempt >= policy.max_attempts or pacer.ledger.clock() + delay >= deadline:
                log.warning('%s embedding retries stopped after %d attempt(s): attempt/time budget reached.', pacer.provider, attempt)
                emit_index('request_stopped', reason='Retry attempt/time budget reached.', status=failure.status)
                raise
            log.warning(
                '%s embedding request failed (%s); retry %d/%d after at least %.2fs, subject to quota pacing.',
                pacer.provider, failure.status or 'transport', attempt + 1, policy.max_attempts, delay
            )
            emit_index('retry', attempt=attempt + 1, maximum=policy.max_attempts, seconds=delay, status=failure.status)
            try:
                batch, ticket = reserve(deadline)
            except QuotaWaitExpired:
                log.warning('%s embedding retries stopped: quota wait exceeds the remaining retry time budget.', pacer.provider)
                emit_index('request_stopped', reason='Quota wait exceeds the remaining retry time budget.', status=failure.status)
                raise error from None
            attempt += 1
        else:
            return batch, response, ticket
