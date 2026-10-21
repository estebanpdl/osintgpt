'''Generation quotas and HTTP retries using the shared reservation ledger.'''

import random
from dataclasses import dataclass, field

from osintgpt.rate_settings import RateLimits
from .embedding_retry import RetryPolicy, request_with_retries
from .rate_ledger import RateLedger, quota_scope

GENERATION_LEDGER = RateLedger()


@dataclass(frozen=True)
class GenerationLimits:
    rates: RateLimits = field(default_factory=RateLimits)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    state_path: str = ''


class GenerationRequestError(RuntimeError):
    pass


class GenerationRequests:
    def __init__(self, limits=None, *, ledger=None):
        self.limits = limits or GenerationLimits()
        self.ledger = ledger or (RateLedger(self.limits.state_path) if self.limits.state_path else GENERATION_LEDGER)
        self.pacers = {}

    def for_provider(self, provider):
        key = id(provider)
        if key not in self.pacers:
            self.pacers[key] = GenerationPacer(provider, self.limits, ledger=self.ledger)
        return self.pacers[key]


class GenerationPacer:
    def __init__(self, provider, limits, *, ledger):
        self.provider = provider.provider
        self.configured, self.retry_policy = limits.rates, limits.retry
        client = provider.client
        self.scope = quota_scope(self.provider, provider.model, str(client.base_url),
                                 str(client.api_key or ''), self.configured.scope)
        self.ledger = ledger
        self.retry_jitter = random.uniform

    def observe(self, ticket, **kwargs):
        self.ledger.observe(self.scope, ticket, self.configured, **kwargs)

    def request(self, operation, tokens, *, before, failed, emit, check_cancel):
        def reserve(deadline=None):
            check_cancel()
            ticket = self.ledger.reserve(self.scope, tokens, self.configured, deadline=deadline,
                                         emit=emit, check_cancel=check_cancel)
            return [None], ticket

        def attempt(_):
            check_cancel()
            before()
            try:
                return operation()
            except BaseException as error:
                failed(error)
                raise

        batch, ticket = reserve()
        try:
            _, response, _ = request_with_retries(self, batch, ticket, attempt, reserve,
                                                 emit=emit, workload='generation')
        except Exception as error:
            raise GenerationRequestError(str(error)) from error
        return response
