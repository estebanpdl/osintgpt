'''Token-aware batches and quota reservations for paid embedding providers.'''

import hashlib
import json
import random
from functools import lru_cache

from osintgpt.rate_settings import RateLimits
from .rate_ledger import PROCESS_LEDGER, RateLedger, TokenBudgetExceeded, lower_limit
from .embedding_retry import RetryPolicy, request_with_retries

# API request ceilings, not account TPM tiers. Unknown Voyage models use the
# smallest documented aggregate ceiling. Operators can choose smaller batches.
OPENAI_MODELS = {'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'}
VOYAGE_BATCH_TOKENS = {
    'voyage-4-lite': 1_000_000, 'voyage-3.5-lite': 1_000_000,
    'voyage-4': 320_000, 'voyage-3.5': 320_000, 'voyage-2': 320_000
}


@lru_cache(maxsize=1)
def openai_encoding():
    import tiktoken
    return tiktoken.get_encoding('cl100k_base')


def token_count(provider, model, text):
    if provider == 'openai' and model in OPENAI_MODELS:
        # Treat special-looking strings as ordinary source text.
        return len(openai_encoding().encode(text, disallowed_special=()))
    # An estimate, never billing usage. Includes allowance for tokenizer
    # framing; counts bytes rather than assuming English characters/token.
    return len(text.encode('utf-8')) + 16


def quota_scope(provider, model, endpoint, credential, explicit=''):
    account = {'scope': explicit} if explicit else {
        'credential': hashlib.sha256(credential.encode()).hexdigest(), 'model': model
    }
    material = {'provider': provider, 'endpoint': endpoint.rstrip('/'), **account}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


class EmbeddingPacer:
    def __init__(self, provider, model, endpoint, credential, limits=None,
                 state_path='', *, ledger=None, discover=False, retry_policy=None):
        self.provider, self.model = provider, model
        self.configured = limits or RateLimits()
        self.scope = quota_scope(provider, model, endpoint, credential, self.configured.scope)
        self.ledger = ledger or (RateLedger(state_path) if state_path else PROCESS_LEDGER)
        self.discover = discover
        self.retry_policy = retry_policy or RetryPolicy()
        self.retry_jitter = random.uniform

    def budget(self):
        _, tpm = self.ledger.limits(self.scope, self.configured)
        ceiling = (
            300_000 if self.provider == 'openai' else
            VOYAGE_BATCH_TOKENS.get(self.model, 120_000) if self.provider == 'voyage' else
            100_000  # Local batch target; Gemini's account quota is configured separately.
        )
        return lower_limit(ceiling, self.configured.batch_tokens, tpm)

    def batches(self, texts, batch_size, *, render=lambda text: text, start=0, deadline=None):
        '''Yield (input batch, reservation) in original order, without truncation.

        Re-evaluate the token budget after every response because provider
        headers, another process, or a tier change can update shared limits.
        '''
        position = start
        maximum = min(batch_size, {'openai': 2048, 'voyage': 1000, 'gemini': 100}[self.provider])
        if maximum < 1:
            raise ValueError('batch_size must be positive')
        while position < len(texts):
            size = maximum
            if self.discover:
                rpm, tpm = self.ledger.limits(self.scope, self.configured)
                if not rpm or not tpm:
                    # A normal one-input embedding request supplies headers;
                    # this avoids starting an unknown tier with a large batch.
                    size = 1
            budget = self.budget()
            count, end = 0, position
            for text in texts[position:position + size]:
                if not text:
                    raise ValueError(f'Embedding input {end} is empty')
                tokens = token_count(self.provider, self.model, render(text))
                if self.provider == 'openai' and self.model in OPENAI_MODELS and tokens > 8192:
                    raise ValueError(f'Embedding input {end} exceeds the OpenAI 8192-token input limit')
                if tokens > budget:
                    if end > position:
                        break
                    raise TokenBudgetExceeded(
                        f'Embedding input {end} needs {tokens} estimated tokens, '
                        f'exceeding the {budget}-token batch/TPM budget. '
                        'Reduce input size or raise the configured budget within your account limits.'
                    )
                if count + tokens > budget:
                    break
                count += tokens
                end += 1
            try:
                ticket = self.ledger.reserve(self.scope, count, self.configured, deadline=deadline)
            except TokenBudgetExceeded:
                # Another process learned a lower TPM between sizing and
                # reserving. Repartition locally; no HTTP call has been made.
                continue
            self.discover = False
            yield texts[position:end], ticket
            position = end

    def responses(self, texts, batch_size, request, *, render=lambda text: text):
        '''Yield successful batches only; callers validate and checkpoint each.

        A retry can shrink the unfinished prefix if an error reports a lower
        TPM. Position advances only after the caller receives that success.
        '''
        position = 0
        while position < len(texts):
            batch, ticket = next(self.batches(texts, batch_size, render=render, start=position))
            maximum = len(batch)
            def reserve(deadline):
                return next(self.batches(texts, maximum, render=render, start=position, deadline=deadline))
            batch, response, ticket = request_with_retries(self, batch, ticket, request, reserve)
            yield batch, response, ticket
            position += len(batch)

    def observe(self, ticket, *, tokens=None, headers=None, cooldown=0):
        self.ledger.observe(self.scope, ticket, self.configured, tokens=tokens, headers=headers, cooldown=cooldown)
