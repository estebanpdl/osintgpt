'''Operator-owned embedding quotas, independent of vector configuration.'''

from dataclasses import dataclass, fields
from typing import Optional

PAID_EMBEDDING_PROVIDERS = ('openai', 'gemini', 'voyage')


@dataclass(frozen=True, kw_only=True)
class EmbeddingRateSettings:
    # None inherits/defaults. Zero disables that operator ceiling (provider
    # headers still apply). Provider-specific fields survive model switches.
    openai_embedding_rpm: Optional[int] = None
    openai_embedding_tpm: Optional[int] = None
    openai_embedding_batch_tokens: Optional[int] = None
    openai_embedding_quota_scope: str = ''
    gemini_embedding_rpm: Optional[int] = None
    gemini_embedding_tpm: Optional[int] = None
    gemini_embedding_batch_tokens: Optional[int] = None
    gemini_embedding_quota_scope: str = ''
    voyage_embedding_rpm: Optional[int] = None
    voyage_embedding_tpm: Optional[int] = None
    voyage_embedding_batch_tokens: Optional[int] = None
    voyage_embedding_quota_scope: str = ''

    def __post_init__(self):
        for name in RATE_FIELDS:
            value = getattr(self, name)
            if name.endswith('_scope'):
                if not isinstance(value, str):
                    raise ValueError(f'{name} must be text')
            elif value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a nonnegative whole number or unset')


RATE_FIELDS = tuple(field.name for field in fields(EmbeddingRateSettings))
RATE_ENV_VARS = {name: f'OSINTGPT_{name.upper()}' for name in RATE_FIELDS}


@dataclass(frozen=True)
class RateLimits:
    rpm: int = 0
    tpm: int = 0
    batch_tokens: int = 0
    scope: str = ''

    def __post_init__(self):
        for name in ('rpm', 'tpm', 'batch_tokens'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative whole number')


def limits_for(settings, provider: str) -> RateLimits:
    return RateLimits(**{
        key: getattr(settings, f'{provider}_embedding_{suffix}') or default
        for key, suffix, default in (
            ('rpm', 'rpm', 0), ('tpm', 'tpm', 0),
            ('batch_tokens', 'batch_tokens', 0), ('scope', 'quota_scope', '')
        )
    })
