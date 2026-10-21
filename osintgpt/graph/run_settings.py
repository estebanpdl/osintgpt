'''Saved app build preferences translated into the existing extraction and request contracts.'''

import math
from dataclasses import asdict, dataclass

from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.generation_pacing import GenerationLimits
from osintgpt.projects.toml_io import read_toml, write_toml
from osintgpt.rate_settings import RateLimits

from .windowing import ExtractionOptions


@dataclass(frozen=True)
class GraphRunSettings:
    context_budget: int = 32768
    repair_quotes: bool = True
    rpm: int = 0
    tpm: int = 0
    retry_attempts: int = 6
    retry_seconds: float = 300.0
    cost_ceiling_usd: float | None = None

    def __post_init__(self):
        for name in ('context_budget', 'rpm', 'tpm', 'retry_attempts'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be a whole number.')
        if type(self.repair_quotes) is not bool:
            raise ValueError('repair_quotes must be true or false.')
        if self.cost_ceiling_usd is not None and (isinstance(self.cost_ceiling_usd, bool)
                or not isinstance(self.cost_ceiling_usd, (int, float))
                or not math.isfinite(self.cost_ceiling_usd) or self.cost_ceiling_usd <= 0):
            raise ValueError('Cost ceiling must be a positive finite number or unset.')
        self.extraction()
        self.limits()

    def extraction(self):
        return ExtractionOptions(context_budget=self.context_budget, repair_quotes=self.repair_quotes)

    def limits(self):
        return GenerationLimits(RateLimits(rpm=self.rpm, tpm=self.tpm),
                                RetryPolicy(max_attempts=self.retry_attempts, max_elapsed=self.retry_seconds))

    @classmethod
    def load(cls, project):
        data = read_toml(project.paths.root / 'graph-build.toml')
        return cls(**{name: value for name, value in data.items() if name in cls.__dataclass_fields__})

    def save(self, project):
        write_toml(project.paths.root / 'graph-build.toml',
                   {name: value for name, value in asdict(self).items() if value is not None})
