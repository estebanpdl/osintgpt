# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: usage.py
# Description: What each provider call consumed. Tokens are the record; money
#   is a reading taken over them, and an approximate one.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Dict, List, Optional

# import osintgpt pricing
from osintgpt.pricing import estimate_cost


def format_usd(value: float) -> str:
    return f'${value:.6f}'.rstrip('0').rstrip('.')


# Usage class
@dataclass(frozen=True)
class Usage:
    '''
    One provider call's consumption.
    '''
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # False where the provider runs on infrastructure the operator owns. Its
    # cost is zero, which is a different statement from unknown.
    billable: bool = True
    # False where the backend reports no counts at all — an in-process encoder
    # returns vectors, not a usage block. Keeps a real zero apart from silence.
    counted: bool = True

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    # money, approximately
    @property
    def estimated_cost(self) -> Optional[float]:
        '''
        Estimated USD for this call.

        Returns:
            Optional[float]: 0.0 for a provider that cannot bill, None when \
                the model carries no price, otherwise an estimate that ignores \
                cached input, batch rates and free allowances.
        '''
        if not self.billable:
            return 0.0

        cost = estimate_cost(self.model, self.input_tokens)
        if cost is None:
            return None

        if self.output_tokens:
            output = estimate_cost(self.model, self.output_tokens, 'output')
            if output is None:
                return None
            cost += output

        return cost


class CostLimitReached(BaseException):
    '''
    A hard run stop that ordinary fail-soft exception boundaries cannot hide.
    '''

    def __init__(self, message: str, ceiling_usd: float) -> None:
        super().__init__(message)
        self.ceiling_usd = ceiling_usd
        self.completed = None
        self.remaining = None

    def with_index_progress(self, completed: int, remaining: int):
        self.completed = completed
        self.remaining = remaining

        return self

    def __str__(self) -> str:
        message = super().__str__()
        if self.remaining is None:
            return message

        return (
            f'{message} {self.completed} documents indexed; '
            f'{self.remaining} remain. Completed documents were saved; '
            're-run index to continue, increasing the ceiling if one '
            'remaining call exceeds it.'
        )


# UsageRecorder class
@dataclass
class UsageRecorder:
    '''
    Collects what a run consumed. One per project session; providers append to
    it when they are given one.
    '''
    records: List[Usage] = field(default_factory=list)
    cost_ceiling_usd: Optional[float] = None

    def record(self, usage: Usage) -> None:
        self.records.append(usage)
        self._enforce(usage)

    def _enforce(self, latest: Usage) -> None:
        ceiling = self.cost_ceiling_usd
        if ceiling is None or not latest.billable:
            return

        if not latest.counted:
            raise CostLimitReached(
                f'Cost ceiling {format_usd(ceiling)} cannot be enforced '
                f'because '
                f'{latest.provider} returned no usage for {latest.model}.',
                ceiling
            )

        if latest.estimated_cost is None:
            raise CostLimitReached(
                f'Cost ceiling {format_usd(ceiling)} cannot be enforced '
                f'because '
                f'{latest.model} has no verified price.',
                ceiling
            )

        if self.estimated_billable_cost > ceiling:
            raise CostLimitReached(
                f'Cost ceiling {format_usd(ceiling)} exceeded after a '
                f'provider '
                f'call; estimated run spend is '
                f'~{format_usd(self.estimated_billable_cost)}.',
                ceiling
            )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.records)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    # calls whose model carries no price
    @property
    def unpriced_calls(self) -> int:
        '''
        Returns:
            int: Billable calls whose cost could not be estimated. Reported \
                beside the total so a partial sum is never read as complete.
        '''
        return sum(
            1 for u in self.records
            if u.billable and u.estimated_cost is None
        )

    # calls the backend reported no counts for
    @property
    def uncounted_calls(self) -> int:
        return sum(1 for u in self.records if not u.counted)

    @property
    def uncounted_billable_calls(self) -> int:
        return sum(1 for u in self.records if u.billable and not u.counted)

    @property
    def billable_calls(self) -> int:
        return sum(1 for u in self.records if u.billable)

    @property
    def estimated_billable_cost(self) -> float:
        return sum(
            u.estimated_cost for u in self.records
            if u.billable and u.counted and u.estimated_cost is not None
        )

    @property
    def cost_complete(self) -> bool:
        return not self.unpriced_calls and not self.uncounted_billable_calls

    @property
    def estimated_cost(self) -> float:
        '''
        Returns:
            float: Summed estimate over the calls that could be priced. Read \
                it with `unpriced_calls`, not on its own.
        '''
        return sum(
            u.estimated_cost for u in self.records
            if u.estimated_cost is not None
        )

    # per-model breakdown
    @property
    def by_model(self) -> Dict[str, int]:
        '''
        Returns:
            Dict[str, int]: Total tokens per model, which is what tells an \
                operator where a run actually went.
        '''
        totals: Dict[str, int] = {}
        for usage in self.records:
            totals[usage.model] = totals.get(usage.model, 0) + usage.total_tokens

        return totals

    # operator-facing summary
    @property
    def summary(self) -> str:
        '''
        Returns:
            str: One line naming tokens first and money second, with the \
                estimate's gaps stated rather than hidden.
        '''
        if not self.records:
            return 'no provider calls'

        parts = [
            f'{self.calls} call{"s" if self.calls != 1 else ""}',
            f'{self.total_tokens:,} tokens'
        ]
        if self.unpriced_calls:
            parts.append(
                f'~${self.estimated_cost:.4f} '
                f'({self.unpriced_calls} unpriced)'
            )
        else:
            parts.append(f'~${self.estimated_cost:.4f}')
        if self.uncounted_calls:
            parts.append(f'{self.uncounted_calls} not counted')

        return ', '.join(parts)
