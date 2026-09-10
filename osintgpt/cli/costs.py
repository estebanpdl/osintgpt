'''Shared cost recording, output, and hard-stop handling for CLI runs.'''

from typing import Dict

from osintgpt.llm.usage import (
    CostLimitReached,
    UsageRecorder,
    format_usd
)

from .output import fail


def recorder_for(settings) -> UsageRecorder:
    return UsageRecorder(cost_ceiling_usd=settings.cost_ceiling_usd)


def usage_data(recorder: UsageRecorder) -> Dict[str, object]:
    known_cost = recorder.estimated_billable_cost
    priced = any(
        usage.billable and usage.counted
        and usage.estimated_cost is not None
        for usage in recorder
    )

    return {
        'calls': recorder.calls,
        'billable_calls': recorder.billable_calls,
        'input_tokens': recorder.input_tokens,
        'output_tokens': recorder.output_tokens,
        'total_tokens': recorder.total_tokens,
        'estimated_cost_usd': known_cost if priced else None,
        'complete': recorder.cost_complete,
        'unpriced_calls': recorder.unpriced_calls,
        'uncounted_calls': recorder.uncounted_billable_calls,
        'by_model': recorder.by_model,
        'ceiling_usd': recorder.cost_ceiling_usd
    }


def add_usage(data: Dict[str, object], recorder: UsageRecorder) -> None:
    data['usage'] = usage_data(recorder)


def render_usage(target, recorder: UsageRecorder) -> None:
    if not recorder.billable_calls:
        return

    gaps = recorder.unpriced_calls + recorder.uncounted_billable_calls
    known = recorder.estimated_billable_cost
    priced = any(
        usage.billable and usage.counted
        and usage.estimated_cost is not None
        for usage in recorder
    )
    if not gaps:
        cost = f'~{format_usd(known)}'
    elif priced:
        cost = f'at least ~{format_usd(known)}; estimate incomplete'
    else:
        cost = 'cost unknown'

    details = []
    if recorder.unpriced_calls:
        details.append(f'{recorder.unpriced_calls} unpriced')
    if recorder.uncounted_billable_calls:
        details.append(
            f'{recorder.uncounted_billable_calls} not counted'
        )
    suffix = f' ({", ".join(details)})' if details else ''
    target.print(
        f'Usage: {recorder.calls} call'
        f'{"s" if recorder.calls != 1 else ""}, '
        f'{recorder.total_tokens:,} tokens, {cost}{suffix}'
    )


def fail_for_cost(
    error: CostLimitReached,
    recorder: UsageRecorder,
    json_output: bool
) -> None:
    details = {'usage': usage_data(recorder)}
    if error.remaining is not None:
        details.update({
            'indexed_documents': error.completed,
            'remaining_documents': error.remaining
        })
    fail(str(error), json_output, details)
