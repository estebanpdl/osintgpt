'''Work counters and recorded usage for graph extraction runs.'''

from dataclasses import asdict, dataclass, fields, replace

from osintgpt.llm.usage import UsageRecorder


def usage_messages(usage):
    messages = []
    if usage.get('awaiting_usage_requests'):
        messages.append(f"Awaiting usage for {usage['awaiting_usage_requests']} active requests.")
    if usage.get('unavailable_usage_requests'):
        messages.append(f"Usage unavailable for {usage['unavailable_usage_requests']} requests; their cost is unknown, not zero.")
    if usage.get('unpriced_calls'):
        messages.append(f"Model pricing unavailable for {usage['unpriced_calls']} calls; recorded token usage is shown separately.")
    if not usage.get('complete', True) and not messages:
        messages.append('Earlier usage records are incomplete; inspect last-run details.')
    return messages


@dataclass
class ExtractionCounts:
    windows_total: int = 0
    windows_completed: int = 0
    windows_reused: int = 0
    responses_saved: int = 0
    responses_reused: int = 0
    responses_recovered: int = 0
    context_restarts: int = 0
    structured_corrections: int = 0
    completion_requests: int = 0
    structured_correction_requests: int = 0
    no_progress_stops: int = 0
    retries: int = 0
    request_controls_unenforced: int = 0
    requests_attempted: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    requests_interrupted: int = 0
    invalid_responses: int = 0
    candidate_claims: int = 0
    rejected_claims: int = 0
    rejected_entities: int = 0
    repairs_attempted: int = 0
    repaired_claims: int = 0
    repaired_entities: int = 0
    field_warnings: int = 0
    entity_lookups: int = 0
    entity_lookups_reused: int = 0
    identity_unresolved: int = 0
    interactive_lookup_unavailable: int = 0
    output_limits_unenforced: int = 0
    structured_output_unenforced: int = 0
    completion_status_unavailable: int = 0


@dataclass
class BuildCounts(ExtractionCounts):
    documents_total: int = 0
    documents_completed: int = 0
    documents_failed: int = 0
    documents_skipped: int = 0
    documents_removed: int = 0
    documents_changed: int = 0
    checkpoints_discarded: int = 0
    records_loaded: int = 0
    images_skipped: int = 0

    def including(self, current):
        return replace(self, **{
            item.name: getattr(self, item.name) + getattr(current, item.name)
            for item in fields(ExtractionCounts)
        })

    def to_dict(self):
        return asdict(self)


def recorded_usage(generator, counts, offset=0):
    recorder = getattr(generator, 'recorder', None)
    records = list(recorder.records[offset:]) if recorder is not None else []
    usage = UsageRecorder(records=records)
    missing = max(counts.requests_attempted - usage.calls, 0)
    counted = any(record.counted for record in records)
    priced = any(
        record.billable and record.counted and record.estimated_cost is not None
        for record in records
    )
    empty = counts.requests_attempted == 0
    pending = max(counts.requests_attempted - counts.requests_succeeded - counts.requests_failed - counts.requests_interrupted, 0)
    tokens_complete = missing == 0 and usage.uncounted_calls == 0
    cost_complete = missing == 0 and usage.cost_complete
    return {
        'calls': usage.calls,
        'recorded_responses': usage.calls,
        'missing_usage_requests': missing,
        'awaiting_usage_requests': min(pending, missing),
        'unavailable_usage_requests': max(missing - pending, 0) + usage.uncounted_calls,
        'input_tokens': usage.input_tokens if counted or empty else None,
        'output_tokens': usage.output_tokens if counted or empty else None,
        'total_tokens': usage.total_tokens if counted or empty else None,
        'estimated_cost_usd': usage.estimated_billable_cost if priced else None,
        'complete': tokens_complete and cost_complete,
        'tokens_complete': tokens_complete,
        'cost_complete': cost_complete,
        'unpriced_calls': usage.unpriced_calls,
        'uncounted_calls': usage.uncounted_calls,
        'billable_calls': usage.billable_calls,
        'by_model': usage.by_model,
        'request_scope': 'generation_provider_calls' if counts.request_controls_unenforced else 'http_attempts',
        'transport_attempts': None if counts.request_controls_unenforced else counts.requests_attempted,
        'responses_reused': counts.responses_reused,
        'ceiling_usd': getattr(recorder, 'cost_ceiling_usd', None),
    }
