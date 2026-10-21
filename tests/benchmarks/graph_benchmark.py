'''Shared role attribution and configuration for offline-validated benchmark runners.'''

from contextlib import contextmanager
from dataclasses import asdict

from osintgpt.graph.recipes import extraction_recipe
from osintgpt.llm.usage import UsageRecorder


def identity(provider):
    if provider is None:
        return None
    named = getattr(provider, 'provider', '')
    return {'provider': named if isinstance(named, str) else '', 'model': getattr(provider, 'model', ''),
            'implementation': f'{type(provider).__module__}.{type(provider).__qualname__}'}


def model_roles(answerer=None, extractor=None, embedder=None):
    return {'answering': identity(answerer), 'extraction': identity(extractor), 'embedding': identity(embedder),
            'extraction_selection': 'inherit_answering' if extractor is answerer and answerer is not None else 'independent'}


def usage_summary(records):
    usage = UsageRecorder(records=list(records))
    return {'calls': usage.calls, 'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens,
            'tokens': usage.total_tokens, 'estimated_cost_usd': usage.estimated_billable_cost if usage.cost_complete else None,
            'known_cost_usd': usage.estimated_billable_cost, 'cost_complete': usage.cost_complete,
            'uncounted_calls': usage.uncounted_calls, 'records': [asdict(record) for record in usage.records]}


class RoleUsage:
    '''Slice sequential service calls without creating independent spending ceilings.'''
    def __init__(self, recorder, **providers):
        self.recorder = recorder
        self.records = {role: [] for role in ('extraction', 'answering', 'embedding')}
        self.available = {}
        for role, provider in providers.items():
            owner = getattr(provider, 'recorder', None)
            if provider is not None and owner is not recorder and (owner is not None or recorder.cost_ceiling_usd is not None):
                raise ValueError('Benchmark providers must share one usage recorder and spending ceiling.')
            self.available[role] = owner is recorder

    @contextmanager
    def stage(self, role):
        before = len(self.recorder)
        try:
            yield
        finally:
            self.records[role].extend(self.recorder.records[before:])

    def report(self):
        return {role: {**usage_summary(records), 'recorder_attached': self.available.get(role, False)}
                for role, records in self.records.items()}


def extraction_metadata(provider, options=None):
    return {'models': model_roles(extractor=provider), 'extraction_recipe': extraction_recipe(provider, options)}
