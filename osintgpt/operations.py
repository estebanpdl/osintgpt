'''Settings, dependency ownership and accounting for a single operation.

Create one context per invocation. It contains no UI/session selection or global
client cache. Dependencies can only be used while the context is active.
'''

import logging
import math
from asyncio import CancelledError
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Optional

from osintgpt.config import Settings
from osintgpt.credentials import resolve_credentials
from osintgpt.index_events import IndexEvent, emit_index, observe_index
from osintgpt.llm import build_embedding_provider
from osintgpt.llm.base import EmbeddingProvider, EmbeddingPurpose
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder
from osintgpt.projects import Project, ProjectSettings, default_home, load_user_defaults
from osintgpt.projects.settings import GRAPH_MODEL_FIELDS, configuration_for, graph_model_settings
from osintgpt.vector_store import BaseVectorEngine, store_for


def lower_ceiling(*values):
    '''Only explicit nonempty ceilings constrain an operation; zero is set.'''
    present = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in present):
        raise ValueError('Cost ceilings must be finite, nonnegative numbers.')
    return min(present) if present else None


def usage_data(recorder: UsageRecorder) -> dict:
    '''A conservative, secret-free snapshot; partial known cost is separate.'''
    return {
        'calls': recorder.calls,
        'billable_calls': recorder.billable_calls,
        'input_tokens': recorder.input_tokens,
        'output_tokens': recorder.output_tokens,
        'total_tokens': recorder.total_tokens,
        'estimated_cost_usd': recorder.estimated_billable_cost if recorder.cost_complete else None,
        'known_cost_usd': recorder.estimated_billable_cost,
        'complete': recorder.cost_complete,
        'unpriced_calls': recorder.unpriced_calls,
        'uncounted_calls': recorder.uncounted_billable_calls,
        'ceiling_usd': recorder.cost_ceiling_usd,
    }


@dataclass(frozen=True)
class OperationOutcome:
    '''Safe status and usage, never a config, query, exception message or DSN.'''

    status: str
    usage: dict
    error: Optional[str] = None


class OperationContext:
    '''A single-use policy snapshot with lazy embedding and store dependencies.

    ``overrides`` contains invocation-level ProjectSettings fields. Explicit
    connection-config models win over project/default models unless overridden.
    A ceiling override can only tighten the resolved project/user ceiling.
    ``project=None`` supports fully injected standalone calls without home I/O.

    Injected resources remain caller-owned, including any provider recorder.
    Factories create operation-owned resources, closed once on every exit if
    they expose ``close``. Factory providers must emit the standard usage events;
    do not attach this context's recorder too, which would count responses twice.
    '''

    def __init__(
        self, project: Optional[Project], *, home: Optional[Path] = None,
        config: Optional[Settings] = None, defaults: Optional[ProjectSettings] = None,
        overrides: Optional[Mapping] = None, cost_ceiling_usd: Optional[float] = None,
        embedder: Optional[EmbeddingProvider] = None, store: Optional[BaseVectorEngine] = None,
        embedding_factory: Optional[Callable] = None, store_factory: Optional[Callable] = None,
        on_event: Optional[Callable[[IndexEvent], None]] = None,
    ):
        changes = dict(overrides or {})
        invocation_ceiling = changes.pop('cost_ceiling_usd', None)
        if project is None:
            effective = defaults if defaults is not None else ProjectSettings()
            base = config if config is not None else Settings()
        else:
            home = Path(home) if home is not None else default_home()
            defaults = defaults if defaults is not None else load_user_defaults(home)
            effective = project.effective_settings(defaults)
            base = config if config is not None else resolve_credentials(home)
        ceiling = lower_ceiling(effective.cost_ceiling_usd, invocation_ceiling, cost_ceiling_usd)
        graph_changes = {name: changes[name] for name in GRAPH_MODEL_FIELDS if name in changes}
        if graph_changes:
            changes.update(graph_model_settings(ProjectSettings.from_dict(graph_changes), effective))
        for field, target in (('embedding_model', 'openai_embedding_model'),
                              ('generation_model', 'openai_gpt_model')):
            if field in changes:
                base = base.with_overrides(**{target: changes[field]})
            elif getattr(base, target):
                changes[field] = getattr(base, target)
        self.settings = replace(effective, **changes, cost_ceiling_usd=ceiling)
        self.project = replace(project, settings=self.settings) if project is not None else None
        self.config = (configuration_for(self.settings, base, project.paths.store)
                       if project is not None else base)
        self.usage = UsageRecorder(cost_ceiling_usd=ceiling)
        self.unrecorded_calls = 0
        self.outcome = None
        self._embedder, self._store = embedder, store
        self._embedding_factory = embedding_factory or build_embedding_provider
        self._store_factory = store_factory or store_for
        self._owned = []
        self._started = False
        self._active = False
        self._on_event = on_event

    def _require_active(self):
        if not self._active:
            raise RuntimeError('Operation dependencies require an active context.')

    @property
    def embedder(self) -> EmbeddingProvider:
        self._require_active()
        if self._embedder is None:
            self._embedder = self._embedding_factory(
                self.settings.embedding_provider, self.config,
                model=self.settings.embedding_model or None,
            )
            self._owned.append(self._embedder)
        expected = self.settings.embedding_model
        if expected and self._embedder.model != expected:
            raise ValueError('Embedding provider model does not match the operation settings.')
        return self._embedder

    @property
    def store(self) -> BaseVectorEngine:
        self._require_active()
        if self._store is None:
            if self.project is None:
                raise ValueError('A project or an injected store is required.')
            self._store = self._store_factory(self.project, self.config)
            self._owned.append(self._store)
        return self._store

    def embed(self, texts, *, purpose=EmbeddingPurpose.DOCUMENT):
        '''Account for successful custom-provider calls even without usage events.'''
        self._require_active()
        if not texts:
            return []
        provider = self.embedder
        before = self.usage.calls
        vectors = provider.embed(texts, purpose=purpose)
        if self.usage.calls == before:
            self.unrecorded_calls += 1
            emit_index(
                'usage', provider=self.settings.embedding_provider, model=provider.model,
                billable=getattr(provider, 'billable', True), counted=False,
            )
        return vectors

    def _notify(self, kind, **data):
        if self._on_event is not None:
            try:
                self._on_event(IndexEvent(kind, data))
            except Exception:
                logging.getLogger('osintgpt.progress').warning('Operation display could not be updated.')

    def _record(self, event):
        if event.kind == 'usage':
            try:
                self.usage.record(Usage(**event.data))
            except BaseException:
                try:
                    self._notify('usage', **usage_data(self.usage))
                except BaseException:
                    pass  # Preserve the accounting stop, including its recorded usage.
                raise
            else:
                self._notify('usage', **usage_data(self.usage))

    def __enter__(self):
        if self._started:
            raise RuntimeError('Create a fresh context for each operation.')
        self._started = self._active = True
        self._observer = observe_index(self._record, forward=True)
        self._observer.__enter__()
        return self

    def __exit__(self, exc_type, error, traceback):
        self._active = False
        self._observer.__exit__(exc_type, error, traceback)
        cleanup_error = None
        for resource in reversed(self._owned):
            try:
                close = getattr(resource, 'close', None)
                if close is not None:
                    close()
            except BaseException as failure:
                if cleanup_error is None:
                    cleanup_error = failure
        self._owned.clear()
        failure = error if error is not None else cleanup_error
        status = ('cost_limit' if isinstance(failure, CostLimitReached) else
                  'cancelled' if isinstance(failure, (CancelledError, KeyboardInterrupt, SystemExit)) else
                  'failed' if failure is not None else 'completed')
        self.outcome = OperationOutcome(status, usage_data(self.usage),
                                        status if failure is not None else None)
        if isinstance(failure, CostLimitReached):
            failure.operation_outcome = self.outcome
        # The original hard stop must not be replaced by display/cleanup errors.
        try:
            self._notify('outcome', status=status, error=self.outcome.error,
                         usage=self.outcome.usage.copy())
        except BaseException:
            if failure is None:
                raise
        if error is None and cleanup_error is not None:
            raise cleanup_error
        return False
