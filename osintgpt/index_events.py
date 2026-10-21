'''Run-local indexing observations, isolated between app sessions and threads.'''

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class IndexEvent:
    kind: str
    data: dict


EventSink = Callable[[IndexEvent], None]


class EventChannel:
    def __init__(self, name):
        self.observer = ContextVar(name, default=None)

    @contextmanager
    def observe(self, callback):
        token = self.observer.set(callback)
        try:
            yield
        finally:
            self.observer.reset(token)

    def emit(self, kind, **data):
        callback = self.observer.get()
        if callback is not None:
            try:
                callback(IndexEvent(kind, data))
            except Exception:
                # Display errors cannot invalidate saved work; cancellation still propagates.
                logging.getLogger('osintgpt.progress').warning('Progress display could not be updated.')


_index = EventChannel('index_observer')
observe_index, emit_index = _index.observe, _index.emit
