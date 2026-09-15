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


_observer = ContextVar('index_observer', default=None)
EventSink = Callable[[IndexEvent], None]


@contextmanager
def observe_index(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


def emit_index(kind, **data):
    callback = _observer.get()
    if callback is not None:
        try:
            callback(IndexEvent(kind, data))
        except Exception:
            # A broken display must not turn a saved response into a retry.
            # Cancellation and CostLimitReached are BaseExceptions and pass on.
            logging.getLogger('osintgpt.indexing').warning('Index progress display could not be updated.')
