'''Run-local graph observations and cooperative cancellation boundaries.'''

from contextlib import contextmanager
from contextvars import ContextVar

from osintgpt.index_events import EventChannel

_channel = EventChannel('graph_observer')
emit_graph = _channel.emit
_cancel = ContextVar('graph_cancel', default=None)


class GraphCancelled(BaseException):
    pass


def check_cancelled():
    callback = _cancel.get()
    if callback is not None and callback():
        raise GraphCancelled('Graph build cancelled; saved work can be resumed.')


@contextmanager
def observe_graph(callback=None, should_cancel=None):
    token = _cancel.set(should_cancel)
    try:
        with _channel.observe(callback):
            yield
    finally:
        _cancel.reset(token)
