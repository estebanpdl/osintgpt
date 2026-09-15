'''Atomic quota reservations shared by embedding jobs on one machine.'''

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from osintgpt.index_events import emit_index

from .rate_headers import quota_headers

log = logging.getLogger('osintgpt.indexing')
WINDOW = 60.0
HEADER_TTL = 300.0
SCHEMA = '''
CREATE TABLE IF NOT EXISTS embedding_quotas (
    scope TEXT PRIMARY KEY, limits TEXT NOT NULL DEFAULT '{}',
    next_at REAL NOT NULL DEFAULT 0, remaining TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS embedding_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, at REAL NOT NULL,
    tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS embedding_requests_scope ON embedding_requests(scope, at);
'''


def lower_limit(*values):
    positive = [value for value in values if value and value > 0]
    return min(positive) if positive else 0


class TokenBudgetExceeded(ValueError):
    pass


class QuotaWaitExpired(TimeoutError):
    pass


class RateLedger:
    '''A sliding minute window plus spacing between reservations.

    Reservations precede HTTP calls. Failed calls remain charged to the local
    quota window. Cancellation while waiting reserves nothing. No source text,
    API keys, URLs or user-supplied scope labels are written here.
    '''

    def __init__(self, path='', *, clock=time.time, sleep=time.sleep):
        self.path = str(path)
        self.clock, self.sleep = clock, sleep
        self.lock = threading.RLock()
        self.memory = None
        if not self.path:
            self.memory = sqlite3.connect(':memory:', check_same_thread=False)

    @contextmanager
    def transaction(self):
        with self.lock:
            if self.path:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            connection = self.memory or sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
            try:
                connection.executescript(SCHEMA)
                connection.execute('BEGIN IMMEDIATE')
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                if self.memory is None:
                    connection.close()

    def _state(self, connection, scope):
        connection.execute('INSERT OR IGNORE INTO embedding_quotas(scope) VALUES (?)', (scope,))
        return connection.execute('SELECT * FROM embedding_quotas WHERE scope = ?', (scope,)).fetchone()

    def _limits(self, state, configured, now):
        hints = {
            key: hint['value'] for key, hint in json.loads(state['limits']).items()
            if now - hint['at'] < HEADER_TTL
        }
        return (
            lower_limit(configured.rpm, hints.get('requests')),
            lower_limit(configured.tpm, hints.get('tokens'), hints.get('project-tokens'))
        )

    def limits(self, scope, configured):
        with self.transaction() as connection:
            return self._limits(self._state(connection, scope), configured, self.clock())

    def reserve(self, scope, tokens, configured, *, deadline=None):
        if type(tokens) is not int or tokens < 1:
            raise ValueError('A quota reservation needs a positive token count')
        announced = False
        while True:
            with self.transaction() as connection:
                now = self.clock()
                if deadline is not None and now >= deadline:
                    raise QuotaWaitExpired('Embedding retry deadline reached before quota reservation')
                state = self._state(connection, scope)
                rpm, tpm = self._limits(state, configured, now)
                if tpm and tokens > tpm:
                    raise TokenBudgetExceeded(
                        f'One embedding input/batch needs {tokens} estimated tokens, '
                        f'exceeding the current {tpm} TPM budget. Reduce input size '
                        'or configure a larger quota only if your account supports it.'
                    )
                connection.execute('DELETE FROM embedding_requests WHERE at <= ?', (now - WINDOW,))
                events = connection.execute(
                    'SELECT at, tokens FROM embedding_requests WHERE scope = ? ORDER BY at, id',
                    (scope,)
                ).fetchall()
                wait = max(0.0, state['next_at'] - now)
                count, used = len(events), sum(event['tokens'] for event in events)
                for event in events:
                    if (not rpm or count < rpm) and (not tpm or used + tokens <= tpm):
                        break
                    wait = max(wait, event['at'] + WINDOW - now)
                    count -= 1
                    used -= event['tokens']
                remaining = json.loads(state['remaining'])
                for axis, hint in remaining.items():
                    needed = 1 if axis == 'requests' else tokens
                    if hint['until'] > now and hint['left'] < needed:
                        wait = max(wait, hint['until'] - now)
                if deadline is not None and now + wait >= deadline:
                    raise QuotaWaitExpired('Embedding quota wait exceeds the retry deadline')
                if wait <= 0:
                    cursor = connection.execute(
                        'INSERT INTO embedding_requests(scope, at, tokens) VALUES (?, ?, ?)',
                        (scope, now, tokens)
                    )
                    spacing = max(WINDOW / rpm if rpm else 0, WINDOW * tokens / tpm if tpm else 0)
                    for axis, hint in remaining.items():
                        if hint['until'] > now:
                            hint['left'] = max(0, hint['left'] - (1 if axis == 'requests' else tokens))
                    connection.execute(
                        'UPDATE embedding_quotas SET next_at = ?, remaining = ? WHERE scope = ?',
                        (now + spacing, json.dumps(remaining), scope)
                    )
                    return cursor.lastrowid
            if not announced:
                log.info('Embedding quota pacing: waiting %.2f seconds before the next request.', wait)
                announced = True
            # Short sleeps keep Ctrl+C/Streamlit cancellation responsive. The
            # transaction has closed; another process can use/update its quota.
            emit_index('waiting', seconds=wait)
            self.sleep(min(wait, 1.0))

    def observe(self, scope, ticket, configured, *, tokens=None, headers=None, cooldown=0):
        now = self.clock()
        limits, hints = quota_headers(headers, now)
        with self.transaction() as connection:
            state = self._state(connection, scope)
            event = connection.execute(
                'SELECT at, tokens FROM embedding_requests WHERE id = ? AND scope = ?',
                (ticket, scope)
            ).fetchone()
            remaining = json.loads(state['remaining'])
            extra = max(0, tokens - event['tokens']) if event and type(tokens) is int else 0
            if extra:
                connection.execute('UPDATE embedding_requests SET tokens = tokens + ? WHERE id = ?', (extra, ticket))
                for axis, hint in remaining.items():
                    if axis != 'requests' and hint['until'] > now:
                        hint['left'] = max(0, hint['left'] - extra)
            later = connection.execute(
                'SELECT COUNT(*) AS requests, COALESCE(SUM(tokens), 0) AS tokens '
                'FROM embedding_requests WHERE scope = ? AND id > ? AND at > ?',
                (scope, ticket, now - WINDOW)
            ).fetchone()
            for axis, hint in hints.items():
                # A response snapshot can predate another in-flight request,
                # even when it is the first set of headers we have seen.
                hint['left'] = max(0, hint['left'] - later['requests' if axis == 'requests' else 'tokens'])
                previous = remaining.get(axis)
                if previous and previous['until'] > now:
                    hint = {
                        'left': min(previous['left'], hint['left']),
                        'until': max(previous['until'], hint['until'])
                    }
                remaining[axis] = hint
            learned = json.loads(state['limits'])
            for axis, value in limits.items():
                previous = learned.get(axis)
                if previous and previous.get('ticket', 0) > ticket:
                    # An older in-flight request must not raise a ceiling a
                    # newer response has already lowered.
                    value = min(value, previous['value'])
                learned[axis] = {
                    'value': value, 'at': now,
                    'ticket': max(ticket, previous.get('ticket', 0) if previous else 0)
                }
            encoded_limits = json.dumps(learned)
            effective_rpm, effective_tpm = self._limits({'limits': encoded_limits}, configured, now)
            next_at = max(state['next_at'], now + cooldown)
            if event:
                spacing = max(
                    WINDOW / effective_rpm if effective_rpm else 0,
                    WINDOW * (event['tokens'] + extra) / effective_tpm if effective_tpm else 0
                )
                next_at = max(next_at, event['at'] + spacing)
            connection.execute(
                'UPDATE embedding_quotas SET limits = ?, '
                'next_at = ?, remaining = ? WHERE scope = ?',
                (encoded_limits, next_at, json.dumps(remaining), scope)
            )


# Standalone library calls share an in-process ledger. CLI/app callers supply
# an explicit file under their selected osintgpt home for cross-process use.
PROCESS_LEDGER = RateLedger()
