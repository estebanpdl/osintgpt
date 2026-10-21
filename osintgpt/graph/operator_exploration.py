'''Session-owned operator queries with resumable work and revision-checked publication.'''

import sqlite3
from threading import RLock
from uuid import uuid4

from .exploration import ExplorationCursor, positive_depth
from .inspection import OperatorQuery, operator_query_result, inspect_assertion
from .retrieval import _base, claim_summary, select_endpoints, source_selection, text_argument
from .store import GraphStore
from .verify import verify_claims
from .verification_cache import EvidenceCache, VerificationChanged


class OperatorExploration(OperatorQuery):
    '''The read-only connection is serialized across Streamlit rerun threads.

    Transactions last one batch, so a paused query does not hold a SQLite reader
    lock. data_version detects writes between batches, including equal-size source
    replacement and identity review. A changed graph invalidates continuation.
    '''

    def __init__(self, project, entity, target=None, *, direction='connectivity', depth=1, refs=None):
        positive_depth(depth)
        entity = text_argument(entity, 'entity')
        target = text_argument(target, 'target') if target is not None else None
        refs = source_selection(refs)
        self.project, self.token = project, uuid4().hex
        self._lock, self._graph, self.cursor = RLock(), None, None
        self.verification = None
        self.state, self.batches = 'complete', 0
        super().__init__(_base(project, allow_disabled=True), {})
        self.result.update(delivery='operator_summaries_v1',
                           limits={'depth': depth, 'direction': direction},
                           source_scope={'mode': 'all' if refs is None else 'selected', 'refs': refs})
        if not self.result['built']:
            self.result = operator_query_result(self.result)
            return
        try:
            self.verification = EvidenceCache(project)
            path = project.paths.root / 'graph.sqlite'
            stat = path.stat()
            self._identity = stat.st_dev, stat.st_ino
            self._graph = GraphStore(path, read_only=True, check_same_thread=False)
            self._graph.connection.execute('BEGIN')
            self._version = self._data_version()
            selected = select_endpoints(self._graph, self.result, entity, target, refs)
            self._graph.connection.commit()
            if selected is None:
                self.result = operator_query_result(self.result)
                self.close()
                return
            self.result['selected'] = selected
            self.collection = 'path' if target is not None else 'claims'
            self.result[self.collection] = []
            self.cursor = ExplorationCursor(selected['entity']['id'], selected.get('target', {}).get('id'),
                                            direction, depth, refs)
            self.state = 'complete' if self.cursor.complete else 'running'
            self._refresh()
            self.check_current()
        except (OSError, ValueError, sqlite3.Error) as error:
            self._fail(error)

    def _file_identity(self):
        stat = self._graph.path.stat()
        return stat.st_dev, stat.st_ino

    def _data_version(self):
        return self._graph.connection.execute('PRAGMA data_version').fetchone()[0]

    def check_current(self):
        with self._lock:
            if self.state in ('invalidated', 'failed'):
                return False
            if self._graph is None:
                return True
            try:
                changed = self._file_identity() != self._identity or self._data_version() != self._version
            except (OSError, sqlite3.Error):
                changed = True
            if changed:
                self._invalidate('The graph changed. Run a new exploration before continuing or exporting.')
            elif self.verification and (problem := self.verification.problem()):
                self._invalidate(problem + ' Run a new exploration to refresh evidence checks.', 'source_changed')
                return False
            return not changed

    def advance(self, *, batch_size=100):
        with self._lock:
            if self.state != 'running' or not self.check_current():
                return
            try:
                self._graph.connection.execute('BEGIN')
                steps = self.cursor.advance(self._graph, batch_size=batch_size)
                self._graph.connection.commit()
                verified = verify_claims(self.project, [step.edge for step in steps], first_only=True,
                                         require_registered=True, cache=self.verification)
                if not self.check_current():
                    return
                for step, evidence in zip(steps, verified.results):
                    self._edges[step.edge.assertion_id] = step.edge
                    self.append_summary(claim_summary(step.edge, evidence, step))
                self.batches += 1
                if self.cursor.complete:
                    self.state = 'complete'
                self._refresh()
            except VerificationChanged as error:
                self._invalidate(str(error), 'source_changed')
            except (OSError, ValueError, sqlite3.Error) as error:
                self._fail(error)

    def inspect_assertion(self, assertion_id):
        with self._lock:
            if not self.check_current():
                raise VerificationChanged(self.result['problem'])
            try:
                result = inspect_assertion(self.project, assertion_id, verification_cache=self.verification)
                if not self.check_current():
                    raise VerificationChanged(self.result['problem'])
                self._refresh()
                return result
            except VerificationChanged as error:
                if self.state != 'invalidated':
                    self._invalidate(str(error), 'source_changed')
                raise

    def prepare_export(self, **kwargs):
        from .operator_exports import prepare_selection
        with self._lock:
            return prepare_selection(self, **kwargs)

    def _invalidate(self, problem, reason='graph_changed'):
        self.state, self.invalidation_reason = 'invalidated', reason
        self.result.update(status='unavailable', problem=problem)
        self._refresh()
        self.close()

    def pause(self):
        with self._lock:
            if self.state == 'running':
                self.state = 'paused'
                self._refresh()

    def resume(self):
        with self._lock:
            if self.state == 'paused' and self.check_current():
                self.state = 'running'
                self._refresh()

    def expand(self, depth):
        with self._lock:
            if self.cursor is None or not self.check_current():
                raise ValueError('Run a new exploration before expanding.')
            self.cursor.expand(depth)
            self.state = 'complete' if self.cursor.complete else 'running'
            self._refresh()

    def _refresh(self):
        if self.cursor is None:
            self.result = operator_query_result(self.result)
            return
        cursor = self.cursor
        complete = self.state == 'complete'
        if self.state not in ('invalidated', 'failed'):
            self.result['status'] = 'ok' if complete else 'exploring'
            if complete and not self.rows:
                self.result['status'] = 'ok' if cursor.path is not None else 'no_connection' if cursor.goal else 'no_claims'
                counts = self.result['selected']['entity']['relationships']
                self.result['empty_reason'] = ('empty_source_selection' if cursor.refs == [] else
                    'isolated_identity' if not counts['all_sources'] else
                    'source_filter' if not counts['selected_sources'] else 'direction_or_limits')
        reasons = [] if complete else [{'invalidated': getattr(self, 'invalidation_reason', 'graph_changed'),
                                        'failed': 'exploration_error'}.get(self.state, 'exploration_pending')]
        self.result.update(truncated=not complete, truncation_reasons=reasons,
                           nodes_visited=len(cursor.visited), edges_examined=cursor.edges_examined,
                           progress={'state': self.state, 'batches': self.batches, 'exhausted': cursor.exhausted,
                                     'pending_identities': 0 if complete else len(cursor.queue) + int(cursor.active is not None),
                                     'frontier_identities': len(cursor.queue) + int(cursor.active is not None)})
        self.result['limits']['depth'] = cursor.depth
        if cursor.goal is not None:
            found = cursor.path is not None and complete
            self.result.update(connected=found if complete else None, path_found=found, path_complete=found,
                               path_assertion_ids=[row['assertion_id'] for row in self.rows] if found else [])
        self.result = operator_query_result(self.result)
        self.result['inspection'].update(discovery_complete=complete, assertions_found=len(self.rows),
                                         assertions_returned=len(self.rows))
        self.result['query_snapshot'] = {'id': self.token, 'graph_data_version': self._version}
        self.result['verification_state'] = {
            **self.verification.metadata(), 'checked_assertions': len(self.rows),
            'unchecked_discovered_assertions': 0, 'undiscovered_assertions': 0 if complete else None,
            'discovery_complete': complete,
            'valid': self.state not in ('invalidated', 'failed')}

    def _fail(self, error):
        self.state = 'failed'
        self.result.update(status='unavailable', problem=str(error), truncated=True,
                           truncation_reasons=['exploration_error'])
        self._refresh()
        self.close()

    def close(self):
        with self._lock:
            if self._graph is not None:
                self._graph.close()
                self._graph = None

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()
