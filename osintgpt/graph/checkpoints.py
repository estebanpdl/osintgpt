'''Durable, unpublished extraction windows and their identity context.'''

import json
from uuid import uuid4
from dataclasses import asdict

from osintgpt.llm.generation import response_from

from .records import Edge, Entity, RejectedClaim
from .state import transaction


class SavedInvalidResponse(ValueError):
    def __init__(self, response, problem):
        self.response = response
        super().__init__(problem)

SCHEMA = '''
CREATE TABLE graph_jobs (
    source_ref TEXT PRIMARY KEY, source_version TEXT NOT NULL,
    recipe TEXT NOT NULL, recipe_data TEXT NOT NULL, seed TEXT NOT NULL
);
CREATE TABLE graph_records (
    source_ref TEXT NOT NULL, ref TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL,
    PRIMARY KEY (source_ref, ref)
);
CREATE TABLE graph_windows (
    source_ref TEXT NOT NULL, ref TEXT NOT NULL, position INTEGER NOT NULL,
    data TEXT NOT NULL, PRIMARY KEY (source_ref, ref, position)
);
CREATE TABLE graph_responses (
    source_ref TEXT NOT NULL, ref TEXT NOT NULL, position INTEGER NOT NULL,
    purpose TEXT NOT NULL, request_hash TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (source_ref, ref, position, purpose)
);
'''


def entity_from(data):
    return Entity(**{**data, 'aliases': tuple(data.get('aliases', [])),
                     'identifiers': tuple(data.get('identifiers', []))})


def edge_from(data):
    return Edge.from_dict(data)


def rejection_from(data):
    return RejectedClaim(**{**data, 'corrected_evidence': tuple(data.get('corrected_evidence', []))})


class Checkpoints:
    def __init__(self, graph):
        self.graph, self.connection = graph, graph.connection

    def jobs(self):
        return {row['source_ref']: dict(row) for row in
                self.connection.execute('SELECT source_ref, source_version, recipe, recipe_data FROM graph_jobs')}

    def discard(self, ref):
        with transaction(self.connection):
            self.supersede(ref)
            self.connection.execute("DELETE FROM graph_responses WHERE source_ref = ? AND purpose NOT LIKE 'journal:%'", (ref,))
            self.connection.execute("DELETE FROM graph_responses WHERE source_ref = ? AND purpose = 'journal:plan'", (ref,))
            for table in ('graph_windows', 'graph_records', 'graph_jobs'):
                self.connection.execute(f'DELETE FROM {table} WHERE source_ref = ?', (ref,))

    def supersede(self, ref, record=None, position=0, purpose=None):
        '''Retain audits while retiring active responses tied to an obsolete request/plan.'''
        rows = self.connection.execute("SELECT * FROM graph_responses WHERE source_ref=? AND position>=? "
            "AND purpose NOT LIKE 'journal:%' AND purpose NOT LIKE 'lookup-%'", (ref, position)).fetchall()
        for row in rows:
            if (record is not None and row['ref'] != record) or (purpose is not None and row['purpose'] != purpose):
                continue
            data = json.loads(row['data'])
            meta = data.setdefault('_journal', {'purpose': row['purpose'], 'audit_id': 'journal:' + uuid4().hex})
            meta.update(previous_state=meta.get('state', 'received'), state='superseded', reusable=False)
            self.connection.execute('INSERT OR REPLACE INTO graph_responses VALUES (?, ?, ?, ?, ?, ?)',
                (ref, row['ref'], row['position'], meta['audit_id'], row['request_hash'], json.dumps(data)))

    def prepare(self, ref, version, recipe, data):
        with transaction(self.connection):
            self.discard(ref)
            self.connection.execute('INSERT INTO graph_jobs VALUES (?, ?, ?, ?, ?)',
                                    (ref, version, recipe, json.dumps(data),
                                     json.dumps([asdict(entity) for entity in self.graph.entities()])))

    def seed(self, ref):
        data = self.connection.execute('SELECT seed FROM graph_jobs WHERE source_ref = ?', (ref,)).fetchone()[0]
        return [entity_from(item) for item in json.loads(data)]

    def recipe(self, ref):
        row = self.connection.execute('SELECT recipe_data FROM graph_jobs WHERE source_ref = ?', (ref,)).fetchone()
        return json.loads(row[0]) if row else {}

    def record(self, source_ref, ref, seed, total):
        with transaction(self.connection):
            self.connection.execute('INSERT OR IGNORE INTO graph_records '
                                    '(source_ref, ref, total) VALUES (?, ?, ?)', (source_ref, ref, total))
        return RecordCheckpoint(self, source_ref, ref, seed)

    def summary(self):
        result = dict(self.connection.execute('''
            SELECT COUNT(*) AS records, COALESCE(SUM(total), 0) AS windows_total
            FROM graph_records
        ''').fetchone())
        result['documents'] = self.connection.execute('SELECT COUNT(*) FROM graph_jobs').fetchone()[0]
        result['windows_saved'] = self.connection.execute('SELECT COUNT(*) FROM graph_windows').fetchone()[0]
        result['windows_pending'] = result['windows_total'] - result['windows_saved']
        result['responses_saved'] = self.connection.execute("SELECT COUNT(*) FROM graph_responses WHERE purpose NOT LIKE 'lookup-%' AND purpose NOT LIKE 'journal:%'").fetchone()[0]
        result['lookups_saved'] = self.connection.execute("SELECT COUNT(*) FROM graph_responses WHERE purpose LIKE 'lookup-%'").fetchone()[0]
        return result

    def records(self, source_ref):
        rows = self.connection.execute('''
            SELECT ref, attempts, total,
                (SELECT COUNT(*) FROM graph_windows w WHERE w.source_ref = r.source_ref AND w.ref = r.ref) AS saved,
                (SELECT COUNT(*) FROM graph_responses p WHERE p.source_ref = r.source_ref AND p.ref = r.ref
                 AND p.purpose NOT LIKE 'lookup-%' AND p.purpose NOT LIKE 'journal:%') AS responses,
                (SELECT COUNT(*) FROM graph_responses p WHERE p.source_ref = r.source_ref AND p.ref = r.ref
                 AND p.purpose NOT LIKE 'journal:%' AND json_extract(p.data,'$._journal.state')='invalid') AS responses_invalid
            FROM graph_records r WHERE source_ref = ?
        ''', (source_ref,))
        return {row['ref']: dict(row) for row in rows}


class RecordCheckpoint:
    def __init__(self, journal, source_ref, ref, seed):
        self.journal, self.source_ref, self.ref = journal, source_ref, ref
        row = journal.connection.execute('SELECT * FROM graph_records WHERE source_ref = ? AND ref = ?',
                                         (source_ref, ref)).fetchone()
        self.seed = seed
        self.attempts = row['attempts']
        self.legacy = journal.recipe(source_ref).get('version') == 4

    def windows(self):
        return [(row['position'], json.loads(row['data'])) for row in self.journal.connection.execute(
            'SELECT position, data FROM graph_windows WHERE source_ref = ? AND ref = ? ORDER BY position',
            (self.source_ref, self.ref))]

    def progress(self, counts):
        with transaction(self.journal.connection):
            self.journal.connection.execute('UPDATE graph_records SET attempts = ? '
                                           'WHERE source_ref = ? AND ref = ?',
                                           (self.attempts + counts.requests_attempted, self.source_ref, self.ref))

    def response(self, position, purpose, request_hash):
        row = self.journal.connection.execute(
            'SELECT data FROM graph_responses WHERE source_ref = ? AND ref = ? AND position = ? '
            'AND purpose = ? AND request_hash = ?',
            (self.source_ref, self.ref, position, purpose, request_hash)).fetchone()
        return self._decode(row[0]) if row else None

    @staticmethod
    def _decode(raw):
        data = json.loads(raw)
        state = data.pop('_journal', {})
        response = response_from(data)
        if state.get('state') == 'invalid':
            raise SavedInvalidResponse(response, state.get('problem', 'Saved response is invalid.'))
        return response

    def previous(self, position, purpose):
        '''Recover completed work under a compatible legacy window, not exact request replay.'''
        row = self.journal.connection.execute('SELECT data FROM graph_responses WHERE source_ref=? AND ref=? '
            'AND position=? AND purpose=?', (self.source_ref, self.ref, position, purpose)).fetchone()
        if not row:
            return None
        metadata = json.loads(row[0]).get('_journal')
        return self._decode(row[0]) if metadata is None or metadata.get('origin') == 'legacy' else None

    def save_response(self, position, purpose, request_hash, response, request_metadata=None):
        with transaction(self.journal.connection):
            data = asdict(response)
            if not purpose.startswith(('lookup-', 'journal:')):
                self.journal.supersede(self.source_ref, self.ref, position, purpose)
                audit_id = 'journal:' + uuid4().hex
                recipe = self.journal.recipe(self.source_ref)
                job = self.journal.connection.execute('SELECT source_version FROM graph_jobs WHERE source_ref=?',
                                                     (self.source_ref,)).fetchone()
                data['_journal'] = {'state': 'received', 'reusable': True, 'purpose': purpose, 'audit_id': audit_id,
                                    'source_version': job[0] if job else '', 'provider': recipe.get('provider', {}),
                                    **(request_metadata or {})}
                self.journal.connection.execute('INSERT INTO graph_responses VALUES (?, ?, ?, ?, ?, ?)',
                    (self.source_ref, self.ref, position, audit_id, request_hash, json.dumps(data)))
            self.journal.connection.execute('INSERT OR REPLACE INTO graph_responses VALUES (?, ?, ?, ?, ?, ?)',
                (self.source_ref, self.ref, position, purpose, request_hash, json.dumps(data)))

    def plan(self, parts):
        from .windowing import Window
        saved = self.response(0, 'journal:plan', 'source-plan')
        return [Window(**part) for part in json.loads(saved.text)] if saved else parts

    def replan(self, parts, position):
        from osintgpt.llm.generation import GenerationResponse
        with transaction(self.journal.connection):
            self.save_response(0, 'journal:plan', 'source-plan', GenerationResponse(json.dumps([asdict(p) for p in parts])))
            self.journal.connection.execute('UPDATE graph_records SET total=? WHERE source_ref=? AND ref=?',
                                           (len(parts), self.source_ref, self.ref))
            self.journal.supersede(self.source_ref, self.ref, position)
            self.journal.connection.execute("DELETE FROM graph_responses WHERE source_ref=? AND ref=? AND position>=? "
                                           "AND purpose NOT LIKE 'journal:%'", (self.source_ref, self.ref, position))

    def discard_response(self, position, purpose, problem='Invalid response.', state='invalid'):
        with transaction(self.journal.connection):
            row = self.journal.connection.execute('SELECT request_hash,data FROM graph_responses WHERE source_ref=? '
                'AND ref=? AND position=? AND purpose=?', (self.source_ref, self.ref, position, purpose)).fetchone()
            if not row:
                return
            data = json.loads(row['data'])
            meta = data.setdefault('_journal', {'purpose': purpose, 'audit_id': 'journal:' + uuid4().hex, 'origin': 'legacy'})
            meta.update(state=state, problem=problem, reusable=state != 'invalid')
            raw = json.dumps(data)
            self.journal.connection.execute('INSERT OR REPLACE INTO graph_responses VALUES (?, ?, ?, ?, ?, ?)',
                (self.source_ref, self.ref, position, meta['audit_id'], row['request_hash'], raw))
            self.journal.connection.execute('UPDATE graph_responses SET data=? WHERE source_ref=? AND ref=? '
                'AND position=? AND purpose=?', (raw, self.source_ref, self.ref, position, purpose))

    def save(self, position, entities, edges, rejected):
        data = {'entities': [asdict(e) for e in entities], 'edges': [asdict(e) for e in edges],
                'rejected': [asdict(item) for item in rejected]}
        with transaction(self.journal.connection):
            self.journal.connection.execute('INSERT INTO graph_windows VALUES (?, ?, ?, ?)',
                                           (self.source_ref, self.ref, position, json.dumps(data)))
            self.journal.graph.record_rejections(rejected)
            self.journal.connection.execute('DELETE FROM graph_responses WHERE source_ref = ? AND ref = ? '
                                           "AND position = ? AND purpose NOT LIKE 'journal:%'", (self.source_ref, self.ref, position))
