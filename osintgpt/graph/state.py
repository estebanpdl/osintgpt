'''Persistent build state, source completion, and provider-free status reads.'''

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from osintgpt.ingestion import Corpus, is_image
from osintgpt.ingestion.snapshots import read_snapshot, source_fingerprint, transcriptions_current
from osintgpt.ingestion.source_view import reference_for as source_ref
from .model_metadata import published_models

SCHEMA_VERSION = 8
UNSUPPORTED_SCHEMA = ('Unsupported graph schema version {version}; '
                      f'current version is {SCHEMA_VERSION}. '
                      'Run "osintgpt graph migrate --dry-run" '
                      'to check whether this layout can be upgraded.')
SCHEMA = '''
CREATE TABLE graph_build (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    status TEXT NOT NULL DEFAULT 'never_built',
    available INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    problem TEXT NOT NULL DEFAULT '',
    counts TEXT NOT NULL DEFAULT '{}',
    usage TEXT NOT NULL DEFAULT '{}',
    details TEXT NOT NULL DEFAULT '[]',
    last_event TEXT NOT NULL DEFAULT '{}'
);
INSERT INTO graph_build (id) VALUES (1);
CREATE TABLE graph_documents (
    ref TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    source_version TEXT NOT NULL DEFAULT '',
    quality_gaps INTEGER NOT NULL DEFAULT 0,
    recipe TEXT NOT NULL DEFAULT '',
    recipe_data TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE graph_runs (id INTEGER PRIMARY KEY, data TEXT NOT NULL);
'''


def initialize(connection, graph_schema):
    from .checkpoints import SCHEMA as CHECKPOINT_SCHEMA
    from .identity_storage import SCHEMA as IDENTITY_SCHEMA
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    if version == SCHEMA_VERSION:
        return
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    if version or tables:
        raise ValueError(UNSUPPORTED_SCHEMA.format(version=version))
    connection.executescript(
        'BEGIN;\n' + graph_schema + SCHEMA + CHECKPOINT_SCHEMA + IDENTITY_SCHEMA
        + f'\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;'
    )


@contextmanager
def transaction(connection):
    # Savepoints allow publication to include store, state, and journal writes atomically.
    name = 'graph_' + uuid4().hex
    connection.execute(f'SAVEPOINT {name}')
    try:
        yield
    except BaseException:
        connection.execute(f'ROLLBACK TO {name}')
        connection.execute(f'RELEASE {name}')
        raise
    else:
        connection.execute(f'RELEASE {name}')


def _lock_path(path):
    return Path(path).with_suffix('.build-lock.sqlite')


@contextmanager
def build_lock(path):
    # A separate SQLite transaction survives status commits and releases on crash.
    connection = sqlite3.connect(
        ':memory:' if str(path) == ':memory:' else str(_lock_path(path)), timeout=0
    )
    try:
        try:
            connection.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as error:
            raise ValueError('A graph build is already running for this project.') from error
        yield
    finally:
        connection.close()


def _running(path):
    if str(path) == ':memory:':
        return True
    lock = _lock_path(path)
    if not lock.exists():
        return False
    try:
        with build_lock(path):
            return False
    except ValueError:
        return True


def _now():
    return datetime.now(timezone.utc).isoformat()


def legacy_counts(connection):
    return {table: connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE json_extract(data, '$.provenance.requires_reextraction') = 1"
    ).fetchone()[0] for table in ('entities', 'edges')}


def _interrupted(row):
    row['status'] = 'interrupted'
    row['problem'] = 'The build process stopped before recording completion.'
    counts = row['counts']
    unresolved = counts.get('requests_attempted', 0) - sum(
        counts.get(key, 0) for key in ('requests_succeeded', 'requests_failed', 'requests_interrupted'))
    counts['requests_interrupted'] = counts.get('requests_interrupted', 0) + max(unresolved, 0)


class GraphState:
    def __init__(self, connection, path):
        self.connection = connection
        self.path = path

    def read(self):
        row = dict(self.connection.execute('SELECT * FROM graph_build WHERE id = 1').fetchone())
        row.pop('id')
        row['available'] = bool(row['available'])
        row['schema_version'] = SCHEMA_VERSION
        row['legacy'] = legacy_counts(self.connection)
        for key in ('counts', 'usage', 'details', 'last_event'):
            row[key] = json.loads(row[key])
        if row['status'] == 'building' and not _running(self.path):
            _interrupted(row)
        return row

    def begin(self, counts):
        with self.connection:
            previous = self.read()
            if previous['status'] == 'building':
                _interrupted(previous)
            if previous['started_at']:
                self.connection.execute('INSERT INTO graph_runs (data) VALUES (?)', (json.dumps(previous),))
            self.connection.execute(
                "UPDATE graph_build SET status = 'building', started_at = ?, "
                "finished_at = '', problem = '', counts = ?, usage = '{}', details = '[]', last_event = '{}' WHERE id = 1",
                (_now(), json.dumps(counts.to_dict()))
            )

    def progress(self, counts, usage, details=()):
        with self.connection:
            self.connection.execute(
                'UPDATE graph_build SET counts = ?, usage = ?, details = ? WHERE id = 1',
                (json.dumps(counts.to_dict()), json.dumps(usage), json.dumps(details))
            )

    def event(self, event):
        with self.connection:
            self.connection.execute('UPDATE graph_build SET last_event = ? WHERE id = 1',
                                    (json.dumps({'kind': event.kind, 'data': event.data, 'at': _now()}),))

    def finish(self, status, problem=''):
        with self.connection:
            self.connection.execute(
                'UPDATE graph_build SET status = ?, finished_at = ?, problem = ?, '
                'available = CASE WHEN ? IN (\'complete\', \'complete_with_gaps\') '
                'THEN 1 ELSE available END WHERE id = 1',
                (status, _now(), problem, status)
            )

    def available(self):
        self.connection.execute(
            "UPDATE graph_build SET available = 1, "
            "status = CASE WHEN status = 'never_built' THEN 'complete' ELSE status END WHERE id = 1"
        )

    def completed(self, ref, fingerprint, source_version='', quality_gaps=False, recipe='', recipe_data=None):
        with transaction(self.connection):
            self.connection.execute(
                'INSERT OR REPLACE INTO graph_documents VALUES (?, ?, ?, ?, ?, ?)',
                (ref, fingerprint, source_version, int(quality_gaps), recipe, json.dumps(recipe_data or {}))
            )

    def entries(self):
        return {row['ref']: dict(row) for row in self.connection.execute('SELECT * FROM graph_documents')}

    def documents(self):
        return dict(self.connection.execute('SELECT ref, fingerprint FROM graph_documents').fetchall())

    def versions(self):
        return dict(self.connection.execute('SELECT ref, source_version FROM graph_documents').fetchall())

    def has_quality_gaps(self):
        return self.connection.execute('SELECT 1 FROM graph_documents WHERE quality_gaps = 1 LIMIT 1').fetchone() is not None


def freshness(project, documents, versions=None):
    corpus = Corpus.load(project.paths.sources)
    root = project.paths.root
    mappings = corpus.mappings(root)
    current = {
        source_ref(path, root): source_fingerprint(path, mappings.get(path.resolve()))
        for path in corpus.files(root) if not is_image(path)
    }
    if current != documents:
        return False
    for version in (versions or {}).values():
        if version and not transcriptions_current(read_snapshot(project.paths.extracts, version),
                                                  project.paths.extracts):
            return False
    return True


def saved_graph_status(project, *, check_freshness=True):
    path = project.paths.root / 'graph.sqlite'
    if not path.is_file():
        return {'status': 'never_built', 'available': False, 'counts': {}, 'usage': {},
                'entities': 0, 'edges': 0, 'problem': '', 'schema_version': None,
                'stale': False, 'checkpoints': {}, 'previous_runs': [], 'details': [],
                'extraction_models': {'sources': [], 'mixed': False, 'unknown_sources': 0, 'unattributed_entities': 0},
                'run_model': {}}
    connection = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError(UNSUPPORTED_SCHEMA.format(version=version))
        connection.execute('BEGIN')
        state = GraphState(connection, path)
        result = state.read()
        result['extraction_models'] = published_models(connection)
        result['run_model'] = result['usage'].get('graph_model', {})
        from .checkpoints import Checkpoints
        result['checkpoints'] = Checkpoints(state).summary()
        result['previous_runs'] = [json.loads(row[0]) for row in connection.execute(
            'SELECT data FROM graph_runs ORDER BY id DESC LIMIT 10')]
        result.update({
            'identity_revision': connection.execute('SELECT COALESCE(MAX(rowid),0) FROM identity_decisions').fetchone()[0],
            'entities': connection.execute('SELECT COUNT(DISTINCT COALESCE(l.canonical_key,e.key)) '
                                           'FROM entities e LEFT JOIN entity_links l ON l.member_key=e.key').fetchone()[0],
            'edges': connection.execute('SELECT COUNT(*) FROM edges').fetchone()[0],
        })
        result['stale'] = False
        if check_freshness:
            refresh_status(project, state, result)
        return result
    finally:
        connection.close()


def refresh_status(project, state, result):
    result['stale'] = False
    if not result['available'] or not result['started_at']:
        return
    from .checkpoints import Checkpoints
    try:
        entries = state.entries()
        pending_changes = any(ref in entries and (job['recipe'], job['source_version']) !=
                              (entries[ref]['recipe'], entries[ref]['source_version'])
                              for ref, job in Checkpoints(state).jobs().items())
        result['stale'] = pending_changes or not freshness(project, state.documents(), state.versions())
        if result['stale'] and result['status'] == 'complete':
            result['status'] = 'stale'
    except (OSError, ValueError) as error:
        result['stale'] = True
        if result['status'] == 'complete':
            result.update(status='stale', problem=str(error))
