'''Explicit graph upgrades preserve a SQLite backup and commit schema changes atomically.'''

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .migration_records import LEGACY_NOTE, RENAME_NOTE, counts, inspect_schema, rename_fields, upgrade_records
from .state import SCHEMA_VERSION, build_lock, legacy_counts


def _backup(connection_path, version):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = connection_path.with_name(f'graph.backup-v{version}-{stamp}-{uuid4().hex[:8]}.sqlite')
    destination.touch(exist_ok=False)
    try:
        with closing(sqlite3.connect(connection_path.as_uri() + '?mode=ro', uri=True, timeout=0)) as source, \
                closing(sqlite3.connect(destination)) as target:
            source.backup(target)
            if target.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('Graph backup integrity check failed.')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _report(connection, project, path, dry_run):
    version = inspect_schema(connection)
    stored = counts(connection)
    requires_reextraction = any(stored.values()) if version == 0 else any(legacy_counts(connection).values())
    renames = rename_fields(connection) if version != SCHEMA_VERSION else {}
    return {'project': project.slug, 'path': str(path), 'dry_run': dry_run,
            'status': 'current' if version == SCHEMA_VERSION else 'migration_required',
            'from_version': version, 'to_version': SCHEMA_VERSION, **stored, 'backup': None,
            'requires_reextraction': requires_reextraction,
            'field_renames': renames,
            'notes': ([RENAME_NOTE] if version != SCHEMA_VERSION else []) + ([LEGACY_NOTE] if requires_reextraction else []),
            'provider_calls': 0}


def migrate_graph(project, *, dry_run=False):
    path = (Path(project.paths.root) / 'graph.sqlite').resolve()
    if not path.is_file():
        return {'project': project.slug, 'path': str(path), 'dry_run': dry_run, 'status': 'no_graph',
                'from_version': None, 'to_version': SCHEMA_VERSION, 'entities': 0, 'edges': 0,
                'backup': None, 'requires_reextraction': False, 'provider_calls': 0,
                'notes': ['No graph.sqlite exists. An explicit graph build will create the current format.']}
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=0)
    try:
        connection.execute('BEGIN')
        report = _report(connection, project, path, dry_run)
        if dry_run or report['status'] == 'current':
            return report
    finally:
        connection.close()
    with build_lock(path):
        connection = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=0)
        backup = None
        try:
            connection.execute('BEGIN IMMEDIATE')
            report = _report(connection, project, path, dry_run)
            if report['status'] == 'current':
                return report
            backup = _backup(path, report['from_version'])
            connection.row_factory = sqlite3.Row
            upgrade_records(connection, project, report['from_version'])
            connection.commit()
            return {**report, 'status': 'migrated', 'backup': str(backup)}
        except BaseException as error:
            connection.rollback()
            if isinstance(error, (sqlite3.Error, ValueError, TypeError, KeyError, OSError)):
                saved = f' Backup: {backup}.' if backup else ''
                raise ValueError(f'Graph migration failed; the original database was preserved. {error}.{saved}') from error
            raise
        finally:
            connection.close()
