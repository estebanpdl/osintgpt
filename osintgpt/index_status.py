'''Read saved index/checkpoint counts without opening providers or changing data.'''

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .ingestion.indexing import IndexState


def saved_index_status(project):
    result = {'published': [], 'pending': [], 'problems': []}
    try:
        result['published'] = [asdict(row) for row in IndexState.load(project.paths.index_state).documents.values()]
    except (OSError, ValueError, TypeError) as error:
        result['problems'].append(f'Could not read the completed index status: {error}')
    path = Path(project.paths.index_journal)
    if not path.exists():
        return result
    connection = None
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        columns = {row[1] for row in connection.execute('PRAGMA table_info(jobs)')}
        details = 'details' if 'details' in columns else "'{}' AS details"
        for row in connection.execute(f'SELECT ref, total, completed, status, {details} FROM jobs ORDER BY ref'):
            data = dict(row)
            data['details'] = json.loads(data['details'])
            if not isinstance(data['details'], dict):
                raise ValueError('Invalid checkpoint display metadata')
            result['pending'].append(data)
    except (OSError, sqlite3.Error, ValueError) as error:
        result['pending'] = []
        result['problems'].append(f'Could not read saved checkpoints: {error}')
    finally:
        if connection is not None:
            connection.close()
    return result


def model_changes(status, provider, model):
    '''Only identify explicit mismatches; older records may omit the provider.'''
    known = [(r['embedding_provider'], r['embedding_model']) for r in status['published']]
    known += [(r['details'].get('provider', ''), r['details'].get('model', '')) for r in status['pending']]
    return sorted({(p, m) for p, m in known if (p and p != provider) or (m and m != model)})


def status_rows(status):
    published = {r['ref']: r for r in status['published']}
    pending = {r['ref']: r for r in status['pending']}
    rows = []
    for ref in sorted(published.keys() | pending.keys()):
        old, job = published.get(ref, {}), pending.get(ref, {})
        details = job.get('details', {})
        stats = details.get('statistics', {}) if job else old.get('statistics', {})
        rows.append({
            'Source': ref,
            'State': ('Ready to publish' if job['completed'] == job['total'] else 'Checkpoint saved') if job else 'Last completed',
            'Provider': details.get('provider', '') if job else old.get('embedding_provider', ''),
            'Model': details.get('model', '') if job else old.get('embedding_model', ''),
            'Last completed chunks': old.get('chunks', 0),
            'Saved checkpoint chunks': job.get('completed', 0),
            'Pending embedding chunks': job['total'] - job['completed'] if job else 0,
            'Records kept': stats.get('kept', stats.get('documents')),
            'No content': stats.get('without_content'),
            'Below minimum': stats.get('too_short'),
        })
    return rows
