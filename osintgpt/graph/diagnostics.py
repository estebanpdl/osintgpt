'''Explicit local inspection of retained generation diagnostics, independent of providers.'''

import json

from .store import GraphStore


def generation_diagnostics(project, *, offset=0, limit=50, raw=False):
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
        raise ValueError('Use a nonnegative offset and positive limit.')
    path = project.paths.root / 'graph.sqlite'
    if not path.exists():
        return {'results': [], 'total': 0, 'next_offset': 0, 'complete': True}
    with GraphStore(path, read_only=True) as graph:
        query = "FROM graph_responses WHERE purpose LIKE 'journal:%' AND purpose != 'journal:plan'"
        total = graph.connection.execute('SELECT COUNT(*) ' + query).fetchone()[0]
        rows = graph.connection.execute('SELECT * ' + query + ' ORDER BY rowid LIMIT ? OFFSET ?', (limit, offset))
        results = []
        for row in rows:
            data = json.loads(row['data'])
            meta = data.pop('_journal', {})
            usage = data.get('usage') or {}
            provider = meta.get('provider', {})
            item = {'source': row['source_ref'], 'record': row['ref'], 'window': row['position'],
                    'purpose': meta.get('purpose'), 'state': meta.get('state'), 'problem': meta.get('problem', ''),
                    'source_version': meta.get('source_version'), 'previous_state': meta.get('previous_state'),
                    'source_range': meta.get('source_range'),
                    'reusable': bool(meta.get('reusable') and graph.connection.execute(
                        'SELECT 1 FROM graph_responses WHERE source_ref=? AND ref=? AND position=? AND purpose=? AND request_hash=?',
                        (row['source_ref'], row['ref'], row['position'], meta.get('purpose'), row['request_hash'])).fetchone()),
                    'request_hash': row['request_hash'], 'provider': usage.get('provider', provider.get('provider')),
                    'model': usage.get('model', provider.get('model')),
                    'status': data.get('status'), 'stop_reason': data.get('stop_reason'),
                    'text_bytes': len(data.get('text', '').encode('utf-8')), 'tool_calls': len(data.get('tool_calls', []))}
            if raw:
                item['response'] = data
            results.append(item)
        end = offset + len(results)
        return {'results': results, 'total': total, 'next_offset': end, 'complete': end >= total}
