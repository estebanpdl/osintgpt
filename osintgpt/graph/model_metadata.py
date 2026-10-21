'''Model identities for build previews and published evidence, without constructing providers.'''

import json

from .recipes import extraction_recipe, fingerprint


def recipe_data(entry):
    value = (entry or {}).get('recipe_data', {})
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def model_identity(recipe):
    value = recipe.get('provider', {})
    return value if isinstance(value, dict) else {}


def selection_metadata(generator, identity=None):
    identity = identity if identity is not None else extraction_recipe(generator)['provider']
    selection = getattr(generator, 'selection', None)
    return {**identity, 'mode': selection.mode if selection else 'explicit_generator'}


def model_label(identity):
    if not identity:
        return 'unknown'
    return f"{identity.get('provider') or identity.get('implementation') or 'unknown provider'} / {identity.get('model') or 'unknown model'}"


def selection_label(identity):
    mode = {'inherit': 'inherited from Chat/RAG', 'independent': 'independent graph selection',
            'override': 'this invocation only', 'explicit_generator': 'library generator'}.get(identity.get('mode'), 'unknown selection mode')
    return f'{model_label(identity)} ({mode})'


def published_models(connection, *, versions=None, refs=None):
    '''Describe completed publications, including sources without recoverable model identities.

    An optional version selection is from displayed query evidence. If that version
    is no longer published, report its model as unknown rather than borrowing a newer one.
    '''
    full = versions is None and refs is None
    entries = {row['ref']: dict(row) for row in connection.execute('SELECT * FROM graph_documents')}
    if versions is None:
        sources = {row[0] for row in connection.execute('''
            SELECT ref FROM graph_documents UNION SELECT source_ref FROM edges
            UNION SELECT source_ref FROM entity_sources
        ''')}
        versions = [(ref, entries.get(ref, {}).get('source_version', '')) for ref in sorted(sources)
                    if refs is None or ref in refs]
    rows = []
    for ref, version in sorted(set(versions)):
        entry = entries.get(ref)
        identity = model_identity(recipe_data(entry)) if entry and entry['source_version'] == version else {}
        rows.append({'ref': ref, 'source_version': version, 'identity': identity,
                     'known': bool(identity.get('model') and (identity.get('provider') or identity.get('implementation')))})
    identities = {fingerprint(row['identity']) for row in rows if row['known']}
    unattributed = connection.execute('''SELECT COUNT(*) FROM entities e
        WHERE NOT EXISTS (SELECT 1 FROM entity_sources s WHERE s.entity_key = e.key)''').fetchone()[0]
    return {'sources': rows, 'mixed': len(identities) > 1,
            'unknown_sources': sum(not row['known'] for row in rows),
            'unattributed_entities': unattributed if full else 0}


def selected_models(metadata, versions):
    wanted = set(versions)
    rows = [row for row in metadata.get('sources', []) if (row['ref'], row['source_version']) in wanted]
    found = {(row['ref'], row['source_version']) for row in rows}
    rows.extend({'ref': ref, 'source_version': version, 'identity': {}, 'known': False}
                for ref, version in sorted(wanted - found))
    return {'sources': rows, 'mixed': len({fingerprint(row['identity']) for row in rows if row['known']}) > 1,
            'unknown_sources': sum(not row['known'] for row in rows), 'unattributed_entities': 0}
