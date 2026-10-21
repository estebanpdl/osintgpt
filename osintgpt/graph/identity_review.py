'''Operator review and reversible reconciliation of already published identities.'''

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from uuid import uuid4

from .entity_catalog import EntityCatalog, evidence_entries
from .identity_storage import identity_links, resolutions
from .migration import _backup
from .recipes import fingerprint
from .state import SCHEMA_VERSION, build_lock, transaction
from .store import GraphStore
from .records import merge_entity
from .checkpoints import entity_from


def review_identities(project, query='', *, limit=30, offset=0):
    if type(limit) is not int or not 1 <= limit <= 60 or type(offset) is not int or offset < 0:
        raise ValueError('Use limit 1–60 and a nonnegative offset.')
    with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
        entities = graph.entities()
        links = identity_links(graph.connection)
        catalog = EntityCatalog(entities)
        if query:
            found, _, total = catalog.search(links.get(query, query), offset=offset, limit=limit)
        else:
            duplicate_ids = {key for ids in catalog.names.values() if len(ids) > 1 for key in ids}
            candidates = sorted((entity for entity in entities if entity.key in duplicate_ids), key=lambda e: e.key)
            found, total = candidates[offset:offset + limit], len(candidates)
        return {'status': 'ok', 'candidates': [{**asdict(entity), 'evidence': evidence_entries(entity),
                                                'relationships': graph.relationship_counts(entity.key)} for entity in found],
                'total': total, 'next_offset': offset + len(found), 'truncated': offset + len(found) < total,
                'decisions': resolutions(graph.connection), 'provider_calls': 0,
                'note': 'Matching names identify review candidates, not automatic merges. '
                        'Reconciliation preserves original assertions and source contributions.'}


def _links_snapshot(connection):
    return [list(row) for row in connection.execute('SELECT * FROM entity_links ORDER BY member_key')]


def reconcile_identities(project, source_id='', target_id='', *, reason='', undo='', apply=False,
                         expected_revision=None, split_source=''):
    path = (project.paths.root / 'graph.sqlite').resolve()
    with build_lock(path), GraphStore(path, read_only=not apply) as graph:
        connection = graph.connection
        if apply:
            connection.execute('BEGIN IMMEDIATE')
        try:
            links = identity_links(connection)
            before = _links_snapshot(connection)
            revision = fingerprint({'links': before, 'entities': [asdict(entity) for entity in graph.entities()]})
            if expected_revision is not None and revision != expected_revision:
                raise ValueError('Graph identities changed after preview. Preview the decision again.')
            decision_id = 'review_' + uuid4().hex
            if split_source:
                return _split(graph, project, source_id, split_source, reason, apply, revision)
            if undo:
                entries = resolutions(connection)
                selected = next((item for item in entries if item['id'] == undo), None)
                if selected is None or selected.get('undo') or selected.get('reversed_by') or selected.get('action') == 'split_source':
                    raise ValueError('Select an active reconciliation decision to undo.')
                if before != selected['after']:
                    raise ValueError('Later identity decisions depend on this mapping. Undo those decisions first.')
                after = selected['before']
                note = 'Restore the identity mapping before this decision; preserve all source claims.'
            else:
                if not source_id or not target_id or not reason.strip():
                    raise ValueError('Supply both identity IDs and a source-based reason for reconciliation.')
                source_id, target_id = links.get(source_id, source_id), links.get(target_id, target_id)
                catalogue = {entity.key: entity for entity in graph.entities()}
                if source_id not in catalogue or target_id not in catalogue or source_id == target_id:
                    raise ValueError('Select two distinct existing identities.')
                members = {source_id, *(member for member, canonical in links.items() if canonical == source_id)}
                after = [row for row in before if row[0] not in members]
                after += [[member, target_id, decision_id] for member in members]
                after.sort()
                note = 'Reuse the target identity while retaining original node IDs and assertions for reversible review.'
            report = {'status': 'preview', 'id': decision_id, 'from_id': source_id, 'to_id': target_id,
                      'reason': reason, 'undo': undo, 'before': before, 'after': after, 'revision': revision,
                      'note': note, 'provider_calls': 0, 'backup': None,
                      'recorded_at': datetime.now(timezone.utc).isoformat()}
            if not apply:
                return report
            report['backup'] = str(_backup(path, SCHEMA_VERSION))
            with transaction(connection):
                connection.execute('DELETE FROM entity_links')
                connection.executemany('INSERT INTO entity_links VALUES (?, ?, ?)', after)
                report['status'] = 'applied'
                connection.execute('INSERT INTO identity_decisions VALUES (?, ?)', (decision_id, json.dumps(report)))
                if undo:
                    selected['reversed_by'] = decision_id
                    connection.execute('UPDATE identity_decisions SET data=? WHERE id=?', (json.dumps(selected), undo))
            connection.commit()
            return report
        except BaseException:
            if apply:
                connection.rollback()
            raise


def _split(graph, project, identity, source_ref, reason, apply, revision):
    '''Detach one source's contribution after an incorrect model reuse decision.'''
    if not identity or not source_ref or not reason.strip():
        raise ValueError('Splitting requires an identity ID, an exact source reference, and a reason.')
    links = identity_links(graph.connection)
    canonical = links.get(identity, identity)
    members = [canonical, *(member for member, target in links.items() if target == canonical)]
    rows = graph.connection.execute('''SELECT * FROM entity_sources WHERE source_ref=?
        AND entity_key IN (SELECT value FROM json_each(?))''', (source_ref, json.dumps(members))).fetchall()
    if not rows:
        raise ValueError('No sourced contribution matches this identity and source. Inspect its mentions first.')
    remaining = graph.connection.execute('''SELECT 1 FROM entity_sources WHERE source_ref<>?
        AND entity_key IN (SELECT value FROM json_each(?)) LIMIT 1''', (source_ref, json.dumps(members))).fetchone()
    if not remaining:
        raise ValueError('This identity has no other sourced contribution to separate from.')
    new_id, decision_id = 'ent_' + uuid4().hex, 'review_' + uuid4().hex
    report = {'status': 'preview', 'id': decision_id, 'action': 'split_source', 'from_id': canonical,
              'to_id': new_id, 'source_ref': source_ref, 'reason': reason, 'revision': revision,
              'provider_calls': 0, 'backup': None, 'recorded_at': datetime.now(timezone.utc).isoformat(),
              'note': 'Separate all mentions and assertion endpoints contributed by this source. '
                      'Original assertion IDs and evidence are preserved. Reconcile the IDs to combine them again.'}
    if not apply:
        return report
    report['backup'] = str(_backup(graph.path, SCHEMA_VERSION))
    contribution = entity_from(json.loads(rows[0]['data']))
    for row in rows[1:]:
        contribution = merge_entity(contribution, entity_from(json.loads(row['data'])))
    decision = {'action': 'create', 'method': 'operator_split', 'reason': reason, 'id': decision_id,
                'previous_identity': canonical, 'source_ref': source_ref}
    contribution = replace(contribution, key=new_id,
                           provenance={**contribution.provenance, 'resolution': decision})
    with transaction(graph.connection):
        graph.add([contribution], [])
        for column in ('source_key', 'target_key'):
            graph.connection.execute(f'''UPDATE edges SET {column}=? WHERE source_ref=?
                AND {column} IN (SELECT value FROM json_each(?))''', (new_id, source_ref, json.dumps(members)))
        graph.connection.execute('''DELETE FROM entity_sources WHERE source_ref=?
            AND entity_key IN (SELECT value FROM json_each(?))''', (source_ref, json.dumps(members)))
        for key in members:
            graph._refresh_entity(key)
            orphan = graph.connection.execute('''SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM entity_sources WHERE entity_key=?)
                AND NOT EXISTS (SELECT 1 FROM edges WHERE source_key=? OR target_key=?)''', (key, key, key)).fetchone()
            if orphan:
                graph.connection.execute('DELETE FROM entities WHERE key=?', (key,))
                if key != canonical:
                    graph.connection.execute('INSERT OR REPLACE INTO entity_links VALUES (?, ?, ?)', (key, new_id, decision_id))
        report['status'] = 'applied'
        graph.connection.execute('INSERT INTO identity_decisions VALUES (?, ?)', (decision_id, json.dumps(report)))
    graph.connection.commit()
    return report
