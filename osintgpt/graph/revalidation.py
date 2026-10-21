'''Recheck saved rejected proposals locally, preserving their original evidence scope.'''

import json
from dataclasses import asdict

from osintgpt.ingestion.snapshots import read_snapshot
from .checkpoints import rejection_from
from .grounding import EvidenceScope
from .identity import EntityRegistry
from .migration import _backup
from .recipes import fingerprint
from .state import SCHEMA_VERSION, build_lock, transaction
from .store import GraphStore
from .validation import validate_edge
from .records import normalize_relation_polarity


def revalidate_proposals(project, *, refs=None, limit=100, offset=0, apply=False, expected_revision=None):
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise ValueError('Revalidation needs a limit from 1 to 100 and a nonnegative offset.')
    path = project.paths.root / 'graph.sqlite'
    with build_lock(path), GraphStore(path, read_only=not apply) as graph:
        if apply:
            graph.connection.execute('BEGIN IMMEDIATE')
        try:
            rows = list(graph.connection.execute('''SELECT id,data FROM graph_rejections
                WHERE json_extract(data,'$.status')='rejected' ORDER BY id'''))
            wanted = [(row['id'], rejection_from(json.loads(row['data']))) for row in rows]
            wanted = [(key, item) for key, item in wanted if refs is None or item.source_ref in refs or item.ref in refs]
            total = len(wanted)
            wanted = wanted[offset:offset + limit]
            registry = EntityRegistry(graph.entities())
            revision = fingerprint({'audit': [(key, asdict(item)) for key, item in wanted],
                                    'entities': [asdict(entity) for entity in registry.entities.values()],
                                    'sources': graph.state.versions()})
            if expected_revision is not None and expected_revision != revision:
                raise ValueError('Saved evidence changed after preview. Preview revalidation again.')
            result = {'status': 'preview', 'provider_calls': 0, 'revision': revision, 'outcomes': [],
                      'candidates_total': total, 'candidates_checked': len(wanted), 'offset': offset,
                      'next_offset': offset + len(wanted), 'truncated': offset + len(wanted) < total,
                      'accepted_entities': [], 'accepted_claims': [], 'backup': None,
                      'note': 'Original evidence ranges and endpoint identities are required. Older incomplete audit '
                              'records remain rejected; this action never expands their evidence scope or calls a model.'}
            versions = graph.state.versions()
            found, local = {}, {}
            accepted = []
            # Entity evidence is checked first so dependent claims can reuse recovered endpoints.
            for key, item in sorted(wanted, key=lambda row: row[1].candidate.get('kind') != 'entity'):
                registry.warnings.clear()
                registry.window = item.window
                group = (item.ref, item.source_version, item.window)
                try:
                    if versions.get(item.source_ref) != item.source_version:
                        raise ValueError('Proposal does not belong to the currently published source version.')
                    if not item.supplied_ranges:
                        raise ValueError('Original supplied source ranges were not recorded; explicit re-extraction is needed.')
                    snapshot = read_snapshot(project.paths.extracts, item.source_version)
                    doc = snapshot.document(item.ref)
                    if doc is None:
                        raise ValueError('Saved source record is unavailable.')
                    if any(type(start) is not int or type(end) is not int or not 0 <= start < end <= len(doc.text)
                           for start, end in item.supplied_ranges):
                        raise ValueError('Saved source ranges are invalid.')
                    scope = EvidenceScope(doc, snapshot, item.supplied_ranges)
                    entities = found.setdefault(group, {})
                    names = local.setdefault(group, {})
                    for entity in registry.entities.values():
                        for mention in entity.provenance.get('mentions', []):
                            if (mention.get('document_ref') == item.ref and mention.get('source_version') == item.source_version
                                    and mention.get('window') == item.window):
                                if mention.get('local_id'):
                                    names.setdefault(mention['local_id'], entity.key)
                    allowed = set(registry.entities)
                    if item.candidate.get('kind') == 'entity':
                        resolved, declared = registry.resolve([item.candidate], scope, allowed)
                        entities.update(resolved)
                        names.update(declared)
                        registry.commit(resolved.values())
                        result['accepted_entities'].extend(asdict(entity) for entity in resolved.values())
                    else:
                        edge = validate_edge(normalize_relation_polarity(item.candidate), scope, entities, names, registry, allowed)
                        result['accepted_claims'].append(asdict(edge))
                    accepted.append((key, item))
                    result['outcomes'].append({'id': key, 'source': item.ref, 'status': 'accepted_locally',
                                               'warnings': [warning for _, warning in registry.warnings]})
                    registry.warnings.clear()
                except (ValueError, TypeError, OSError, KeyError) as error:
                    result['outcomes'].append({'id': key, 'source': item.ref, 'status': 'rejected', 'reason': str(error)})
            if not apply or not accepted:
                return result
            from .checkpoints import entity_from, edge_from
            result['backup'] = str(_backup(path.resolve(), SCHEMA_VERSION))
            with transaction(graph.connection):
                graph.add([entity_from(row) for row in result['accepted_entities']],
                          [edge_from(row) for row in result['accepted_claims']])
                for key, item in accepted:
                    data = {**asdict(item), 'status': 'accepted_locally'}
                    graph.connection.execute('UPDATE graph_rejections SET data=? WHERE id=?', (json.dumps(data), key))
                for ref in {item.source_ref for _, item in accepted}:
                    unresolved = graph.connection.execute('''SELECT 1 FROM graph_rejections WHERE source_ref=?
                        AND source_version=? AND json_extract(data,'$.status')='rejected' LIMIT 1''',
                        (ref, versions[ref])).fetchone()
                    graph.connection.execute('UPDATE graph_documents SET quality_gaps=? WHERE ref=?', (bool(unresolved), ref))
            graph.connection.commit()
            result['status'] = 'applied'
            # Accepted entries leave the rejected set; do not skip shifted rows.
            result['next_offset'] -= len(accepted)
            return result
        except BaseException:
            if apply:
                graph.connection.rollback()
            raise
