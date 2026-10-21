'''Recheck saved identity proposals against the published graph before committing.'''

from dataclasses import replace

from .entity_catalog import evidence_entries
from .entity_resolution import shared_identifier, conflicting_identifiers
from .identity_storage import identity_links
from .records import merge_entity
from .recipes import fingerprint
from .store import _source_entity


def validate_identities(graph, snapshot, entities, edges):
    """Keep paid extraction evidence, but do not publish a stale cross-source merge."""
    current = {entity.key: entity for entity in graph.entities()}
    links = identity_links(graph.connection)
    staged = [_source_entity(entity) for entity in entities]
    resolved, mappings, decisions, warnings = {}, {}, {}, []
    for extracted in entities:
        entity = _source_entity(extracted)
        key = links.get(entity.key, entity.key)
        candidate = current.get(key)
        stale = False
        for mention in entity.provenance.get('mentions', []):
            decision = mention.get('decision', {})
            if decision.get('action') != 'reuse' or decision.get('method') == 'same_source':
                continue
            if decision.get('method') == 'identifier':
                donors = ([candidate] if candidate else []) + [other for other in staged
                    if other.provenance.get('document_ref') != mention.get('document_ref')
                    and links.get(other.key, other.key) == key]
                valid = any(shared_identifier(donor, entity.identifiers, entity.type)
                            and not conflicting_identifiers(donor, entity.identifiers) for donor in donors)
            else:
                quoted = decision.get('candidate_evidence', [])
                available = {item['quote'] for item in evidence_entries(candidate)} if candidate else set()
                for other in staged:
                    if links.get(other.key, other.key) == key:
                        available.update(item['quote'] for item in evidence_entries(other)
                                         if item['document_ref'] != mention.get('document_ref'))
                valid = bool(quoted) and all(quote and any(quote in evidence for evidence in available) for quote in quoted)
            if not valid:
                stale = True
                break
        if stale:
            key = 'ent_' + fingerprint([snapshot.ref, snapshot.version, entity.key, 'stale_identity'])[:32]
            reason = 'Saved identity support changed or was removed. Kept this source mention separate for review.'
            decision = {'action': 'unresolved', 'method': 'publication_recheck', 'reason': reason,
                        'requested_id': entity.key, 'previous': entity.provenance.get('resolution', {})}
            decision['id'] = 'resolution_' + fingerprint(decision)
            entity = replace(entity, provenance={**entity.provenance, 'resolution': decision,
                'mentions': [{**mention, 'decision': {**decision, 'previous': mention.get('decision', {})}}
                             for mention in entity.provenance.get('mentions', [])]})
            warnings.append({'id': entity.key, 'replacement_id': key, 'reason': reason})
        elif key != entity.key:
            decision = {'action': 'reuse', 'method': 'operator_review', 'requested_id': entity.key,
                        'canonical_id': key, 'previous': entity.provenance.get('resolution', {})}
            entity = replace(entity, provenance={**entity.provenance, 'resolution': decision})
        mappings[extracted.key] = key
        entity = replace(entity, key=key)
        decisions[key] = entity.provenance.get('resolution', {})
        resolved[key] = merge_entity(resolved[key], entity) if key in resolved else entity
    claims = []
    for edge in edges:
        source, target = mappings.get(edge.source_id, edge.source_id), mappings.get(edge.target_id, edge.target_id)
        if source != edge.source_id or target != edge.target_id:
            edge = replace(edge, source_id=source, target_id=target, provenance={**edge.provenance,
                'identity_decisions': {'source': decisions.get(source, {}), 'target': decisions.get(target, {})}})
        claims.append(edge)
    return list(resolved.values()), claims, warnings
