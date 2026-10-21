'''Source-backed identity decisions; semantic equivalence remains model judgment.'''

from .grounding import GroundingError, string
from .records import entity_origin, merge_key
from .recipes import fingerprint
from osintgpt.ingestion.quotation_projection import comparable, is_markdown

IDENTITY_POLICY = 1


def shared_identifier(entity, identifiers, entity_type):
    if not entity.type or not entity_type or merge_key(entity.type) != merge_key(entity_type):
        return False
    known = {(merge_key(item['issuer']), item['value']) for item in entity.identifiers}
    return any((merge_key(item['issuer']), item['value']) in known for item in identifiers)


def conflicting_identifiers(entity, identifiers):
    for issuer in {merge_key(item['issuer']) for item in identifiers}:
        known = {item['value'] for item in entity.identifiers if merge_key(item['issuer']) == issuer}
        proposed = {item['value'] for item in identifiers if merge_key(item['issuer']) == issuer}
        if known and proposed and not known & proposed:
            return True
    return False


def decide(registry, row, scope, allowed, identifiers, entity_type, name):
    requested = string(row, 'existing_id', string(row, 'id') if string(row, 'id').startswith('ent_') else '')
    existing = registry.entities.get(requested)
    decision = {'action': 'create', 'method': 'new', 'reason': 'New sourced mention.',
                'requested_id': requested, 'candidate_ids': sorted(allowed), 'policy_version': IDENTITY_POLICY,
                'document_ref': scope.document.ref, 'source_version': scope.snapshot.version,
                'model': getattr(registry, 'model', {})}
    if requested:
        try:
            if requested not in allowed or existing is None:
                raise GroundingError('Requested identity was not supplied by candidate lookup.')
            if conflicting_identifiers(existing, identifiers):
                raise GroundingError('Source-supported identifiers conflict with this candidate.')
            if entity_origin(existing, scope.document.ref, scope.snapshot.version):
                decision.update(action='reuse', method='same_source', reason='Supplied identity from this source version.')
            elif shared_identifier(existing, identifiers, entity_type):
                decision.update(action='reuse', method='identifier', reason='Matching supported issuer and identifier.')
            else:
                resolution = row.get('resolution', {})
                if not isinstance(resolution, dict) or resolution.get('action') != 'reuse':
                    raise GroundingError('Cross-document identity needs supported contextual resolution or a matching identifier.')
                if resolution.get('method') not in {'context', 'explicit_alias'} or not string(resolution, 'reason'):
                    raise GroundingError('Contextual reuse requires a method and an evidence-based explanation.')
                if resolution.get('type_compatible') is not True or not string(resolution, 'type_reason'):
                    raise GroundingError('Contextual reuse requires explicit, explained type compatibility.')
                current, provenance = scope.prove(resolution.get('evidence'))
                cited = resolution.get('candidate_evidence', [])
                supplied = registry.candidate_evidence.get(requested, set())
                if not cited or any(quote not in supplied for quote in cited):
                    raise GroundingError('Read candidate evidence through entity lookup before citing it for reuse.')
                markdown = is_markdown(scope.document)
                if not any(comparable(name, markdown=markdown) in comparable(q, markdown=markdown) and
                           comparable(q, markdown=markdown) != comparable(name, markdown=markdown) for q in current):
                    raise GroundingError('Identity context must identify this mention with more than its bare name.')
                if not any(comparable(q, markdown=True) != comparable(existing.name, markdown=True) for q in cited):
                    raise GroundingError('A candidate name alone does not justify contextual reuse.')
                decision.update(action='reuse', method=resolution['method'], reason=resolution['reason'],
                                evidence=provenance, candidate_evidence=cited,
                                type_compatible=True, type_reason=resolution['type_reason'],
                                source_type=entity_type, candidate_type=existing.type,
                                interpretation='model_proposal', reviewable=True)
        except (GroundingError, ValueError, TypeError) as error:
            decision.update(action='unresolved', method='unresolved', reason=str(error))
            existing = None
    elif row.get('resolution', {}).get('action') == 'unresolved':
        decision.update(action='unresolved', method='unresolved', reason=row['resolution'].get('reason') or 'Ambiguous identity.')
    elif identifiers:
        candidates = [entity for entity in registry.entities.values()
                      if shared_identifier(entity, identifiers, entity_type) and not conflicting_identifiers(entity, identifiers)]
        decision['candidate_ids'] = [entity.key for entity in candidates]
        if len(candidates) == 1:
            existing = candidates[0]
            decision.update(action='reuse', method='identifier', reason='Unique matching supported issuer and identifier.')
        elif len(candidates) > 1:
            decision.update(action='unresolved', method='unresolved', reason='Several identities share this identifier.')
    decision['id'] = 'resolution_' + fingerprint(decision)
    return existing, decision
