'''Turn model proposals into qualified, text-grounded graph claims.'''

from .grounding import GroundingError, string
from .contract import ASSERTIONS, POLARITIES
from .records import Edge, RejectedClaim, entity_origin
from osintgpt.ingestion.quotation_projection import supported_wording

def validate_edge(row, scope, entities, local, registry, allowed):
    if 'polarity' in row:
        raise ValueError('New extraction relationships must use relation_polarity, not the retired polarity key.')
    source, target = string(row, 'source'), string(row, 'target')
    relation = string(row, 'relation')
    if not source or not target or not relation:
        raise ValueError('A relationship requires both endpoints and a relation.')
    quotes, provenance = scope.prove(row.get('evidence'))
    endpoints = []
    for value in (source, target):
        key = local.get(value, value if value in allowed else '')
        entity = entities.get(key) or registry.entities.get(key)
        if entity is None:
            raise ValueError('Relationship endpoint does not identify a supplied or extracted entity.')
        if key not in entities and not entity_origin(entity, scope.document.ref, scope.snapshot.version):
            raise ValueError('Cross-document endpoints need a supported identity declaration.')
        endpoints.append(entity)
    relation_polarity = string(row, 'relation_polarity', 'unknown') or 'unknown'
    assertion = string(row, 'assertion', 'unknown') or 'unknown'
    if relation_polarity not in POLARITIES or assertion not in ASSERTIONS:
        raise ValueError('Invalid relation polarity or assertion status.')
    qualifiers = {}
    for name in ('attributed_to', 'event_time', 'statement_time'):
        value = string(row, name)
        if value and not supported_wording(value, quotes, scope.document):
            raise GroundingError(f'{name} must retain wording from a supporting quotation.')
        qualifiers[name] = value
    provenance['identity_decisions'] = {side: entity.provenance.get('resolution', {})
                                        for side, entity in zip(('source', 'target'), endpoints)}
    return Edge(endpoints[0].name, endpoints[1].name, relation, scope.document.ref, quotes[0],
                provenance, endpoints[0].key, endpoints[1].key, relation_polarity, assertion,
                **qualifiers, evidence_quotes=quotes)


def rejection(scope, window, row, reason):
    return RejectedClaim(scope.document.ref, scope.snapshot.ref, scope.snapshot.version,
                         window, row, str(reason), supplied_ranges=tuple(scope.ranges))
