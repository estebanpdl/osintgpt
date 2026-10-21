'''Reversible identity mappings keep original mentions and assertion IDs intact.'''

import json
from dataclasses import replace

from .records import merge_entity

SCHEMA = '''
CREATE TABLE entity_links (
    member_key TEXT PRIMARY KEY, canonical_key TEXT NOT NULL, decision_id TEXT NOT NULL
);
CREATE INDEX links_by_canonical ON entity_links (canonical_key);
CREATE TABLE identity_decisions (id TEXT PRIMARY KEY, data TEXT NOT NULL);
'''


def identity_links(connection):
    return {row[0]: row[1] for row in connection.execute('SELECT member_key, canonical_key FROM entity_links')}


def canonical_entities(entities, links):
    if not links:
        return entities
    result = {}
    for entity in entities:
        key = links.get(entity.key, entity.key)
        provenance = {**entity.provenance, 'member_ids': [entity.key]}
        entity = replace(entity, key=key, provenance=provenance)
        if key in result:
            previous = result[key]
            merged = merge_entity(previous, entity)
            provenance = {**merged.provenance,
                          'member_ids': [*previous.provenance['member_ids'], *entity.provenance['member_ids']]}
            entity = replace(merged, mentions=previous.mentions + entity.mentions, provenance=provenance)
        result[key] = entity
    return list(result.values())


def resolutions(connection):
    return [json.loads(row[0]) for row in connection.execute('SELECT data FROM identity_decisions ORDER BY rowid')]
