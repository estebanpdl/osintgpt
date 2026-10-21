'''Indexed, provider-free candidate lookup shared by extraction and discovery.'''

from collections import defaultdict
from dataclasses import asdict
from functools import cached_property

from .records import merge_key
from .recipes import fingerprint
from .identity_storage import canonical_entities
import json


class EntityCatalog:
    def __init__(self, entities):
        self.entities = {entity.key: entity for entity in entities}
        self.names, self.identifiers, self.types = defaultdict(set), defaultdict(set), defaultdict(set)
        for entity in self.entities.values():
            for name in (entity.name, *entity.aliases):
                if merge_key(name):
                    self.names[merge_key(name)].add(entity.key)
            for item in entity.identifiers:
                self.identifiers[merge_key(item['value'])].add(entity.key)
            self.types[merge_key(entity.type)].add(entity.key)

    def search(self, query, *, offset=0, limit=20, entity_type=''):
        key = merge_key(query)
        if query in self.entities:
            ids, mode = {query}, 'id'
        else:
            ids = self.names.get(key, set()) | self.identifiers.get(key, set())
            mode = 'exact'
            if not ids and key:
                ids = {identity for name, values in self.names.items() if key in name for identity in values}
                mode = 'partial' if ids else 'none'
        if entity_type:
            ids = ids & self.types.get(merge_key(entity_type), set())
        ordered = sorted(ids, key=lambda identity: (-self.entities[identity].mentions,
                                                   self.entities[identity].name, identity))
        return [self.entities[identity] for identity in ordered[offset:offset + limit]], mode, len(ordered)

    @cached_property
    def version(self):
        return fingerprint([asdict(self.entities[key]) for key in sorted(self.entities)])


def evidence_entries(entity):
    origins = [entity.provenance, *entity.provenance.get('mentions', []),
               *entity.provenance.get('identity_sources', [])]
    entries = {}
    for origin in origins:
        for support in origin.get('supporting', []):
            for match in support.get('matches', []):
                quote = match.get('source_excerpt') or support.get('quote', '')
                if quote:
                    item = {'quote': quote, 'source_ref': origin.get('source_ref', ''),
                            'document_ref': origin.get('document_ref', ''),
                            'source_version': origin.get('source_version', ''), 'location': match}
                    entries[fingerprint(item)] = item
        for match in origin.get('matches', []):
            if match.get('source_excerpt'):
                item = {'quote': match['source_excerpt'], 'source_ref': origin.get('source_ref', ''),
                        'document_ref': origin.get('document_ref', ''),
                        'source_version': origin.get('source_version', ''), 'location': match}
                entries[fingerprint(item)] = item
    return list(entries.values())


def stored_candidates(graph, query, limit):
    '''Bound SQL discovery before loading evidence; traversal never scans all nodes.'''
    links = dict(graph.connection.execute('SELECT member_key, canonical_key FROM entity_links'))
    query = links.get(query, query)
    base = '''SELECT DISTINCT COALESCE(l.canonical_key,e.key) AS key FROM entities e
              LEFT JOIN entity_links l ON l.member_key=e.key WHERE '''
    modes = [('id', 'COALESCE(l.canonical_key,e.key)=?', [query])]
    key = merge_key(query)
    if key:
        for mode, expression in [('exact', 'graph_key(value)=?'), ('partial', 'instr(graph_key(value),?)>0')]:
            condition = ('EXISTS (SELECT 1 FROM json_each(json_array(e.name)) WHERE ' + expression + ')'
                         ' OR EXISTS (SELECT 1 FROM json_each(e.data,\'$.aliases\') WHERE ' + expression + ')'
                         ' OR EXISTS (SELECT 1 FROM json_each(e.data,\'$.identifiers\') '
                         'WHERE graph_key(json_extract(value,\'$.value\'))=?)')
            modes.append((mode, condition, [key, key, key]))
    for mode, condition, parameters in modes:
        identities = [row[0] for row in graph.connection.execute(base + '(' + condition + ') ORDER BY key LIMIT ?',
                                                               [*parameters, limit])]
        if identities:
            rows = graph.connection.execute('''SELECT e.* FROM entities e
                LEFT JOIN entity_links l ON l.member_key=e.key
                WHERE COALESCE(l.canonical_key,e.key) IN (SELECT value FROM json_each(?)) ORDER BY e.key''',
                (json.dumps(identities),))
            entities = canonical_entities([graph._entity_row(row) for row in rows], links)
            return sorted(entities, key=lambda entity: (-entity.mentions, entity.name, entity.key)), mode
    return [], 'none'


def stored_entities(graph, keys):
    '''Fetch exact IDs and reviewed members once, preserving candidate ID semantics.'''
    keys = list(dict.fromkeys(keys))
    if not keys:
        return []
    owns_transaction = not graph.connection.in_transaction
    if owns_transaction:
        graph.connection.execute('BEGIN')
    try:
        links = dict(graph.connection.execute('SELECT member_key, canonical_key FROM entity_links'))
        wanted = json.dumps(sorted({links.get(key, key) for key in keys}))
        rows = graph.connection.execute('''SELECT e.* FROM entities e WHERE e.key IN (
            SELECT value FROM json_each(?) UNION
            SELECT member_key FROM entity_links WHERE canonical_key IN (SELECT value FROM json_each(?))
            ) ORDER BY e.key''', (wanted, wanted))
        return canonical_entities([graph._entity_row(row) for row in rows], links)
    finally:
        if owns_transaction:
            graph.connection.rollback()
