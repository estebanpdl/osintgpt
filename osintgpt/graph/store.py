'''SQLite storage for persistent entity identities, claims, and extraction audit records.'''

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .records import Edge, Entity, RejectedClaim, merge_entity, merge_key
from .state import GraphState, SCHEMA_VERSION, UNSUPPORTED_SCHEMA, initialize, transaction
from .identity_storage import identity_links, canonical_entities

SCHEMA = '''
CREATE TABLE entities (
    key TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL DEFAULT '',
    mentions INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE edges (
    id INTEGER PRIMARY KEY, source_key TEXT NOT NULL, target_key TEXT NOT NULL,
    relation TEXT NOT NULL, ref TEXT NOT NULL, evidence TEXT NOT NULL,
    source_ref TEXT NOT NULL, data TEXT NOT NULL, assertion_id TEXT NOT NULL UNIQUE,
    UNIQUE (source_key, target_key, relation, ref, evidence, data)
);
CREATE INDEX edges_by_source ON edges (source_key);
CREATE INDEX edges_by_target ON edges (target_key);
CREATE INDEX edges_by_ref ON edges (ref);
CREATE INDEX edges_by_file ON edges (source_ref);
CREATE TABLE entity_sources (
    entity_key TEXT NOT NULL, source_ref TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY (entity_key, source_ref)
);
CREATE TABLE graph_rejections (
    id INTEGER PRIMARY KEY, source_ref TEXT NOT NULL, ref TEXT NOT NULL,
    source_version TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX rejected_by_file ON graph_rejections (source_ref);
'''


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _filter(refs, prefix=''):
    if refs is None:
        return '', []
    wanted = list(refs)
    if not wanted:
        return ' WHERE 0', []
    # One JSON parameter avoids SQLite's variable limit for large source sets.
    selection = 'SELECT value FROM json_each(?)'
    return (f' WHERE {prefix}ref IN ({selection}) OR {prefix}source_ref IN ({selection})',
            [_json(wanted), _json(wanted)])


@dataclass
class GraphStore:
    path: Path
    connection: Optional[sqlite3.Connection] = field(default=None, repr=False)
    read_only: bool = False
    check_same_thread: bool = True

    def __post_init__(self):
        self.path = Path(self.path) if str(self.path) != ':memory:' else self.path
        if isinstance(self.path, Path) and not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = (sqlite3.connect(f'{self.path.resolve().as_uri()}?mode=ro', uri=True,
                                           check_same_thread=self.check_same_thread)
                           if self.read_only else sqlite3.connect(str(self.path), check_same_thread=self.check_same_thread))
        self.connection.row_factory = sqlite3.Row
        self.connection.create_function('graph_key', 1, merge_key, deterministic=True)
        try:
            if self.read_only:
                version = self.connection.execute('PRAGMA user_version').fetchone()[0]
                if version != SCHEMA_VERSION:
                    raise ValueError(UNSUPPORTED_SCHEMA.format(version=version))
            else:
                initialize(self.connection, SCHEMA)
        except BaseException:
            self.connection.close()
            raise
        self.state = GraphState(self.connection, self.path)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def add(self, entities, edges):
        with transaction(self.connection):
            for entity in entities:
                if entity.provenance.get('source_ref'):
                    entity = _source_entity(entity)
                data = asdict(entity)
                old = self.connection.execute('SELECT data FROM entities WHERE key = ?', (entity.key,)).fetchone()
                if old:
                    previous = _entity(json.loads(old['data']))
                    data = asdict(merge_entity(previous, entity))
                if entity.provenance.get('source_ref'):
                    source = _source_entity(entity)
                    self.connection.execute('INSERT OR REPLACE INTO entity_sources VALUES (?, ?, ?)',
                                            (entity.key, entity.provenance['source_ref'], _json(asdict(source))))
                self.connection.execute('''
                    INSERT INTO entities (key, name, type, mentions, data) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET mentions = mentions + excluded.mentions,
                        type = CASE WHEN entities.type = '' THEN excluded.type ELSE entities.type END,
                        data = excluded.data
                ''', (entity.key, entity.name, entity.type, max(entity.mentions, 1), _json(data)))
            stored = 0
            for edge in edges:
                source = edge.source_id or merge_key(edge.source)
                target = edge.target_id or merge_key(edge.target)
                for key, name in ((source, edge.source), (target, edge.target)):
                    self.connection.execute('''
                        INSERT INTO entities (key, name, data) VALUES (?, ?, ?)
                        ON CONFLICT(key) DO NOTHING
                    ''', (key, name.strip(), _json(asdict(Entity(key, name.strip())))))
                stored += self.connection.execute('''
                    INSERT OR IGNORE INTO edges
                        (source_key, target_key, relation, ref, evidence, source_ref, data, assertion_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (source, target, edge.relation, edge.ref, edge.evidence,
                      edge.provenance.get('source_ref', edge.ref), _json(asdict(edge)), edge.assertion_id)).rowcount
            self.state.available()
        return stored

    def record_rejections(self, rejected):
        with transaction(self.connection):
            self.connection.executemany('''
                INSERT INTO graph_rejections (source_ref, ref, source_version, data) VALUES (?, ?, ?, ?)
            ''', [(item.source_ref, item.ref, item.source_version, _json(asdict(item))) for item in rejected])

    def rejections(self, refs=None, *, limit=None):
        clause, parameters = _filter(refs)
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError('Rejection limit must be a positive integer.')
            parameters.append(limit)
        result = []
        for row in self.connection.execute('SELECT data FROM graph_rejections' + clause + ' ORDER BY id'
                                            + (' LIMIT ?' if limit is not None else ''), parameters):
            data = json.loads(row['data'])
            data['corrected_evidence'] = tuple(data.get('corrected_evidence', []))
            result.append(RejectedClaim(**data))
        return result

    def forget(self, refs):
        wanted = list(refs)
        clause, parameters = _filter(wanted)
        with transaction(self.connection):
            affected = {key for row in self.connection.execute(
                'SELECT source_key, target_key FROM edges' + clause, parameters) for key in row}
            removed = self.connection.execute('DELETE FROM edges' + clause, parameters).rowcount
            if wanted:
                placeholders = ','.join('?' for ref in wanted)
                affected.update(row[0] for row in self.connection.execute(
                    f'SELECT entity_key FROM entity_sources WHERE source_ref IN ({placeholders})', wanted))
                self.connection.execute(f'DELETE FROM entity_sources WHERE source_ref IN ({placeholders})', wanted)
                for key in affected:
                    self._refresh_entity(key)
                self.connection.execute('DELETE FROM graph_documents WHERE ref IN (' +
                                        ','.join('?' for ref in wanted) + ')', wanted)
            self.connection.executemany('''DELETE FROM entities WHERE key = ?
                AND key NOT IN (SELECT entity_key FROM entity_sources)
                AND key NOT IN (SELECT source_key FROM edges UNION SELECT target_key FROM edges)''',
                [(key,) for key in affected])
        return removed

    def _refresh_entity(self, key):
        contributions = [_entity(json.loads(row[0])) for row in self.connection.execute(
            'SELECT data FROM entity_sources WHERE entity_key = ? ORDER BY source_ref', (key,))]
        if contributions:
            entity = contributions[0]
            for contribution in contributions[1:]:
                entity = merge_entity(entity, contribution)
            self.connection.execute('UPDATE entities SET name = ?, type = ?, mentions = ?, data = ? WHERE key = ?',
                                    (entity.name, entity.type, sum(e.mentions for e in contributions),
                                     _json(asdict(entity)), key))

    def replace_source(self, snapshot, entities, edges, quality_gaps, recipe, recipe_data):
        from .checkpoints import Checkpoints
        with transaction(self.connection):
            removed = self.forget([snapshot.ref])
            self.add(entities, edges)
            self.state.completed(snapshot.ref, snapshot.fingerprint, snapshot.version,
                                 quality_gaps, recipe, recipe_data)
            Checkpoints(self).discard(snapshot.ref)
        return removed

    def source_refs(self):
        return {row[0] for row in self.connection.execute('''
            SELECT ref FROM graph_documents UNION SELECT source_ref FROM edges
            UNION SELECT source_ref FROM entity_sources UNION SELECT source_ref FROM graph_jobs
        ''')}

    def edges(self, refs=None):
        clause, parameters = _filter(refs, 'e.')
        return [self._edge(row) for row in self._edge_rows(clause, parameters, order='e.ref, e.id')]

    def _edge_rows(self, clause, parameters, limit=None, order='e.id'):
        if limit is not None:
            parameters = [*parameters, limit]
        return self.connection.execute('''
            SELECT e.id, e.data, e.assertion_id, COALESCE(s.name, e.source_key) AS source,
                   COALESCE(t.name, e.target_key) AS target,
                   COALESCE(sl.canonical_key, e.source_key) AS canonical_source,
                   COALESCE(tl.canonical_key, e.target_key) AS canonical_target,
                   COALESCE(sl.decision_id, CASE WHEN json_extract(s.data,'$.provenance.resolution.method')='operator_split'
                       THEN json_extract(s.data,'$.provenance.resolution.id') END) AS source_review,
                   COALESCE(tl.decision_id, CASE WHEN json_extract(t.data,'$.provenance.resolution.method')='operator_split'
                       THEN json_extract(t.data,'$.provenance.resolution.id') END) AS target_review
            FROM edges e LEFT JOIN entities s ON s.key = e.source_key
            LEFT JOIN entities t ON t.key = e.target_key
            LEFT JOIN entity_links sl ON sl.member_key = e.source_key
            LEFT JOIN entity_links tl ON tl.member_key = e.target_key
        ''' + clause + ' ORDER BY ' + order + (' LIMIT ?' if limit is not None else ''), parameters)

    @staticmethod
    def _edge(row):
        data = json.loads(row['data'])
        data.update(source=row['source'], target=row['target'], assertion_id=row['assertion_id'],
                    source_id=row['canonical_source'] if data.get('source_id') or row['source_review'] else '',
                    target_id=row['canonical_target'] if data.get('target_id') or row['target_review'] else '',
                    evidence_quotes=tuple(data.get('evidence_quotes', [])))
        if row['source_review'] or row['target_review']:
            data['provenance'] = {**data.get('provenance', {}), 'identity_mapping': 'operator_reviewed'}
        for side in ('source', 'target'):
            if row[side + '_review']:
                decisions = data.setdefault('provenance', {}).setdefault('identity_decisions', {})
                decisions[side] = {'action': 'reuse', 'method': 'operator_review', 'id': row[side + '_review'],
                                   'previous': decisions.get(side, {})}
        edge = Edge.from_dict(data)
        # Entity display-name changes must not rename an existing assertion.
        object.__setattr__(edge, 'assertion_id', row['assertion_id'])
        return edge

    def adjacent(self, entity_id, direction='connectivity', refs=None, limit=1001):
        return [edge for _, edge in self.adjacent_page(entity_id, direction, refs, limit=limit)]

    def adjacent_page(self, entity_id, direction='connectivity', refs=None, *, after=0, limit=1001):
        '''Stable keyset pagination over the same canonical, direction/source-aware adjacency.'''
        if direction not in ('incoming', 'outgoing', 'connectivity'):
            raise ValueError('Direction must be incoming, outgoing, or connectivity.')
        links = identity_links(self.connection)
        entity_id = links.get(entity_id, entity_id)
        members = [entity_id, *(member for member, canonical in links.items() if canonical == entity_id)]
        fields = ('source_key', 'target_key')
        fields = fields if direction == 'connectivity' else (fields[0 if direction == 'outgoing' else 1],)
        clause, parameters = _filter(refs, 'e.')
        restriction = ' AND (' + clause.removeprefix(' WHERE ') + ')' if clause else ''
        rows = {}
        for column in fields:
            for row in self._edge_rows(f' WHERE e.{column} IN (SELECT value FROM json_each(?)) AND e.id > ?' + restriction,
                                       [_json(members), after, *parameters], limit=limit):
                rows[row['id']] = row
        return [(key, self._edge(rows[key])) for key in sorted(rows)[:limit]]

    def assertion(self, assertion_id):
        row = self._edge_rows(' WHERE e.assertion_id = ?', [assertion_id], limit=1).fetchone()
        return self._edge(row) if row else None

    def relationship_counts(self, entity_id, refs=None):
        links = identity_links(self.connection)
        entity_id = links.get(entity_id, entity_id)
        members = [entity_id, *(member for member, canonical in links.items() if canonical == entity_id)]
        clause, parameters = _filter(refs, 'e.')
        restriction = ' AND (' + clause.removeprefix(' WHERE ') + ')' if clause else ''
        def count(scope):
            return self.connection.execute('''SELECT COUNT(*), COUNT(DISTINCT CASE
                WHEN COALESCE(sl.canonical_key,e.source_key) = ?
                THEN NULLIF(COALESCE(tl.canonical_key,e.target_key), ?)
                ELSE NULLIF(COALESCE(sl.canonical_key,e.source_key), ?) END)
                FROM edges e LEFT JOIN entity_links sl ON sl.member_key = e.source_key
                LEFT JOIN entity_links tl ON tl.member_key = e.target_key
                WHERE (source_key IN (SELECT value FROM json_each(?))
                    OR target_key IN (SELECT value FROM json_each(?)))'''
                + (restriction if scope else ''),
                [entity_id, entity_id, entity_id, _json(members), _json(members),
                 *(parameters if scope else [])]).fetchone()
        total = count(False)
        selected = total if refs is None else count(True)
        return {'all_sources': total[0], 'selected_sources': selected[0],
                'distinct_neighbors_all_sources': total[1], 'distinct_neighbors_selected_sources': selected[1]}

    def between(self, source_id, target_id, limit=61):
        clause = (' WHERE (COALESCE(sl.canonical_key,e.source_key) = ? AND COALESCE(tl.canonical_key,e.target_key) = ?)'
                  ' OR (COALESCE(sl.canonical_key,e.source_key) = ? AND COALESCE(tl.canonical_key,e.target_key) = ?)')
        return [self._edge(row) for row in self._edge_rows(
            clause, [source_id, target_id, target_id, source_id], limit=limit)]

    def entity_candidates(self, query, limit=21):
        from .entity_catalog import stored_candidates
        return stored_candidates(self, query, limit)

    def entities_by_ids(self, keys):
        from .entity_catalog import stored_entities
        return stored_entities(self, keys)

    @staticmethod
    def _entity_row(row):
        data = json.loads(row['data'])
        data.update(key=row['key'], name=row['name'], type=row['type'], mentions=row['mentions'])
        return _entity(data)

    def entities(self):
        result = []
        for row in self.connection.execute('SELECT * FROM entities ORDER BY mentions DESC, name'):
            result.append(self._entity_row(row))
        return canonical_entities(result, identity_links(self.connection))

    def refs(self):
        return [row['ref'] for row in self.connection.execute(
            'SELECT ref FROM graph_documents UNION SELECT ref FROM edges ORDER BY ref')]

    @property
    def is_built(self):
        return self.state.read()['available']

    @property
    def edge_count(self):
        return self.connection.execute('SELECT COUNT(*) FROM edges').fetchone()[0]

    @property
    def entity_count(self):
        return self.connection.execute('SELECT COUNT(DISTINCT COALESCE(l.canonical_key,e.key)) '
                                       'FROM entities e LEFT JOIN entity_links l ON l.member_key=e.key').fetchone()[0]


def graph_for(project):
    return GraphStore(Path(project.paths.root) / 'graph.sqlite')


def _entity(data):
    return Entity(**{**data, 'aliases': tuple(data.get('aliases', [])),
                     'identifiers': tuple(data.get('identifiers', []))})


def _source_entity(entity):
    from dataclasses import replace
    ref = entity.provenance['source_ref']
    version = entity.provenance.get('source_version', '')
    def current(origin):
        return origin.get('source_ref') == ref and origin.get('source_version') == version
    aliases = [item for item in entity.provenance.get('alias_evidence', []) if current(item['provenance'])]
    origins = [item for item in entity.provenance.get('identity_sources', []) if current(item)]
    mentions = [item for item in entity.provenance.get('mentions', []) if current(item)]
    return replace(entity, name=entity.provenance.get('source_name', entity.name),
                   aliases=tuple(item['name'] for item in aliases),
                   identifiers=tuple(item for item in entity.identifiers if current(item.get('provenance', {}))),
                   provenance={**entity.provenance, 'identity_sources': origins, 'alias_evidence': aliases,
                               'mentions': mentions})
