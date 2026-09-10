# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: store.py
# Description: Entities and edges in one SQLite file per project. An edge is a
#   sourced claim, so it cannot exist without the document and sentence that
#   assert it.
# =================================================================================

# import modules
import sqlite3
import unicodedata

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Iterable, List, Optional, Sequence

# The graph is relational data and lives in SQLite whatever backend holds the
# vectors. A vector database has no relational query and traversing a graph
# through its payloads would be worse at every size; keeping the two separate
# means choosing Qdrant for scale does not change how the graph works.
SCHEMA = '''
CREATE TABLE IF NOT EXISTS entities (
    key      TEXT PRIMARY KEY,
    name     TEXT NOT NULL,
    type     TEXT NOT NULL DEFAULT '',
    mentions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY,
    source_key   TEXT NOT NULL,
    target_key   TEXT NOT NULL,
    relation     TEXT NOT NULL,
    -- An edge without both of these is a model assertion rather than a
    -- sourced claim, which is the distinction this table exists to keep.
    ref          TEXT NOT NULL,
    evidence     TEXT NOT NULL,
    UNIQUE (source_key, target_key, relation, ref, evidence)
);

CREATE INDEX IF NOT EXISTS edges_by_source ON edges (source_key);
CREATE INDEX IF NOT EXISTS edges_by_target ON edges (target_key);
CREATE INDEX IF NOT EXISTS edges_by_ref ON edges (ref);
'''


# Entity class
@dataclass(frozen=True)
class Entity:
    '''
    A named thing, and the form it was first seen in.
    '''
    key: str
    name: str
    # Whatever the model called it. Not validated against a list: an
    # enumeration of types would be written in one language and would refuse
    # the categories another corpus needs.
    type: str = ''
    mentions: int = 0


# Edge class
@dataclass(frozen=True)
class Edge:
    '''
    One claim that two entities are related, and where it was asserted.
    '''
    source: str
    target: str
    relation: str
    ref: str
    evidence: str

    @property
    def sentence(self) -> str:
        return f'{self.source} — {self.relation} → {self.target}'


# normalize a name for matching
def merge_key(name: str) -> str:
    '''
    The form two spellings of one name have to share to be the same node.

    Case is folded and surrounding punctuation dropped, but accents are kept:
    `Bogota` and `Bogotá` are different strings and an analyst may mean
    either, so merging them would be a decision this layer is not entitled to
    make.

    Args:
        name (str): A name as written.

    Returns:
        str: Its merge key.
    '''
    cleaned = unicodedata.normalize('NFC', (name or '').strip())
    cleaned = cleaned.strip('.,;:!?"\'()[]{}«»„“”‘’')

    return ' '.join(cleaned.casefold().split())


# GraphStore class
@dataclass
class GraphStore:
    '''
    A project's entities and edges.
    '''
    path: Path
    connection: Optional[sqlite3.Connection] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = (
            Path(self.path) if str(self.path) != ':memory:' else self.path
        )
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    # record what one document asserted
    def add(self, entities: Sequence[Entity], edges: Sequence[Edge]) -> int:
        '''
        Store entities and edges from one extraction.

        An entity already present keeps its first-seen name and accumulates
        mentions: the first spelling is as good as any, and rewriting it on
        every pass would make the graph depend on document order.

        Args:
            entities (Sequence[Entity]): Entities found.
            edges (Sequence[Edge]): Claims found.

        Returns:
            int: Edges stored, excluding exact duplicates.
        '''
        with self.connection:
            for entity in entities:
                self.connection.execute(
                    '''
                    INSERT INTO entities (key, name, type, mentions)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        mentions = mentions + excluded.mentions,
                        type = CASE WHEN entities.type = ''
                                    THEN excluded.type ELSE entities.type END
                    ''',
                    (entity.key, entity.name, entity.type,
                     max(entity.mentions, 1))
                )

            stored = 0
            for edge in edges:
                # An edge endpoint is an entity by definition. Recording it
                # keeps its written form: without this a name the extraction
                # did not list separately comes back as its merge key, which
                # is case-folded, and an analyst reads `beta ltd` where the
                # document said `Beta Ltd`. Mentions stay at zero — appearing
                # in a relationship is not the same as being counted.
                for endpoint in (edge.source, edge.target):
                    self.connection.execute(
                        '''
                        INSERT INTO entities (key, name, type, mentions)
                        VALUES (?, ?, '', 0)
                        ON CONFLICT(key) DO NOTHING
                        ''',
                        (merge_key(endpoint), endpoint.strip())
                    )

                cursor = self.connection.execute(
                    '''
                    INSERT OR IGNORE INTO edges
                        (source_key, target_key, relation, ref, evidence)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (merge_key(edge.source), merge_key(edge.target),
                     edge.relation, edge.ref, edge.evidence)
                )
                stored += cursor.rowcount

        return stored

    # drop everything one document contributed
    def forget(self, refs: Iterable[str]) -> int:
        '''
        Remove the edges a document asserted.

        Entities are left alone: another document may still mention them, and
        a name with no edges costs a row rather than a wrong answer.

        Args:
            refs (Iterable[str]): Documents to forget.

        Returns:
            int: Edges removed.
        '''
        wanted = list(refs)
        if not wanted:
            return 0

        with self.connection:
            cursor = self.connection.execute(
                f'DELETE FROM edges WHERE ref IN ({",".join("?" * len(wanted))})',
                wanted
            )

        return cursor.rowcount

    # every edge, with its endpoint names resolved
    def edges(self, refs: Optional[Iterable[str]] = None) -> List[Edge]:
        '''
        Load the whole edge set.

        One query, then traversal in Python. A project graph is small enough
        that a recursive query would cost more complexity than it saves.

        Args:
            refs (Iterable[str], optional): Restrict to these documents.

        Returns:
            List[Edge]: Edges with names rather than keys.
        '''
        query = '''
            SELECT e.relation, e.ref, e.evidence,
                   COALESCE(s.name, e.source_key) AS source,
                   COALESCE(t.name, e.target_key) AS target
            FROM edges e
            LEFT JOIN entities s ON s.key = e.source_key
            LEFT JOIN entities t ON t.key = e.target_key
        '''
        parameters: List[object] = []

        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            query += f' WHERE e.ref IN ({",".join("?" * len(wanted))})'
            parameters += wanted

        query += ' ORDER BY e.ref, e.id'

        return [
            Edge(source=row['source'], target=row['target'],
                 relation=row['relation'], ref=row['ref'],
                 evidence=row['evidence'])
            for row in self.connection.execute(query, parameters).fetchall()
        ]

    def entities(self) -> List[Entity]:
        '''
        Returns:
            List[Entity]: Every entity, most mentioned first.
        '''
        rows = self.connection.execute(
            'SELECT * FROM entities ORDER BY mentions DESC, name'
        ).fetchall()

        return [
            Entity(key=row['key'], name=row['name'], type=row['type'],
                   mentions=row['mentions'])
            for row in rows
        ]

    # documents that have contributed
    def refs(self) -> List[str]:
        '''
        Returns:
            List[str]: Documents with at least one edge, sorted. What tells \
                an incremental pass which documents it has already read.
        '''
        rows = self.connection.execute(
            'SELECT DISTINCT ref FROM edges ORDER BY ref'
        ).fetchall()

        return [row['ref'] for row in rows]

    @property
    def is_built(self) -> bool:
        '''
        Returns:
            bool: True once anything has been extracted. An incremental pass \
                checks this and does nothing when it is False, because a \
                graph built as a side effect is one nobody chose to pay for.
        '''
        return self.edge_count > 0

    @property
    def edge_count(self) -> int:
        return int(
            self.connection.execute('SELECT COUNT(*) AS n FROM edges')
            .fetchone()['n']
        )

    @property
    def entity_count(self) -> int:
        return int(
            self.connection.execute('SELECT COUNT(*) AS n FROM entities')
            .fetchone()['n']
        )


# open a project's graph
def graph_for(project) -> GraphStore:
    '''
    Args:
        project (Project): The project whose graph to open.

    Returns:
        GraphStore: The graph, created empty if it does not exist.
    '''
    return GraphStore(path=Path(project.paths.root) / 'graph.sqlite')
