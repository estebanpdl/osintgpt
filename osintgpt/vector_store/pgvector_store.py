# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: pgvector_store.py
# Description: Postgres behind the same interface as the default store, for
#   operators who already run one. Optional, and never the thing an error
#   message tells someone to go set up.
# =================================================================================

# import modules
import json

# type hints
from typing import Iterable, List, Optional, Sequence, Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

from .base import BaseVectorEngine
from .records import SearchResult, StoredChunk

# One table per project rather than a project column: isolation is structural
# here as it is with a file per project, so a forgotten WHERE cannot leak one
# case into another.
TABLE_PREFIX = 'osintgpt_'

# HNSW is what makes pgvector worth reaching for; without an index this is a
# sequential scan the default store already does better. Built on the cosine
# operator class, matching the distance every backend reports.
INDEX_SUFFIX = '_vector_idx'


# PgVectorStore class
class PgVectorStore(BaseVectorEngine):
    '''
    Vectors in Postgres with pgvector, behind the interface the default store
    satisfies.
    '''
    def __init__(
        self,
        config: Union[Settings, str],
        collection: str = 'default',
        connection=None
    ) -> None:
        '''
        Args:
            config (Union[Settings, str]): Settings carrying `postgres_dsn`, \
                or a path to a .env file (deprecated).
            collection (str): Project identifier. Becomes the table name, so \
                two projects on one server stay separate.
            connection: An open psycopg connection, for tests and for reusing \
                one. Opened from settings when not given.

        Raises:
            ImportError: If the optional Postgres packages are missing.
            MissingEnvironmentVariableError: If no DSN is configured.
        '''
        self.settings = resolve_settings(config)
        self.collection = collection
        self.table = f'{TABLE_PREFIX}{_identifier(collection)}'

        self.psycopg, self.register_vector = _import_drivers()

        if connection is None:
            if not self.settings.postgres_dsn:
                raise MissingEnvironmentVariableError(
                    'POSTGRES_DSN',
                    hint='a Postgres backend needs a connection string, for '
                         'example postgresql://user:password@host:5432/dbname'
                )
            self.connection = self.psycopg.connect(self.settings.postgres_dsn)
        else:
            self.connection = connection

        self._prepare()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def upsert(
        self,
        ref: str,
        chunks: Sequence[StoredChunk],
        vectors: Sequence[Sequence[float]]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f'{ref}: {len(chunks)} chunks and {len(vectors)} vectors. '
                'They are matched by position, so a mismatch would attach '
                'text to the wrong vector.'
            )

        with self.connection.cursor() as cursor:
            # One transaction: a crash between the delete and the insert would
            # otherwise leave a document indexed as nothing.
            if self._table_exists():
                cursor.execute(
                    self._sql('DELETE FROM {} WHERE ref = %s'), (ref,)
                )

            if chunks:
                self._ensure_table(cursor, len(vectors[0]))
                cursor.executemany(
                    self._sql(
                        'INSERT INTO {} ('
                        '  ref, sequence, text, path, "timestamp", author,'
                        '  metadata, embedding_model, vector'
                        ') VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'
                    ),
                    [
                        (
                            chunk.ref,
                            chunk.sequence,
                            chunk.text,
                            chunk.path,
                            chunk.timestamp,
                            chunk.author,
                            json.dumps(chunk.metadata, ensure_ascii=False),
                            chunk.embedding_model,
                            _vector_literal(vector)
                        )
                        for chunk, vector in zip(chunks, vectors)
                    ]
                )

        self.connection.commit()

        return len(chunks)

    def search(
        self,
        vector: Sequence[float],
        embedding_model: str,
        top_k: int = 10,
        refs: Optional[Iterable[str]] = None
    ) -> List[SearchResult]:
        if not self._table_exists():
            return []

        clauses = ['embedding_model = %s']
        parameters: List[object] = [embedding_model]

        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            clauses.append('ref = ANY(%s)')
            parameters.append(wanted)

        # `<=>` is cosine distance, so similarity is one minus it — the same
        # number every other backend reports.
        query = self._sql(
            'SELECT ref, sequence, text, path, "timestamp", author, metadata,'
            '       embedding_model, 1 - (vector <=> %s) AS score '
            'FROM {} WHERE ' + ' AND '.join(clauses) +
            ' ORDER BY vector <=> %s LIMIT %s'
        )
        target = _vector_literal(vector)

        with self.connection.cursor() as cursor:
            cursor.execute(query, [target] + parameters + [target, top_k])
            rows = cursor.fetchall()

        return [
            SearchResult(chunk=_to_chunk(row), score=float(row[8]))
            for row in rows
        ]

    def delete(self, refs: Iterable[str]) -> int:
        wanted = list(refs)
        if not wanted or not self._table_exists():
            return 0

        with self.connection.cursor() as cursor:
            cursor.execute(
                self._sql('DELETE FROM {} WHERE ref = ANY(%s)'), (wanted,)
            )
            removed = cursor.rowcount
        self.connection.commit()

        return removed

    def count(self, embedding_model: Optional[str] = None) -> int:
        if not self._table_exists():
            return 0

        with self.connection.cursor() as cursor:
            if embedding_model is None:
                cursor.execute(self._sql('SELECT COUNT(*) FROM {}'))
            else:
                cursor.execute(
                    self._sql(
                        'SELECT COUNT(*) FROM {} WHERE embedding_model = %s'
                    ),
                    (embedding_model,)
                )

            return int(cursor.fetchone()[0])

    def refs(self, embedding_model: Optional[str] = None) -> List[str]:
        return self._distinct('ref', embedding_model)

    def models(self) -> List[str]:
        return self._distinct('embedding_model', None)

    def purge_other_models(self, keep: str) -> int:
        if not self._table_exists():
            return 0

        with self.connection.cursor() as cursor:
            cursor.execute(
                self._sql('DELETE FROM {} WHERE embedding_model <> %s'),
                (keep,)
            )
            removed = cursor.rowcount
        self.connection.commit()

        return removed

    # chunks of one document, in reading order
    def chunks_for(self, ref: str) -> List[StoredChunk]:
        '''
        Args:
            ref (str): The document.

        Returns:
            List[StoredChunk]: Its chunks, in the order they were stored.
        '''
        if not self._table_exists():
            return []

        with self.connection.cursor() as cursor:
            cursor.execute(
                self._sql(
                    'SELECT ref, sequence, text, path, "timestamp", author,'
                    '       metadata, embedding_model '
                    'FROM {} WHERE ref = %s ORDER BY sequence'
                ),
                (ref,)
            )

            return [_to_chunk(row) for row in cursor.fetchall()]

    def _distinct(
        self, column: str, embedding_model: Optional[str]
    ) -> List[str]:
        '''
        SQL has DISTINCT, so this is one indexed query rather than a walk over
        every row — the one place this backend is plainly better than Qdrant.
        '''
        if not self._table_exists():
            return []

        with self.connection.cursor() as cursor:
            if embedding_model is None:
                cursor.execute(
                    self._sql(f'SELECT DISTINCT {column} FROM {{}} '
                              f'ORDER BY {column}')
                )
            else:
                cursor.execute(
                    self._sql(f'SELECT DISTINCT {column} FROM {{}} '
                              f'WHERE embedding_model = %s ORDER BY {column}'),
                    (embedding_model,)
                )

            return [row[0] for row in cursor.fetchall()]

    def _prepare(self) -> None:
        '''
        The extension has to exist before a vector column can. Creating it
        needs privileges an operator may not have granted, so the failure says
        what to run rather than what went wrong.
        '''
        try:
            with self.connection.cursor() as cursor:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
            self.connection.commit()
        except Exception as error:
            self.connection.rollback()
            raise RuntimeError(
                'the pgvector extension is not available: ask a database '
                'owner to run CREATE EXTENSION vector on this database'
            ) from error

        self.register_vector(self.connection)

    def _ensure_table(self, cursor, dimensions: int) -> None:
        '''
        Create on first write, when the vector size is finally known.
        '''
        cursor.execute(self._sql(
            'CREATE TABLE IF NOT EXISTS {} ('
            '  id              bigserial PRIMARY KEY,'
            '  ref             text NOT NULL,'
            '  sequence        integer NOT NULL,'
            '  text            text NOT NULL,'
            '  path            text NOT NULL DEFAULT \'\','
            '  "timestamp"     text NOT NULL DEFAULT \'\','
            '  author          text NOT NULL DEFAULT \'\','
            '  metadata        jsonb NOT NULL DEFAULT \'{{}}\'::jsonb,'
            '  embedding_model text NOT NULL,'
            f'  vector          vector({dimensions}) NOT NULL'
            ')'
        ))
        cursor.execute(self._sql(
            'CREATE INDEX IF NOT EXISTS ' + self.table + '_ref_idx '
            'ON {} (ref)'
        ))
        cursor.execute(self._sql(
            'CREATE INDEX IF NOT EXISTS ' + self.table + '_model_idx '
            'ON {} (embedding_model)'
        ))
        cursor.execute(self._sql(
            'CREATE INDEX IF NOT EXISTS ' + self.table + INDEX_SUFFIX + ' '
            'ON {} USING hnsw (vector vector_cosine_ops)'
        ))

    def _table_exists(self) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT to_regclass(%s)', (self.table,))

            return cursor.fetchone()[0] is not None

    def _sql(self, template: str):
        '''
        The table name is derived from a project slug, so it is composed as an
        identifier rather than interpolated into the string.
        '''
        from psycopg import sql

        return sql.SQL(template).format(sql.Identifier(self.table))


def _import_drivers():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as error:
        raise ImportError(
            'the postgres backend needs psycopg and pgvector: '
            'pip install osintgpt[postgres]'
        ) from error

    return psycopg, register_vector


def _identifier(name: str) -> str:
    '''
    A slug reduced to what a table name may hold. Composed as an identifier
    anyway; this keeps the result readable rather than making it safe.
    '''
    cleaned = ''.join(
        character if character.isalnum() else '_'
        for character in name.lower()
    ).strip('_')

    return cleaned or 'default'


def _vector_literal(vector: Sequence[float]) -> str:
    '''
    pgvector accepts its own text form, which avoids depending on a numpy
    adapter being registered for every connection a caller might pass in.
    '''
    return '[' + ','.join(repr(float(value)) for value in vector) + ']'


def _to_chunk(row) -> StoredChunk:
    metadata = row[6]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)

    return StoredChunk(
        ref=row[0],
        sequence=int(row[1]),
        text=row[2],
        embedding_model=row[7],
        path=row[3],
        timestamp=row[4],
        author=row[5],
        metadata=metadata or {}
    )
