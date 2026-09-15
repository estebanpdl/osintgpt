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
from .pgvector_schema import INDEX_SUFFIX, ensure_table, prepare

# One table per project rather than a project column: isolation is structural
# here as it is with a file per project, so a forgotten WHERE cannot leak one
# case into another.
TABLE_PREFIX = 'osintgpt_'

# Named explicitly so a term containing it is doubled rather than silently
# escaping the character after it.
LIKE_ESCAPE = chr(92)


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

    def indexing_destination(self, root) -> dict:
        info = self.connection.info
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT current_schema()')
            schema = cursor.fetchone()[0]
        return {
            'host': info.host, 'port': info.port, 'database': info.dbname,
            'schema': schema, 'table': self.table
        }

    def index_inventory(self) -> dict:
        if not self._table_exists():
            return {}
        with self.connection.cursor() as cursor:
            cursor.execute(self._sql(
                'SELECT ref, embedding_model, COUNT(*), COUNT(DISTINCT sequence), '
                'MIN(sequence), MAX(sequence) FROM {} GROUP BY ref, embedding_model'
            ))
            inventory = {}
            for ref, model, n, distinct_n, first, last in cursor.fetchall():
                if (first, last, distinct_n) != (0, n - 1, n):
                    inventory[ref] = None
                elif ref not in inventory or inventory[ref] is not None:
                    inventory.setdefault(ref, {})[model] = n
            return inventory

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def upsert(self, ref, chunks, vectors) -> int:
        try:
            return self._upsert(ref, chunks, vectors)
        except BaseException:
            # A refused replacement must not poison later reads or commit a
            # preceding DELETE when the caller continues with another file.
            self.connection.rollback()
            raise

    def _upsert(
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
                        '  metadata, embedding_model, vector, document_ref'
                        ') VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
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
                            _vector_literal(vector),
                            chunk.document_ref
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
            '       embedding_model, document_ref, 1 - (vector <=> %s) AS score '
            'FROM {} WHERE ' + ' AND '.join(clauses) +
            ' ORDER BY vector <=> %s LIMIT %s'
        )
        target = _vector_literal(vector)

        with self.connection.cursor() as cursor:
            cursor.execute(query, [target] + parameters + [target, top_k])
            rows = cursor.fetchall()

        return [
            SearchResult(chunk=_to_chunk(row), score=float(row[9]))
            for row in rows
        ]

    def match_text(
        self,
        term: str,
        embedding_model: Optional[str] = None,
        limit: int = 100,
        refs: Optional[Iterable[str]] = None
    ) -> List[StoredChunk]:
        if not term or not self._table_exists():
            return []

        # ILIKE folds case per the database collation, which is Unicode-aware
        # on any modern install. LOWER() on both sides would be the same
        # comparison with an extra function call.
        pattern = f'%{_escape_like(term)}%'
        clauses = ['(text ILIKE %s ESCAPE %s OR author ILIKE %s ESCAPE %s)']
        parameters: List[object] = [
            pattern, LIKE_ESCAPE, pattern, LIKE_ESCAPE
        ]

        if embedding_model is not None:
            clauses.append('embedding_model = %s')
            parameters.append(embedding_model)

        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            clauses.append('ref = ANY(%s)')
            parameters.append(wanted)

        query = self._sql(
            'SELECT ref, sequence, text, path, "timestamp", author,'
            '       metadata, embedding_model, document_ref '
            'FROM {} WHERE ' + ' AND '.join(clauses) +
            ' ORDER BY (text ILIKE %s ESCAPE %s) DESC, ref, sequence LIMIT %s'
        )
        # Text matches first: an author match is everything that account
        # wrote, and ahead of them it would fill the limit on its own.
        parameters += [pattern, LIKE_ESCAPE, limit]

        with self.connection.cursor() as cursor:
            cursor.execute(query, parameters)

            return [_to_chunk(row) for row in cursor.fetchall()]

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
                    '       metadata, embedding_model, document_ref '
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
        prepare(self)

    def _ensure_table(self, cursor, dimensions: int) -> None:
        ensure_table(self, cursor, dimensions)

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


def _escape_like(term: str) -> str:
    '''
    A term containing % or _ is ordinary text. Without this a search for `50%`
    matches every row.
    '''
    for character in (LIKE_ESCAPE, '%', '_'):
        term = term.replace(character, LIKE_ESCAPE + character)

    return term


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
        metadata=metadata or {},
        document_ref=row[8]
    )
