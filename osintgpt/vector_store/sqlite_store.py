# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: sqlite_store.py
# Description: The default store: one file per project, ranked in process with
#   NumPy. No server, no port, and nothing to install beyond osintgpt itself.
# =================================================================================

# import modules
import json
import sqlite3

# import submodules
import numpy as np

from array import array
from pathlib import Path

# type hints
from typing import Iterable, List, Optional, Sequence, Union

from .base import BaseVectorEngine
from .records import SearchResult, StoredChunk

# Brute force over every vector is milliseconds at 10^4 chunks and seconds at
# 10^5. Past this, the store still answers but a dedicated backend is the
# better answer — the interface is what makes that a configuration change.
BRUTE_FORCE_CEILING = 50_000

SCHEMA = '''
CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY,
    ref             TEXT    NOT NULL,
    sequence        INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    path            TEXT    NOT NULL DEFAULT '',
    timestamp       TEXT    NOT NULL DEFAULT '',
    author          TEXT    NOT NULL DEFAULT '',
    metadata        TEXT    NOT NULL DEFAULT '{}',
    embedding_model TEXT    NOT NULL,
    dimensions      INTEGER NOT NULL,
    vector          BLOB    NOT NULL
);

-- Every search filters by model and most also by document, so both are worth
-- an index even at this scale.
CREATE INDEX IF NOT EXISTS chunks_by_ref ON chunks (ref);
CREATE INDEX IF NOT EXISTS chunks_by_model ON chunks (embedding_model);
'''


# SQLiteVectorStore class
class SQLiteVectorStore(BaseVectorEngine):
    '''
    Vectors in one SQLite file, ranked by NumPy cosine in this process.
    '''
    def __init__(self, path: Union[str, Path]) -> None:
        '''
        Args:
            path (Union[str, Path]): The project's store file. Parent \
                directories are created; ':memory:' works for tests.
        '''
        self.path = Path(path) if str(path) != ':memory:' else path

        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(
            str(self.path), check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

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

        with self.connection:
            # Replace rather than merge: chunk boundaries move when a document
            # changes, so its old chunks describe text that no longer exists.
            self.connection.execute(
                'DELETE FROM chunks WHERE ref = ?', (ref,)
            )
            self.connection.executemany(
                '''
                INSERT INTO chunks (
                    ref, sequence, text, path, timestamp, author, metadata,
                    embedding_model, dimensions, vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
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
                        len(vector),
                        _pack(vector)
                    )
                    for chunk, vector in zip(chunks, vectors)
                ]
            )

        return len(chunks)

    def search(
        self,
        vector: Sequence[float],
        embedding_model: str,
        top_k: int = 10,
        refs: Optional[Iterable[str]] = None
    ) -> List[SearchResult]:
        query = 'SELECT * FROM chunks WHERE embedding_model = ?'
        parameters: List[object] = [embedding_model]

        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            query += f' AND ref IN ({",".join("?" * len(wanted))})'
            parameters += wanted

        rows = self.connection.execute(query, parameters).fetchall()
        if not rows:
            return []

        matrix = np.array(
            [_unpack(row['vector']) for row in rows], dtype=np.float32
        )
        target = np.asarray(vector, dtype=np.float32)

        # One matrix multiply rather than a cosine per row: the same answer,
        # and the difference between milliseconds and seconds at this scale.
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
        # A zero-length vector has no direction, so it matches nothing rather
        # than dividing by zero.
        norms[norms == 0] = np.inf
        scores = (matrix @ target) / norms

        best = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(chunk=_to_chunk(rows[i]), score=float(scores[i]))
            for i in best
        ]

    def delete(self, refs: Iterable[str]) -> int:
        wanted = list(refs)
        if not wanted:
            return 0

        with self.connection:
            cursor = self.connection.execute(
                f'DELETE FROM chunks WHERE ref IN ({",".join("?" * len(wanted))})',
                wanted
            )

        return cursor.rowcount

    def count(self, embedding_model: Optional[str] = None) -> int:
        if embedding_model is None:
            row = self.connection.execute(
                'SELECT COUNT(*) AS total FROM chunks'
            ).fetchone()
        else:
            row = self.connection.execute(
                'SELECT COUNT(*) AS total FROM chunks WHERE embedding_model = ?',
                (embedding_model,)
            ).fetchone()

        return int(row['total'])

    def refs(self, embedding_model: Optional[str] = None) -> List[str]:
        if embedding_model is None:
            rows = self.connection.execute(
                'SELECT DISTINCT ref FROM chunks ORDER BY ref'
            ).fetchall()
        else:
            rows = self.connection.execute(
                'SELECT DISTINCT ref FROM chunks WHERE embedding_model = ? '
                'ORDER BY ref',
                (embedding_model,)
            ).fetchall()

        return [row['ref'] for row in rows]

    def models(self) -> List[str]:
        rows = self.connection.execute(
            'SELECT DISTINCT embedding_model FROM chunks '
            'ORDER BY embedding_model'
        ).fetchall()

        return [row['embedding_model'] for row in rows]

    def purge_other_models(self, keep: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                'DELETE FROM chunks WHERE embedding_model != ?', (keep,)
            )

        return cursor.rowcount

    # chunks of one document, in reading order
    def chunks_for(self, ref: str) -> List[StoredChunk]:
        '''
        Args:
            ref (str): The document.

        Returns:
            List[StoredChunk]: Its chunks, in the order they were stored.
        '''
        rows = self.connection.execute(
            'SELECT * FROM chunks WHERE ref = ? ORDER BY sequence', (ref,)
        ).fetchall()

        return [_to_chunk(row) for row in rows]

    @property
    def is_past_brute_force(self) -> bool:
        '''
        Returns:
            bool: True when the store holds more than brute force is a good \
                answer for. It still works; a dedicated backend is better.
        '''
        return self.count() > BRUTE_FORCE_CEILING


def _pack(vector: Sequence[float]) -> bytes:
    '''
    Vectors are stored as raw float32 rather than JSON: a 1536-dimension
    embedding is ~6 KB packed and ~30 KB as text, and text has to be parsed
    back on every search.
    '''
    return array('f', vector).tobytes()


def _unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _to_chunk(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        ref=row['ref'],
        sequence=row['sequence'],
        text=row['text'],
        embedding_model=row['embedding_model'],
        path=row['path'],
        timestamp=row['timestamp'],
        author=row['author'],
        metadata=json.loads(row['metadata'])
    )
