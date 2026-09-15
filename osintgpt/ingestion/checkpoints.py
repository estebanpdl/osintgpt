'''Durable, private staging for an indexing pass, separate from search results.'''

import hashlib
import json
import math
import os
import sqlite3
import struct
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

from osintgpt.vector_store.records import StoredChunk


SCHEMA = '''
CREATE TABLE IF NOT EXISTS jobs (
    ref TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    total INTEGER NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'embedding',
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS completed_chunks (
    ref TEXT NOT NULL REFERENCES jobs(ref) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    document_ref TEXT NOT NULL,
    chunk TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (ref, sequence)
);
'''


def fingerprint(source_hash, mapping, embedder, chunks, texts, document_refs,
                *, recipe=None, destination=None):
    '''Only reuse staged vectors for the same input and embedding recipe.

    This guards pending jobs; invalidation of already published indexes is a
    separate concern. No credentials or endpoint URLs are written to disk.
    '''
    digest = hashlib.sha256()

    def include(value):
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)

    if recipe is not None:
        include({
            'version': 2, 'source_hash': source_hash,
            'recipe': recipe, 'destination': destination
        })
        for chunk, text in zip(chunks, texts):
            include([asdict(chunk), text])
        return digest.hexdigest()

    include({
        'version': 1,
        'source_hash': source_hash,
        'mapping': mapping.to_dict(),
        'implementation': f'{type(embedder).__module__}.{type(embedder).__qualname__}',
        'provider': getattr(embedder, 'provider', ''),
        'model': embedder.model,
        'task_style': getattr(embedder, 'task_style', ''),
        'endpoint': str(getattr(getattr(embedder, 'client', None), 'base_url', '')),
        'purpose': 'document'
    })
    for chunk, text, document_ref in zip(chunks, texts, document_refs):
        legacy_chunk = asdict(chunk)
        legacy_chunk.pop('document_ref', None)
        include([legacy_chunk, text, document_ref])
    return digest.hexdigest()


class IndexJournal:
    '''One committed transaction per successful embedding response.

    An OS lock prevents writers in another app session or CLI process from
    racing this journal and the final publication. The lock is released by
    the OS on process exit, including a crash. Its file must not be unlinked:
    replacing an inode while another process holds it permits two owners.
    '''

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self._lock = self.path.with_suffix('.lock').open('a+b')
        try:
            self._acquire()
        except BaseException:
            self._lock.close()
            raise
        try:
            self.connection = sqlite3.connect(str(self.path))
            self.connection.execute('PRAGMA foreign_keys = ON')
            self.connection.execute('PRAGMA synchronous = FULL')
            self.connection.execute('PRAGMA auto_vacuum = INCREMENTAL')
            self.connection.executescript(SCHEMA)
            if 'details' not in {row[1] for row in self.connection.execute('PRAGMA table_info(jobs)')}:
                self.connection.execute("ALTER TABLE jobs ADD COLUMN details TEXT NOT NULL DEFAULT '{}'")
                self.connection.commit()
        except BaseException:
            self.close()
            raise

    def _acquire(self):
        try:
            if os.name == 'nt':
                import msvcrt
                if self._lock.seek(0, os.SEEK_END) == 0:
                    self._lock.write(b'\0')
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('This project is already being indexed.') from error

    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
        finally:
            # Closing releases both Windows byte locks and POSIX flock locks.
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def pending_refs(self):
        return {row[0] for row in self.connection.execute('SELECT ref FROM jobs')}

    def prepare(self, ref: str, signature: str, total: int, reset=False,
                compatible_signatures=(), details=None) -> int:
        '''Return the durable prefix length, or start a new compatible job.'''
        with self.connection:
            known = self.connection.execute(
                'SELECT fingerprint, total, completed, details FROM jobs WHERE ref = ?',
                (ref,)
            ).fetchone()
            if (known is not None and not reset and known[1] == total
                    and known[0] in (signature, *compatible_signatures)):
                if details is not None:
                    merged = {**json.loads(known[3]), **details}
                    self.connection.execute('UPDATE jobs SET details = ? WHERE ref = ?', (json.dumps(merged), ref))
                if known[0] != signature:
                    self.connection.execute(
                        'UPDATE jobs SET fingerprint = ? WHERE ref = ?', (signature, ref)
                    )
                return known[2]
            self.connection.execute('DELETE FROM jobs WHERE ref = ?', (ref,))
            self.connection.execute(
                'INSERT INTO jobs (ref, fingerprint, total, status, details) VALUES (?, ?, ?, ?, ?)',
                (ref, signature, total, 'embedding' if total else 'ready', json.dumps(details or {}))
            )
        return 0

    def save_batch(
        self, ref: str, start: int, chunks: Sequence[StoredChunk],
        document_refs: Sequence[str], vectors: Sequence[Sequence[float]], *, details=None
    ) -> None:
        if not vectors or not (len(chunks) == len(vectors) == len(document_refs)):
            raise ValueError('A checkpoint needs one vector and record ref per chunk.')
        dimensions = len(vectors[0])
        if not dimensions or any(
            len(vector) != dimensions or not all(math.isfinite(v) for v in vector)
            for vector in vectors
        ):
            raise ValueError('Embedding vectors must be finite with matching dimensions.')
        rows = []
        for offset, (chunk, document_ref, vector) in enumerate(
            zip(chunks, document_refs, vectors)
        ):
            if chunk.ref != ref or chunk.sequence != start + offset:
                raise ValueError('Checkpoint chunks must be in source sequence.')
            rows.append((
                ref, chunk.sequence, document_ref,
                json.dumps(asdict(chunk), ensure_ascii=False),
                struct.pack(f'<{dimensions}d', *vector)
            ))
        with self.connection:
            total, completed = self.connection.execute(
                'SELECT total, completed FROM jobs WHERE ref = ?', (ref,)
            ).fetchone()
            end = start + len(rows)
            if start != completed or end > total:
                raise ValueError('Checkpoint does not continue the saved prefix.')
            previous = self.connection.execute(
                'SELECT length(vector) FROM completed_chunks WHERE ref = ? LIMIT 1',
                (ref,)
            ).fetchone()
            if previous and previous[0] != dimensions * 8:
                raise ValueError('Embedding dimensions changed between batches.')
            self.connection.executemany(
                'INSERT INTO completed_chunks VALUES (?, ?, ?, ?, ?)', rows
            )
            self.connection.execute(
                'UPDATE jobs SET completed = ?, status = ? WHERE ref = ?',
                (end, 'ready' if end == total else 'embedding', ref)
            )
            if details is not None:
                self.connection.execute('UPDATE jobs SET details = ? WHERE ref = ?', (json.dumps(details), ref))

    def vectors_for(self, ref: str) -> List[List[float]]:
        total, completed = self.connection.execute(
            'SELECT total, completed FROM jobs WHERE ref = ?', (ref,)
        ).fetchone()
        if completed != total:
            raise ValueError('An incomplete file cannot be published.')
        vectors = []
        for sequence, blob in self.connection.execute(
            'SELECT sequence, vector FROM completed_chunks WHERE ref = ? ORDER BY sequence',
            (ref,)
        ):
            if sequence != len(vectors) or not blob or len(blob) % 8:
                raise ValueError('Invalid saved embedding checkpoint.')
            vectors.append(list(struct.unpack(f'<{len(blob) // 8}d', blob)))
        if len(vectors) != total:
            raise ValueError('Missing saved embedding checkpoints.')
        return vectors

    def forget(self, ref: str):
        with self.connection:
            self.connection.execute('DELETE FROM jobs WHERE ref = ?', (ref,))
        self.connection.execute('PRAGMA incremental_vacuum')
