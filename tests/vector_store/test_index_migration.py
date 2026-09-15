'''Existing store upgrades and failed model replacement safeguards.'''

from unittest.mock import Mock

import pytest

from osintgpt.config import Settings
from osintgpt.vector_store import QdrantVectorStore, SQLiteVectorStore, StoredChunk
from osintgpt.vector_store.pgvector_store import PgVectorStore, _to_chunk


def test_sqlite_additive_migration_keeps_existing_vectors(tmp_path):
    path = tmp_path / 'old.sqlite'
    original = StoredChunk(ref='old.csv', sequence=0, text='evidence', embedding_model='old-model')
    with SQLiteVectorStore(path) as store:
        store.upsert('old.csv', [original], [[1.0, 0.0]])
        before = store.connection.execute('SELECT vector FROM chunks').fetchone()[0]
        # Reproduce the old schema, where record identity had no column.
        store.connection.execute('ALTER TABLE chunks DROP COLUMN document_ref')
        store.connection.commit()
    with SQLiteVectorStore(path) as upgraded:
        assert upgraded.chunks_for('old.csv') == [original]
        assert upgraded.search([1.0, 0.0], 'old-model')[0].chunk == original
        assert upgraded.connection.execute('SELECT vector FROM chunks').fetchone()[0] == before


def test_qdrant_new_dimension_is_refused_before_old_points_are_deleted():
    from qdrant_client import QdrantClient
    client = QdrantClient(':memory:')
    store = QdrantVectorStore(Settings(), client=client)
    original = StoredChunk(ref='source', sequence=0, text='old', embedding_model='old-model')
    try:
        store.upsert('source', [original], [[1.0, 0.0]])
        with pytest.raises(ValueError, match='different vector size'):
            store.upsert('source', [original], [[1.0, 0.0, 0.0]])
        assert store.chunks_for('source') == [original]
        assert store.search([1.0, 0.0], 'old-model')[0].chunk == original
    finally:
        client.close()


def test_postgres_failed_replacement_rolls_back_before_returning_to_indexer():
    store = PgVectorStore.__new__(PgVectorStore)
    store.connection = Mock()
    store._upsert = Mock(side_effect=ValueError('different vector size'))
    with pytest.raises(ValueError):
        store.upsert('source', [], [])
    store.connection.rollback.assert_called_once()
    store.connection.commit.assert_not_called()


def test_postgres_record_adapter_reads_new_identity_column():
    result = _to_chunk(('file.csv', 0, 'text', '', '', '', {}, 'model', 'file.csv#row'))
    assert result.document_ref == 'file.csv#row' and result.embedding_model == 'model'
