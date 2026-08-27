# -*- coding: utf-8 -*-

# ===============================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_qdrant.py
# Description: Testing Qdrant methods against a mocked client.
# ===============================================================

# import modules
import pytest

# import submodules
from qdrant_client.http import models as rest

# import osintgpt config
from osintgpt.config import Settings

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# import Qdrant
from osintgpt.vector_store.connection import TIMEOUT_SECONDS
from osintgpt.vector_store.qdrant import Qdrant

LOCAL = Settings(qdrant_host='localhost', qdrant_port=6333)
REMOTE = Settings(qdrant_api_key='qdrant-key', qdrant_url='https://example.invalid')


@pytest.fixture
def client(mocker):
    '''The mocked QdrantClient class; `.return_value` is the instance.'''
    return mocker.patch('qdrant_client.QdrantClient', autospec=True)


@pytest.fixture
def qdrant(client):
    return Qdrant(LOCAL)


class TestConnection:
    def test_local_settings_connect_by_host_and_port(self, client):
        Qdrant(LOCAL)

        client.assert_called_once_with(
            host='localhost', port=6333, timeout=TIMEOUT_SECONDS
        )

    def test_remote_settings_connect_by_url(self, client):
        Qdrant(REMOTE)

        client.assert_called_once_with(
            url='https://example.invalid', api_key='qdrant-key', https=True,
            timeout=TIMEOUT_SECONDS
        )

    def test_the_timeout_outlasts_creating_a_collection(self, client):
        '''
        The client defaults to five seconds, tuned for search. Creating a
        collection takes longer than that on a real server, so the first write
        of a project would time out — measured against one, not guessed.
        '''
        assert TIMEOUT_SECONDS >= 30

    def test_remote_wins_when_both_pairs_are_present(self, client):
        Qdrant(Settings(
            qdrant_host='localhost', qdrant_port=6333,
            qdrant_api_key='qdrant-key', qdrant_url='https://example.invalid'
        ))

        assert 'url' in client.call_args.kwargs

    @pytest.mark.parametrize('settings', [
        Settings(),
        Settings(qdrant_host='localhost'),
        Settings(qdrant_api_key='qdrant-key')
    ])
    def test_incomplete_settings_are_rejected(self, client, settings):
        with pytest.raises(MissingEnvironmentVariableError):
            Qdrant(settings)

    def test_accepts_a_path_with_a_deprecation_warning(self, client, tmp_path):
        path = tmp_path / '.env'
        path.write_text('QDRANT_HOST=localhost\nQDRANT_PORT=6333\n', encoding='utf-8')

        with pytest.warns(DeprecationWarning):
            Qdrant(str(path))

    def test_an_unreachable_server_raises_connection_error(self, client):
        client.return_value.get_collections.side_effect = OSError('refused')

        with pytest.raises(ConnectionError):
            Qdrant(LOCAL)


class TestCollections:
    def test_get_collections(self, qdrant, client):
        expected = ['test_collection1', 'test_collection2']
        client.return_value.get_collections.return_value = expected

        assert qdrant.get_collections() == expected

    def test_create_collection(self, qdrant, client):
        qdrant.create_collection('test_collection', 128, 'main')

        client.return_value.recreate_collection.assert_called_with(
            collection_name='test_collection',
            vectors_config={
                'main': rest.VectorParams(
                    distance=rest.Distance.COSINE, size=128
                )
            }
        )

    def test_delete_collection(self, qdrant, client):
        qdrant.delete_collection('test_collection')

        client.return_value.delete_collection.assert_called_with(
            collection_name='test_collection'
        )


class TestVectors:
    def test_add_vectors(self, qdrant, client):
        qdrant.add_vectors(
            'test_collection', [[0.1, 0.2], [0.3, 0.4]], 'test_vector',
            [{'id': 1}, {'id': 2}]
        )

        client.return_value.upsert.assert_called_with(
            collection_name='test_collection',
            points=[
                rest.PointStruct(
                    id=0, vector={'test_vector': [0.1, 0.2]}, payload={'id': 1}
                ),
                rest.PointStruct(
                    id=1, vector={'test_vector': [0.3, 0.4]}, payload={'id': 2}
                )
            ]
        )

    def test_add_vectors_rejects_a_mismatched_payload(self, qdrant):
        with pytest.raises(ValueError):
            qdrant.add_vectors(
                'test_collection', [[0.1], [0.2]], 'test_vector', [{'id': 1}]
            )

    def test_update_vector_collection_continues_the_id_sequence(
        self, qdrant, client, mocker
    ):
        client.return_value.count.return_value = mocker.MagicMock(count=2)

        qdrant.update_vector_collection(
            'test_collection', [[0.5, 0.6], [0.7, 0.8]], 'test_vector',
            [{'id': 3}, {'id': 4}]
        )

        client.return_value.upsert.assert_called_with(
            collection_name='test_collection',
            points=[
                rest.PointStruct(
                    id=2, vector={'test_vector': [0.5, 0.6]}, payload={'id': 3}
                ),
                rest.PointStruct(
                    id=3, vector={'test_vector': [0.7, 0.8]}, payload={'id': 4}
                )
            ]
        )


class TestSearchQuery:
    def test_searches_the_named_collection(self, qdrant, client, mocker):
        expected = ['dummy_result']
        client.return_value.query_points.return_value = mocker.MagicMock(
            points=expected
        )

        result = qdrant.search_query(
            [0.1, 0.2, 0.3], 5,
            collection_name='test_collection', vector_name='main'
        )

        client.return_value.query_points.assert_called_with(
            collection_name='test_collection',
            query=[0.1, 0.2, 0.3],
            using='main',
            limit=5
        )
        assert result == expected

    def test_uses_the_current_client_api(self, qdrant, client):
        '''autospec fails here rather than passing against a method the
        installed client does not have.'''
        assert not hasattr(client.return_value, 'search')

    def test_requires_a_collection_name(self, qdrant):
        with pytest.raises(ValueError, match='collection_name'):
            qdrant.search_query([0.1, 0.2], 5)
