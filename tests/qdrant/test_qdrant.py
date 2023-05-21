# -*- coding: utf-8 -*-

# ===============================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_qdrant.py
# Description: Testing Qdrant methods
# ===============================================================

# import modules
import pytest

# import submodules
from unittest.mock import Mock

# import Qdrant
from qdrant_client.http import models as rest
from osintgpt.vector_store.qdrant import Qdrant


def test_get_collections(mocker):
    # Mock qdrant_client.QdrantClient
    mock_qdrant_client = mocker.patch(
        'qdrant_client.QdrantClient',
        autospec=True
    )

    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda x: 'dummy_value')

    # Create Qdrant instance with a mocked QdrantClient
    qdrant = Qdrant(env_file_path='/path/to/your/env/file')

    # The mocked method
    mock_qdrant_client.return_value.get_collections.return_value = [
        'test_collection1',
        'test_collection2'
    ]

    collections = qdrant.get_collections()

    # Verify get_collections method was called on qdrant client
    mock_qdrant_client.return_value.get_collections.assert_called_once()

    # Assert that the collections returned by the method match the expected value
    assert collections == ['test_collection1', 'test_collection2']

def test_create_collection(mocker):
    # Mock qdrant_client.QdrantClient
    mock_qdrant_client = mocker.patch(
        'qdrant_client.QdrantClient',
        autospec=True
    )

    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda x: 'dummy_value')

    # Create Qdrant instance with a mocked QdrantClient
    qdrant = Qdrant(env_file_path='/path/to/your/env/file')

    # Define the arguments for create_collection
    collection_name = 'test_collection'
    vector_size = 128
    vector_name = 'main'

    # Call create_collection
    qdrant.create_collection(collection_name, vector_size, vector_name)

    # Check that recreate_collection was called with the correct arguments
    mock_qdrant_client.return_value.recreate_collection.assert_called_with(
        collection_name=collection_name,
        vectors_config={
            vector_name: rest.VectorParams(
                distance=rest.Distance.COSINE,
                size=vector_size
            )
        }
    )

# test add vectors
def test_add_vectors(mocker):
    # Mock qdrant_client.QdrantClient
    mock_qdrant_client = mocker.patch(
        'qdrant_client.QdrantClient',
        autospec=True
    )

    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda x: 'dummy_value')

    # Create Qdrant instance with a mocked QdrantClien
    
    
    
    
    qdrant = Qdrant(env_file_path='/path/to/your/env/file')

    # The mocked method
    mock_qdrant_client.return_value.upsert.return_value = True

    vectors = [[0.1, 0.2], [0.3, 0.4]]
    payload = [{'id': 1}, {'id': 2}]

    qdrant.add_vectors('test_collection', vectors, 'test_vector', payload)

    # Test if upsert method was called with the correct parameters
    mock_qdrant_client.return_value.upsert.assert_called_with(
        collection_name='test_collection',
        points=[
            {'id': 0, 'vector': {'test_vector': [0.1, 0.2]}, 'payload': {'id': 1}},
            {'id': 1, 'vector': {'test_vector': [0.3, 0.4]}, 'payload': {'id': 2}}
        ]
    )

# test update vector collection
def test_update_vector_collection(mocker):
    # Mock qdrant_client.QdrantClient
    mock_qdrant_client = mocker.patch(
        'qdrant_client.QdrantClient',
        autospec=True
    )

    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda x: 'dummy_value')

    # Create Qdrant instance with a mocked QdrantClient
    qdrant = Qdrant(env_file_path='/path/to/your/env/file')

    # The mocked method
    mock_qdrant_client.return_value.upsert.return_value = True

    # Mock count_vectors method to return count=2
    mock_qdrant_client.return_value.count.return_value = mocker.MagicMock(count=2)

    vectors = [[0.5, 0.6], [0.7, 0.8]]
    payload = [{'id': 3}, {'id': 4}]

    qdrant.update_vector_collection(
        'test_collection',
        vectors,
        'test_vector',
        payload
    )

    # Test if upsert method was called with the correct parameters
    mock_qdrant_client.return_value.upsert.assert_called_with(
        collection_name='test_collection',
        points=[
            {'id': 2, 'vector': {'test_vector': [0.5, 0.6]}, 'payload': {'id': 3}},
            {'id': 3, 'vector': {'test_vector': [0.7, 0.8]}, 'payload': {'id': 4}}
        ]
    )


def test_search_query(mocker):
    # Mock qdrant_client.QdrantClient
    mock_qdrant_client = mocker.patch(
        'qdrant_client.QdrantClient',
        autospec=True
    )

    # Mock environment variables
    mocker.patch('os.getenv', side_effect=lambda x: 'dummy_value')

    # Create Qdrant instance with a mocked QdrantClient
    qdrant = Qdrant(env_file_path='/path/to/your/env/file')

    # The mocked method
    query_result = ['dummy_result']
    mock_qdrant_client.return_value.search.return_value = query_result

    # Define the arguments for search_query
    embedded_query = [0.1, 0.2, 0.3]
    top_k = 5
    collection_name = 'test_collection'
    vector_name = 'main'

    # Call search_query
    result = qdrant.search_query(
        embedded_query,
        top_k,
        collection_name=collection_name,
        vector_name=vector_name
    )

    # Check that search was called with the correct arguments
    mock_qdrant_client.return_value.search.assert_called_with(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k
    )

    # Assert the search_query result
    assert result == query_result, 'The result does not match the expected result'
