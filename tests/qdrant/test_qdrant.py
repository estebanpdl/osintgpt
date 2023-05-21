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
from osintgpt.vector_store.qdrant import Qdrant

# test add vectors
def test_add_vectors(mocker):
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
