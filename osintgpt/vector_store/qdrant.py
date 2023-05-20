# -*- coding: utf-8 -*-

# ===============================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: qdrant.py
# Description: Qdrant API. This file contains the Qdrant class
#   method for managing the Qdrant API connection.
# ===============================================================

# import modules <Qdrant>
import os
import qdrant_client

# import submodules <Qdrant>
from qdrant_client.http import models as rest

# import submodules
from dotenv import load_dotenv

# type hints
from typing import List, Optional

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# import base class
from .base import BaseVectorEngine

# Qdrant class
class Qdrant(BaseVectorEngine):
    '''
    Qdrant class

    This class provides methods for managing connections to a Qdrant server,
    allowing users to store, retrieve, and manipulate high-dimensional vector
    embeddings and associated documents within a Qdrant collection. It offers
    functionality for creating, updating, and deleting collections, as well as
    adding and updating vector embeddings and their associated payloads.

    Main features:
        - Manage connections to a Qdrant server
        - Create, update, and delete Qdrant collections
        - Add and update vector embeddings and payloads
        - Retrieve collection information and count vectors
        - Efficiently store and search embeddings

    For more information about QdrantClient arguments, see:
    github.com/qdrant/qdrant-client/blob/master/qdrant_client/qdrant_client.py
    '''
    # constructor
    def __init__(self, env_file_path: str):
        '''
        Constructor

        args:
            **kwargs: keyword arguments for QdrantClient
        '''
        # load environment variables
        load_dotenv(dotenv_path=env_file_path)

        # set environment variables
        self.set_required_variables()

    # set required environment variables
    def set_required_variables(self):
        '''
        set required environment variables

        This method sets for the required environment variables for connecting
        to a Qdrant server.

        returns:
            use_remote: use remote
            use_local: use local
        '''
        # set required environment variables
        use_remote = os.getenv('QDRANT_API_KEY') and os.getenv('QDRANT_URL')
        use_local = os.getenv('QDRANT_PORT') and os.getenv('QDRANT_HOST')

        if not (use_remote or use_local):
            raise MissingEnvironmentVariableError(
                'QDRANT_API_KEY or QDRANT_URL or QDRANT_HOST or QDRANT_PORT'
            )

        # set environment variables
        if use_remote:
            self.api_key = os.getenv('QDRANT_API_KEY')
            self.url = os.getenv('QDRANT_URL')

            # connect
            self.qdrant = qdrant_client.QdrantClient(
                url=self.url,
                api_key=self.api_key,
                https=True
            )
        else:
            self.host = os.getenv('QDRANT_HOST')
            self.port = int(os.getenv('QDRANT_PORT'))

            # connect
            self.qdrant = qdrant_client.QdrantClient(
                host=self.host,
                port=self.port
            )
    
    # get client
    def get_client(self):
        '''
        Get client

        returns:
            client: qdrant client
        '''
        # return client
        return self.qdrant
    
    # get collections
    def get_collections(self):
        '''
        Get collections

        returns:
            collections: collections
        '''
        # get collections
        collections = self.qdrant.get_collections()
        
        # return collections
        return collections
    
    # get collection info
    def get_collection_info(self, collection_name: str):
        '''
        Get collection info

        args:
            collection_name: collection name
                type: str
        
        returns:
            info: collection info
        '''
        # get collection
        info = self.qdrant.get_collection(collection_name=collection_name)
        
        # return collection info
        return info
    
    # count vectors
    def count_vectors(self, collection_name: str):
        '''
        Count vectors

        args:
            collection_name: collection name
                type: str
        
        returns:
            count: count
        '''
        # return count
        return self.qdrant.count(collection_name=collection_name)
    
    def _validate_payload_length(self, payload: Optional[List], vectors: List):
        '''
        Validate payload length

        args:
            payload: payload
                type: list
            vectors: vectors
                type: list
        '''
        # validate payload length
        if payload and len(payload) != len(vectors):
            raise ValueError('Payload length must be the same as vectors length')
    
    # get vector
    def get_vectors(self, collection_name: str):
        '''
        Get vector

        args:
            collection_name: collection name
                type: str
        
        returns:
            vector: vector
        '''
        # get collection
        collection = self.qdrant.get_collection(collection_name=collection_name)
        
        # get vector
        vector = collection.config.params
        return vector
    
    # create collection
    def create_collection(self, collection_name: str, vector_size: int,
        vector_name: str = 'main'):
        '''
        Create collection

        args:
            collection_name: collection name
                type: str
            vector_size: vector size
                type: int
            vector_name: name
                type: str
        '''
        # create collection
        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: rest.VectorParams(
                    distance=rest.Distance.COSINE,
                    size=vector_size
                )
            }
        )
    
    # add vectors
    def add_vectors(self, collection_name: str, vectors: List,
        vector_name: str = 'main', payload: Optional[List[dict]] = None):
        '''
        Add vectors

        args:
            collection_name: collection name
                type: str
            vectors: vectors
                type: list
            vector_name: name
                type: str
            payload: payload. Should be the same length as vectors and same order.
                type: list
        '''
        # validate payload length
        self._validate_payload_length(payload, vectors)
        
        # add vectors
        self.qdrant.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=k,
                    vector={
                        vector_name: v
                    },
                    payload=payload[k] if payload else None
                )
                for k, v in enumerate(vectors)
            ]
        )
    
    # update vector collection
    def update_vector_collection(self, collection_name: str, vectors: List,
        vector_name: str = 'main', payload: Optional[List] = None):
        '''
        Update vector collection

        args:
            collection_name: collection name
                type: str
            vectors: vectors
                type: list
            vector_name: name
                type: str
            payload: payload. Should be the same length as vectors and same order.
                type: list
        '''
        # validate payload length
        self._validate_payload_length(payload, vectors)

        # count vectors
        count = self.count_vectors(collection_name=collection_name)
        n = count.count

        # add vectors
        self.qdrant.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=k + n,
                    vector={
                        vector_name: v
                    },
                    payload=payload[k] if payload else None
                )
                for k, v in enumerate(vectors)
            ]
        )
    
    # delete collection
    def delete_collection(self, collection_name: str):
        '''
        Delete collection

        args:
            collection_name: collection name
                type: str
        '''
        # delete collection
        self.qdrant.delete_collection(collection_name=collection_name)
    
    # search query
    def search_query(self, embedded_query: List[float], top_k: int = 10, **kwargs):
        '''
        Search query in collection

        args:
            embedded_query: embedded query
                type: list
            top_k: top k
                type: int

            kwargs:
                collection_name: collection name
                    type: str
                vector_name: name
                    type: str
        
        returns:
            result: result
        '''
        # collection name
        collection_name = kwargs.get('collection_name', None)
        if collection_name is None:
            raise ValueError('collection_name must be specified')
        
        # vector name
        vector_name = kwargs.get('vector_name', 'main')

        # query results
        query_results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=(
                vector_name, embedded_query
            ),
            limit=top_k
        )

        return query_results
