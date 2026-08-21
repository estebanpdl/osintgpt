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
import sys
import qdrant_client

# import submodules <Qdrant>
from qdrant_client.http import models as rest

# type hints
from typing import List, Optional, Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

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
    def __init__(self, config: Union[Settings, str]):
        '''
        Constructor

        args:
            config (Union[Settings, str]): Settings, or a path to a .env file \
                (deprecated).
        '''
        # settings
        self.settings = resolve_settings(config)

        # connect
        self.set_required_variables()

    # set required settings
    def set_required_variables(self):
        '''
        set required settings

        This method reads the settings required to connect to a Qdrant server
        and opens the connection. A remote pair (api key + url) wins over a
        local pair (host + port) when both are present.

        returns:
            use_remote: use remote
            use_local: use local
        '''
        # set required settings
        settings = self.settings
        use_remote = settings.qdrant_api_key and settings.qdrant_url
        use_local = settings.qdrant_port and settings.qdrant_host

        if not (use_remote or use_local):
            raise MissingEnvironmentVariableError(
                'QDRANT_API_KEY or QDRANT_URL or QDRANT_HOST or QDRANT_PORT',
                hint='a remote Qdrant needs an api key and a url; a local one '
                     'needs a host and a port'
            )

        # set connection settings
        if use_remote:
            self.api_key = settings.qdrant_api_key
            self.url = settings.qdrant_url

            # connect
            self.qdrant = qdrant_client.QdrantClient(
                url=self.url,
                api_key=self.api_key,
                https=True
            )
        else:
            self.host = settings.qdrant_host
            self.port = settings.qdrant_port

            # connect
            self.qdrant = qdrant_client.QdrantClient(
                host=self.host,
                port=self.port
            )

        '''

        Ensure if is indeed connected
        '''
        # Perform a simple operation to check connectivity
        try:
            collections = self.get_collections()
        except Exception as e:
            m = f'''
            Unable to establish a connection to the Qdrant server. Please ensure
            that the Qdrant server is up and running. If you're using this locally,
            make sure to start the Qdrant server before using this feature.
            '''
            raise ConnectionError(' '.join(m.split()).strip()) from None
    
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
        vector_name: str = 'main', payload: Optional[List[dict]] = None):
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
                type: List[dict]
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
        # qdrant-client removed `search` in 1.19; `query_points` replaces it and
        # wraps the hits, so unwrap to keep returning a plain list of points.
        response = self.qdrant.query_points(
            collection_name=collection_name,
            query=embedded_query,
            using=vector_name,
            limit=top_k
        )

        return response.points
