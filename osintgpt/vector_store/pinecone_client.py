# -*- coding: utf-8 -*-

# ===============================================================
# osintgpt
#
#
# File: qdrant.py
# Description: Qdrant API. This file contains the Qdrant class
#   method for managing the Qdrant API connection.
# ===============================================================

# import modules <Pinecone>
import pinecone

# type hints
from typing import List, Optional

# Pinecone class
class Pinecone(object):
    '''
    Pinecone class

    This class provides methods for managing connections to a Pinecone server,
    allowing users to store, retrieve, and manipulate high-dimensional vector
    embeddings and associated documents within a Pinecone index.
    '''
    # constructor
    def __init__(self, **kwargs):
        '''
        Constructor

        args:
            **kwargs: keyword arguments for Pinecone
                api_key: API key
                environment: environment > find next to API key in console
        '''
        pinecone.init(**kwargs)
        self.pinecone = pinecone
    
    # get client
    def get_client(self):
        '''
        Get client

        returns:
            client: pinecone client
        '''
        # return client
        return self.pinecone
    
    # create index
    def create_index(self, index_name: str, dimension: int, metric: str = 'cosine'):
        '''
        Create index

        args:
            index_name: index name
            dimension: dimension
            metric: metric
        '''
        # create index
        self.pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric
        )
    
    # describe index stats
    def describe_index_stats(self, index_name: str):
        '''
        Describe index stats

        args:
            index_name: index name
        
        returns:
            index_stats: index stats
        '''
        # describe index stats
        index = self.pinecone.Index(index_name=index_name)
        return index.describe_index_stats()
    
    # build ids list
    def _build_ids_list(self, vectors: List):
        '''
        Build ids list

        args:
            vectors: vectors
        
        returns:
            ids_list: ids list
        '''
        # build ids list
        ids_list = []
        for i in range(len(vectors)):
            ids_list.append(i)
        
        return ids_list
    
    # add vectors
    def add_vectors(self, index_name: str, vectors: List, vector_name: str = 'main'):
        '''
        Add vectors

        args:
            index_name: index name
            vectors: vectors
            vector_name: vector name <namespace>: default = 'main'
        '''
        # add vectors
        index = self.pinecone.Index(index_name=index_name)
        index.upsert(
            vectors=zip(self._build_ids_list(vectors), vectors),
            namespace=vector_name
        )

