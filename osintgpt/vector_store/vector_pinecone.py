# -*- coding: utf-8 -*-

# =================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: pinecone_client.py
# Description: Pinecone API. This file contains the Pinecone class
#   method for managing the Pinecone API connection.
# =================================================================

# import modules <Pinecone>
import os
import pinecone

# type hints
from typing import List, Optional

# import submodules
from dotenv import load_dotenv

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# import base class
from .base import BaseVectorEngine

# Pinecone class
class Pinecone(BaseVectorEngine):
    '''
    Pinecone class

    This class provides methods for managing connections to a Pinecone server,
    allowing users to store, retrieve, and manipulate high-dimensional vector
    embeddings and associated documents within a Pinecone index.
    '''
    # constructor
    def __init__(self, env_file_path: str):
        '''
        Constructor
        '''
        # load environment variables
        load_dotenv(dotenv_path=env_file_path)

        # set environment variables
        self.api_key = os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise MissingEnvironmentVariableError('PINECONE_API_KEY')
        
        self.environment = os.getenv('PINECONE_ENVIRONMENT')
        if not self.environment:
            raise MissingEnvironmentVariableError('PINECONE_ENVIRONMENT')

        pinecone.init(api_key=self.api_key, environment=self.environment)
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
