# -*- coding: utf-8 -*-

# =============================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: openai_embeddings.py
# Description: GPT API. This file contains the OpenAIEmbeddingGenerator class
#   method for managing the GPT API connection.
# =============================================================================

# import modules
import os
import openai
import tiktoken
import pandas as pd

# import submodules
from ast import literal_eval
from dotenv import load_dotenv

# type hints
from typing import List, Optional

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# OpenAIEmbeddingGenerator class
class OpenAIEmbeddingGenerator(object):
    '''
    OpenAIEmbeddingGenerator class

    This class contains the methods for managing the GPT API connection, including
    embeddings and vector stores.

    Example:
        .. code-block:: python

            from osintgpt import OpenAIEmbeddingGenerator
        
    '''
    # constructor
    def __init__(self, env_file_path: str):
        '''
        Constructor
        '''
        # load environment variables
        load_dotenv(dotenv_path=env_file_path)

        # set environment variables
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        if not self.OPENAI_API_KEY:
            raise MissingEnvironmentVariableError('OPENAI_API_KEY is not set')

        self.OPENAI_GPT_MODEL = os.getenv('OPENAI_GPT_MODEL', '')
        if not self.OPENAI_GPT_MODEL:
            raise MissingEnvironmentVariableError('OPENAI_GPT_MODEL is not set')

    # get openai api key
    def get_openai_api_key(self):
        '''
        Get OpenAI API key

        Returns:
            OPENAI_API_KEY (str): OpenAI API key
        '''
        return self.OPENAI_API_KEY

    # get openai gpt model
    def get_openai_gpt_model(self):
        '''
        Get OpenAI GPT model

        Returns:
            OPENAI_GPT_MODEL (str): OpenAI GPT model
        '''
        return self.OPENAI_GPT_MODEL
    
    # process text data
    def load_text(self, data: List[str]):
        '''
        Load text.
        It loads text data to be processed.

        Args:
            data (list): List of strings
        '''
        if isinstance(data, list):
            self.data = data
        else:
            raise TypeError('Data must be a list')
    
    # count tokens < GPT model >
    def count_tokens(self):
        '''
        Count tokens

        It counts the number of tokens in the data.

        Returns:
            num_tokens (int): Number of tokens
        '''
        # get model
        model = self.get_openai_gpt_model()
        encoding = tiktoken.encoding_for_model(model)

        # count tokens
        self.num_tokens = 0
        for d in self.data:
            tokens = encoding.encode(d)
            self.num_tokens += len(tokens)

        return self.num_tokens
    
    # calculate estimated cost
    def calculate_embeddings_estimated_cost(self):
        '''
        It calculates the estimated cost of the data based on the number of tokens.
        Costs are based on the OpenAI text-embedding-ada-002 pricing model.

        As of April 2023, the cost is 0.0004 per 1000 tokens.

        Returns:
            estimated_cost (float): Estimated cost
        '''
        return ((self.count_tokens() / 1000) * 0.0004)

    # calculate embeddings
    def calculate_embeddings(self):
        '''
        Calculate embeddings

        This method calculates embeddings using Openai's text-embedding-ada-002
        model.

        Returns:
            embeddings (list): Embeddings
        '''
        EMBEDDING_MODEL = 'text-embedding-ada-002'
        BATCH_SIZE = 1000
        
        # batch data
        documents = [
            self.data[i:i + BATCH_SIZE] for i in range(
                0, len(self.data), BATCH_SIZE
            )
        ]

        # set openai api key
        openai.api_key = self.get_openai_api_key()

        # calculate embeddings
        embeddings = []
        for docs in documents:
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=docs)

            # double check embeddings are in same order as input
            for i, be in enumerate(response['data']):
                assert i == be['index']
            
            # batch embeddings
            batch_embeddings = [e['embedding'] for e in response['data']]
            embeddings.extend(batch_embeddings)
        
        return embeddings

    # property for embeddings
    @property
    def embeddings(self):
        '''
        Get embeddings

        This property calculates the embeddings if they have not been calculated

        Returns:
            embeddings (list): Embeddings
        '''
        if not hasattr(self, '_embeddings'):
            self._embeddings = self.calculate_embeddings()
        
        return self._embeddings
    
    # generate embedding
    def generate_embedding(self, text: str):
        '''
        Generate an embedding for a given text using the text-embedding-ada-002
        model.

        Args:
            text (str): Text to generate embedding for
        
        Returns:
            embedding (list): Embedding
        '''
        EMBEDDING_MODEL = 'text-embedding-ada-002'
        openai.api_key = self.get_openai_api_key()
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=[text])
        embedding = response['data'][0]['embedding']

        return embedding

    # load embeddings
    def load_embeddings_from_csv(self, embeddings_path: str,
        columns: List, **kwargs):
        '''
        Load embeddings from disk

        This method loads the embeddings from disk.

        Args:
            embeddings_path (str): Path to embeddings csv file
            columns (list): List of columns specifying the embeddings
            **kwargs: Keyword arguments for pd.read_csv
        
        Returns:
            embeddings (list): Embeddings
        '''
        data = pd.read_csv(embeddings_path, **kwargs)
        for col in columns:
            data[col] = data[col].apply(literal_eval)
        
        return data
