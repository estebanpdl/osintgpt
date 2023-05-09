# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: openai_gpt.py
# Description: The openai_gpt.py file contains a class that provides an interface 
#   to interact with OpenAI's GPT models, allowing users to read embeddings, and
#   ask questions about the data.
# =================================================================================

# import modules
import os
import openai
import tiktoken
import pandas as pd

# import submodules
from scipy import spatial
from ast import literal_eval
from dotenv import load_dotenv

# type hints
from typing import List, Optional

# import osintgpt vector stores
from osintgpt.vector_store import BaseVectorEngine, Pinecone, Qdrant

# import osintgpt openai embeddings
from osintgpt.embedding import OpenAIEmbeddingGenerator

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# OpenAIGPT class
class OpenAIGPT(object):
    '''
    OpenAIGPT class
    '''
    # constructor
    def __init__(self, env_file_path: str):
        '''
        Constructor
        '''
        # load environment variables
        self.env_file_path = env_file_path
        load_dotenv(dotenv_path=env_file_path)

        # set environment variables
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
        if not self.OPENAI_API_KEY:
            raise MissingEnvironmentVariableError('OPENAI_API_KEY')
        
        self.OPENAI_GPT_MODEL = os.getenv('OPENAI_GPT_MODEL', '')
        if not self.OPENAI_GPT_MODEL:
            raise MissingEnvironmentVariableError('OPENAI_GPT_MODEL')
        
    # set openai api key
    def get_openai_api_key(self):
        '''
        Get OpenAI API key

        Returns:
            OPENAI_API_KEY (str): OpenAI API key
        '''
        return self.OPENAI_API_KEY
    
    # load embeddings
    def load_embeddings_from_csv(self, file_path: str,
        columns: List, **kwargs):
        '''
        Load embeddings from csv file

        Args:
            file_path (str): File path
            columns (list): List of columns specifying the embeddings
            **kwargs: Keyword arguments for pandas.read_csv
        '''
        data = pd.read_csv(file_path, **kwargs)
        for col in columns:
            data[col] = data[col].apply(literal_eval)
        
        self._embeddings = {
            col: data[col].tolist() for col in columns
        }

        return data

    # load embeddings from dataframe
    def load_embeddings_from_dataframe(self, dataframe: pd.DataFrame,
        columns: List):
        '''
        Load embeddings from dataframe

        Args:
            dataframe (pd.DataFrame): Pandas dataframe
            columns (list): List of columns specifying the embeddings
        '''
        for col in columns:
            dataframe[col] = dataframe[col].apply(literal_eval)
        
        self._embeddings = {
            col: data[col].tolist() for col in columns
        }

    # get embeddings
    def get_embeddings(self, column: str):
        '''
        Get embeddings

        Args:
            column (str): Column name

        Returns:
            embeddings (list): List of embeddings
        '''
        return self._embeddings[column]

    # get embeddings dataframe
    def get_embeddings_dataframe(self, columns: List):
        '''
        Get embeddings dataframe

        Args:
            columns (list): List of columns specifying the embeddings

        Returns:
            dataframe (pd.DataFrame): Pandas dataframe
        '''
        if not hasattr(self, '_embeddings'):
            raise AttributeError('No embeddings loaded. Please load embeddings.')
        
        return pd.DataFrame(self._embeddings, columns=columns)

    # load search top k results
    def _load_search_results(self, query: str, vector_engine: BaseVectorEngine,
        top_k: int = 10):
        '''
        '''
        if not isinstance(vector_engine, BaseVectorEngine):
            supported_vector_engines = [Pinecone, Qdrant]
            supported_vector_engine_names = ', '.join(
                [engine.__name__ for engine in supported_vector_engines]
            )

            # build message
            msg_a = 'Invalid vector engine provided'
            msg_b = 'Must be an instance of one of the following classes:'
            message = f'{msg_a}. {msg_b} {supported_vector_engine_names}.'
            raise ValueError(message)
        
        return True
    
    # search -> strings_ranked_by_relatedness
    def strings_ranked_by_relatedness(self, query: str, df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_k: int = 10, embeddings_target_column: str = 'embeddings',
        text_target_column: str = 'text'):
        '''
        Returns a list of strings and relatednesses, sorted from most related
        to least.
        '''
        embedding_generator = OpenAIEmbeddingGenerator(self.env_file_path)
        query_embedding = embedding_generator.generate_embedding(query)
        
        strings_and_relatednesses =[
            (
                row[text_target_column],
                relatedness_fn(query_embedding, row[embeddings_target_column])
            )
            for _, row in df.iterrows()
        ]

        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_k], relatednesses[:top_k]
    
    