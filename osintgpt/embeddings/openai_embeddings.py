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
import warnings
import pandas as pd

# import submodules
from ast import literal_eval

# type hints
from typing import List, Optional, Union

# import osintgpt config
from osintgpt.config import (
    DEFAULT_EMBEDDING_MODEL,
    Settings,
    resolve_settings
)

# import osintgpt llm
from osintgpt.llm import build_embedding_provider

# import osintgpt pricing
from osintgpt.pricing import estimate_cost

# import utils
from osintgpt.utils import encoding_for_model

# OpenAIEmbeddingGenerator class
class OpenAIEmbeddingGenerator(object):
    '''
    OpenAIEmbeddingGenerator class.

    This class contains the methods for managing the GPT API connection, including
    embeddings and vector stores.
    '''
    def __init__(self, config: Union[Settings, str]):
        '''
        Initializes the instance of the class.

        Args:
            config (Union[Settings, str]): Settings, or a path to a .env file \
                (deprecated).

        Raises:
            MissingEnvironmentVariableError: If either 'openai_api_key' or \
                'openai_gpt_model' has no value.
        '''
        # settings
        self.settings = resolve_settings(config).require(
            'openai_api_key', 'openai_gpt_model'
        )

        self.OPENAI_API_KEY = self.settings.openai_api_key
        self.OPENAI_GPT_MODEL = self.settings.openai_gpt_model
        self.OPENAI_EMBEDDING_MODEL = (
            self.settings.openai_embedding_model or DEFAULT_EMBEDDING_MODEL
        )

        warnings.warn(
            'OpenAIEmbeddingGenerator is deprecated and will be removed in '
            "1.0; use osintgpt.llm.build_embedding_provider('openai', "
            'settings) instead.',
            DeprecationWarning,
            stacklevel=2
        )

        # provider
        self.provider = build_embedding_provider(
            'openai', self.settings, model=self.OPENAI_EMBEDDING_MODEL
        )

    # the underlying client, which lives on the provider
    @property
    def client(self):
        return self.provider.client

    @client.setter
    def client(self, value):
        self.provider.client = value

    # get openai embedding model
    def get_openai_embedding_model(self):
        '''
        Get the embedding model these embeddings are generated with.

        Returns:
            str: OpenAI embedding model.
        '''
        return self.OPENAI_EMBEDDING_MODEL

    # get openai api key
    def get_openai_api_key(self):
        '''
        Get OpenAI API key.

        Returns:
            str: OpenAI API key.
        '''
        return self.OPENAI_API_KEY

    # get openai gpt model
    def get_openai_gpt_model(self):
        '''
        Get OpenAI GPT model.

        Returns:
            str: OpenAI GPT model.
        '''
        return self.OPENAI_GPT_MODEL
    
    # process text data
    def load_text(self, data: List[str]):
        '''
        Load text.
        It loads text data to be processed.

        Args:
            data (List): List of strings.
        
        Returns:
            None
        '''
        if isinstance(data, list):
            self.data = data
        else:
            raise TypeError('Data must be a list')
    
    # count tokens < embedding model >
    def count_tokens(self):
        '''
        Count tokens.
        It counts the number of tokens in the data, using the encoding of the
        embedding model the data will actually be sent to.

        Returns:
            int: Number of tokens.
        '''
        # get model
        encoding = encoding_for_model(self.get_openai_embedding_model())

        # count tokens
        self.num_tokens = 0
        for d in self.data:
            tokens = encoding.encode(d)
            self.num_tokens += len(tokens)

        return self.num_tokens

    # calculate estimated cost
    def calculate_embeddings_estimated_cost(self):
        '''
        It calculates the estimated cost of embedding the loaded data.

        Returns:
            Optional[float]: Estimated cost in USD, or None when the embedding \
                model carries no price in the table.
        '''
        return estimate_cost(
            self.get_openai_embedding_model(),
            self.count_tokens()
        )

    # calculate embeddings
    def calculate_embeddings(self):
        '''
        Calculate embeddings.
        This method calculates embeddings using the configured embedding model.

        Returns:
            list: Embeddings.
        '''
        return self.provider.embed(self.data)

    # property for embeddings
    @property
    def embeddings(self):
        '''
        Get embeddings.
        This property calculates the embeddings if they have not been calculated.

        Returns:
            list: Embeddings.
        '''
        if not hasattr(self, '_embeddings'):
            self._embeddings = self.calculate_embeddings()
        
        return self._embeddings
    
    # generate embedding
    def generate_embedding(self, text: str):
        '''
        Generate an embedding for a given text using the configured embedding
        model.

        Args:
            text (str): Text to generate embedding for.
        
        Returns:
            list: Embedding.
        '''
        return self.provider.embed([text])[0]

    # load embeddings
    def load_embeddings_from_csv(self, embeddings_path: str,
        columns: List, **kwargs):
        '''
        Load embeddings from csv file.

        Args:
            embeddings_path (str): Path to csv file containing embeddings.
            columns (List): List of columns specifying the embeddings.
            **kwargs: Keyword arguments for pandas read_csv method.
        
        Returns:
            list: Embeddings.
        '''
        data = pd.read_csv(embeddings_path, **kwargs)
        for col in columns:
            data[col] = data[col].apply(literal_eval)
        
        return data
