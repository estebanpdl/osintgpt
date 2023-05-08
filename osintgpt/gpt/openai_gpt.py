# -*- coding: utf-8 -*-

# =============================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: openai_gpt.py
# Description: The openai_gpt.py file contains a class that provides an interface 
#   to interact with OpenAI's GPT models, allowing users to read embeddings, and
#   ask questions about the data.
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
    def load_embeddings(self, vector_engine=None, file_path: str = None):
        '''
        Load embeddings

        Args:
            file_path (str): File path
        '''
        if vector_engine is not None:
            # load embeddings from the vector engine
            pass
        elif file_path is not None:
            # load embeddings from the file
            pass
        else:
            raise ValueError('Either vector_engine or file_path must be provided')

