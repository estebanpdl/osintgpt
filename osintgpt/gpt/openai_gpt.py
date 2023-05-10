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
import datetime
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
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# import database manager
from osintgpt.databases import SQLDatabaseManager

# import utils
from osintgpt.utils import create_unique_id

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
        
        # set SQL unique id
        self.SQL_UNIQUE_ID = self._generate_unique_id()
        
    # get openai api key
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

        return dataframe

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

    # load search top k results from vector
    def search_results_from_vector(self, query: str, vector_engine: BaseVectorEngine,
        top_k: int = 10, **kwargs):
        '''
        '''
        if not isinstance(vector_engine, BaseVectorEngine):
            supported_vector_engines = [
                Pinecone,
                Qdrant
            ]
            supported_vector_engine_names = ', '.join(
                [engine.__name__ for engine in supported_vector_engines]
            )

            # build message
            msg_a = 'Invalid vector engine provided'
            msg_b = 'Must be an instance of one of the following classes:'
            message = f'{msg_a}. {msg_b} {supported_vector_engine_names}.'
            raise ValueError(message)
        
        # OpenAIEmbeddingGenerator instance
        embedding_generator = OpenAIEmbeddingGenerator(self.env_file_path)
        query_embedding = embedding_generator.generate_embedding(query)

        # search results
        search_results = vector_engine.search_query(
            query_embedding,
            top_k=top_k,
            **kwargs
        )
        
        return search_results
    
    # load search top k results from dataframe
    def search_results_from_dataframe(self, query: str, df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_k: int = 10, embeddings_target_column: str = 'embeddings',
        text_target_column: str = 'text'):
        '''
        Returns a list of strings and relatednesses, sorted from most related
        to least.
        '''
        embedding_generator = OpenAIEmbeddingGenerator(self.env_file_path)
        query_embedding = embedding_generator.generate_embedding(query)
        
        strings_and_relatednesses = [
            (
                row[text_target_column],
                relatedness_fn(query_embedding, row[embeddings_target_column])
            )
            for _, row in df.iterrows()
        ]

        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        return strings[:top_k], relatednesses[:top_k]
    
    # count tokens < GPT model >
    def count_tokens(self, prompt: str):
        '''
        Count tokens

        It counts the number of tokens in the data.

        Returns:
            num_tokens (int): Number of tokens
        '''
        # get model
        model = self.OPENAI_GPT_MODEL
        encoding = tiktoken.encoding_for_model(model)

        # count tokens
        tokens = encoding.encode(prompt)
        num_tokens = len(tokens)

        return num_tokens

    # calculate completion response usage cost
    def estimated_prompt_cost(self, prompt: str):
        '''
        It calculates the estimated cost of the prompt based on the number of
        tokens.
        
        Costs are based on the OpenAI gpt-3.5-turbo or gpt-4 models.

        Returns:
            estimated_cost (float): Estimated cost
        '''
        model = self.OPENAI_GPT_MODEL

        # get number of tokens
        num_tokens = self.count_tokens(prompt)

        # dict based on model costs
        model_costs = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03
        }

        # calculate estimated cost
        estimated_cost = (num_tokens / 1000) * model_costs[model]
        return estimated_cost
    
    # get completion response id
    def _get_completion_response_id(self, response):
        '''
        Get completion response id

        Args:
            response (dict): Response
        
        Returns:
            id (str): Response id
        '''
        return response['id']

    # get completion response usage
    def _get_completion_response_usage(self, response):
        '''
        Get completion response usage

        Args:
            response (dict): Response
        
        Returns:
            usage (dict): Response usage
        '''
        return response['usage']
    
    # get completion response role & message
    def _get_completion_response_role_and_message(self, response):
        '''
        Get completion response role & message

        Args:
            response (dict): Response
        
        Returns:
            role (str): Response role
            message (str): Response message
        '''
        role = response['choices'][0]['message']['role']
        message = response['choices'][0]['message']['content']

        return role, message
    
    # generate unique id
    def _generate_unique_id(self):
        '''
        Generate unique id for sql database
        
        Returns:
            id (str): Unique id
        '''
        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.env_file_path)

        # get connection
        conn = sql_manager.get_connection()

        # get cursor
        cursor = conn.cursor()

        # get all ids from table > chat_gpt_index
        cursor.execute('SELECT id FROM chat_gpt_index')
        ids = cursor.fetchall()

        # convert ids to list
        ids = [id[0] for id in ids]

        # create unique id
        unique_id = create_unique_id(ids)
        return unique_id
    
    # insert system prompt into sql database
    def insert_system_prompt_into_sql_database(self, prompt: str):
        '''
        Insert system prompt into sql database

        Args:
            prompt (str): Prompt
        '''
        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.env_file_path)

        # insert prompt into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID, 'system-init', 'system', prompt
        )
    
    # insert user prompt into sql database
    def insert_user_prompt_into_sql_database(self, response: dict, prompt: str):
        '''
        Insert user prompt into sql database

        Args:
            response (dict): Response
            prompt (str): Prompt
        '''
        # get response id
        chat_id = self._get_completion_response_id(response)

        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.env_file_path)

        # insert prompt into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID, chat_id, 'user', prompt
        )
    
    # insert completion response into sql database
    def insert_completion_response_into_sql_database(self, response: dict):
        '''
        Insert completion response into sql database

        Args:
            response (dict): Response
        '''
        # get response id
        chat_id = self._get_completion_response_id(response)
        role, message = self._get_completion_response_role_and_message(response)
        
        # convert timestamp to %Y-%m-%d %H:%M:%S format
        created_at = datetime.datetime.fromtimestamp(
            response['created']
        ).strftime('%Y-%m-%d %H:%M:%S')

        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.env_file_path)

        # insert response into sql table > chat_gpt_index
        sql_manager.insert_data_to_chat_gpt_index(
            self.SQL_UNIQUE_ID,
            created_at
        )

        # insert response into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID,
            chat_id,
            role,
            message
        )

    # get GPT model completion
    def get_model_completion(self, prompt: str, messages: Optional[List] = None,
        temperature: float = 0, verbose: bool = True):
        '''
        Get GPT model completion

        Args:
            prompt (str): Prompt
            temperature (float): Temperature (default: 0)
            verbose (bool): Verbose mode (default: True)
            messages (Optional[List]): Messages (default: None)
        
        Returns:
            completion (str): Completion
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        openai.api_key = self.OPENAI_API_KEY

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # get completion
        messages = [
            {'role': 'user', 'content': prompt}
        ] if messages is None else messages

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        # insert user prompt into sql database
        self.insert_user_prompt_into_sql_database(response, prompt)

        # insert response into sql database
        self.insert_completion_response_into_sql_database(response)

        # display main values
        if verbose:
            print('Response id: ', self._get_completion_response_id(response))
            for key, value in self._get_completion_response_usage(response).items():
                print(f'{key}: {value}')
        
        return response['choices'][0].message['content']
    
    # interactive completion: role system
    def interactive_completion(self, prompt: str, messages: Optional[List] = None,
        temperature: float = 0):
        '''
        Interactive completion

        Args:
            prompt (str): Prompt
            temperature (float): Temperature (default: 0)
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')
        
        openai.api_key = self.OPENAI_API_KEY

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # build messages
        if messages is None:
            messages = [
                {'role': 'system', 'content': prompt}
            ]

            # insert system prompt into sql database
            self.insert_system_prompt_into_sql_database(prompt)
        else:
            messages = messages

        # interactive chat mode
        print ('Interactive chat mode with GPT. Type "exit" to quit.')
        print ('')
        while True:
            user_input = input('You: ')
            if user_input == 'exit':
                print ('')
                print ('Exiting interactive chat mode...')
                break
            
            # accumulate messages
            msg = {'role': 'user', 'content': user_input}
            messages.append(msg)
            
            # get completion
            gpt_response = self.get_model_completion(
                user_input,
                messages=messages,
                temperature=temperature,
                verbose=False
            )

            # accumulate messages
            msg = {'role': 'assistant', 'content': gpt_response}
            messages.append(msg)

            print (f'{model}: ', gpt_response)
