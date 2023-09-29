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
from typing import Union, Optional, List, Dict

# import osintgpt vector stores
from osintgpt.vector_store import BaseVectorEngine, Qdrant

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
    def __init__(self, env_file_path: str):
        '''
        Initializes the instance of the class.

        Args:
            env_file_path (str): Path to the file containing environment variables.
        
        Raises:
            MissingEnvironmentVariableError: If either 'OPENAI_API_KEY' or \
                'OPENAI_GPT_MODEL' is not found in the environment variables.
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
        self.SQL_UNIQUE_ID_INSERTED = False
        
    # get openai api key
    def get_openai_api_key(self):
        '''
        Get OpenAI API key.

        Returns:
            str: OpenAI API key.
        '''
        return self.OPENAI_API_KEY
    
    # load embeddings
    def load_embeddings_from_csv(self, file_path: str,
        columns: List, **kwargs):
        '''
        Load embeddings from csv file.

        Args:
            file_path (str): CSV file path.
            columns (List): List of columns specifying the embeddings.
            **kwargs: Keyword arguments for pandas read_csv method.
        
        Returns:
            pd.DataFrame: Pandas dataframe.
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
        Load embeddings from dataframe.

        Args:
            dataframe (pd.DataFrame): Pandas dataframe.
            columns (List): List of columns specifying the embeddings.
        
        Returns:
            pd.DataFrame: Pandas dataframe.
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
        Get embeddings from `self._embeddings` property.

        Args:
            column (str): Column name containing the embeddings.

        Returns:
            list: List of embeddings.
        '''
        return self._embeddings[column]

    # get embeddings dataframe
    def get_embeddings_dataframe(self):
        '''
        Get embeddings dataframe.

        Returns:
            pd.DataFrame: Pandas dataframe.
        '''
        if not hasattr(self, '_embeddings'):
            raise AttributeError('No embeddings loaded. Please load embeddings.')
        
        return pd.DataFrame(self._embeddings)

    # load search top k results from vector
    def search_results_from_vector(self, vector_engine: BaseVectorEngine,
        query: Optional[str] = None, embeddings: Optional[List] = None,
        top_k: int = 10, **kwargs):
        '''
        Search top k results from vector database.

        Args:
            vector_engine (BaseVectorEngine): Vector engine.
            query (Optional[str]): Query for the search process.
            embeddings (Optional[List]): List of embeddings.
            top_k (int): Top k results to be retrieved.
            **kwargs: Keyword arguments for the vector engine search query method.

        Returns:
            search_results (Dict): Dictionary containing the search results, \
                with the following keys: 'query', 'query_embedding', 'results'.
        '''
        # check if query or embeddings are provided
        if query is None and embeddings is None:
            raise ValueError('Either query or embeddings must be provided.')

        if not isinstance(vector_engine, BaseVectorEngine):
            supported_vector_engines = [
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
        if query is not None:
            embedding_generator = OpenAIEmbeddingGenerator(self.env_file_path)
            query_embedding = embedding_generator.generate_embedding(query)
        else:
            query_embedding = embeddings

        # search results
        search_results = vector_engine.search_query(
            query_embedding,
            top_k=top_k,
            **kwargs
        )
        
        return {
            'query': query,
            'query_embedding': query_embedding,
            'results': search_results
        }
    
    # relatedness function
    def _relatedness_fn(self, x, y):
        '''
        Relatedness function.

        This function is used to calculate the relatedness between two embeddings.
        It uses the cosine distance to calculate the relatedness.

        Args:
            x (List[float]): List of embeddings.
            y (List[float]): List of embeddings.
        
        Returns:
            float: Relatedness. 1.0 is most similar, 0.0 is least similar.
        '''
        return 1 - spatial.distance.cosine(x, y)
    
    # load search top k results from dataframe
    def search_results_from_dataframe(self, df: pd.DataFrame,
        query: Optional[str] = None, embeddings: Optional[List] = None,
        top_k: int = 10, embeddings_target_column: str = 'embeddings',
        text_target_column: str = 'text', extract_sentence_details: bool = False):
        '''
        Search top k results from dataframe.
        
        Args:
            df (pd.DataFrame): Pandas dataframe containing the embeddings. 
            query (Optional[str]): Query for the search process.
            embeddings (Optional[List]): List of embeddings.
            top_k (int): Top k results to be retrieved.
            embeddings_target_column (str): Embeddings target column.
            text_target_column (str): Text target column.
        
        Returns:
            List[Tuple[str, float]]: List of tuples containing the string and score.
        '''
        # check if query or embeddings are provided
        if query is None and embeddings is None:
            raise ValueError('Either query or embeddings must be provided.')
        
        # OpenAIEmbeddingGenerator instance
        if query is not None:
            embedding_generator = OpenAIEmbeddingGenerator(self.env_file_path)
            if extract_sentence_details:
                '''
                This method will try to extract details from query.
                Only Subject or topics will be embbeded.
                The premise is that, based on this approach, we can improve
                    similarity results.
                '''
                response = self.analyze_sentence_details(query)
                response = literal_eval(response)
                try:
                    query = response['Subject or topics']
                except TypeError:
                    pass
            
            query_embedding = embedding_generator.generate_embedding(query)
        else:
            query_embedding = embeddings
        
        strings_and_relatednesses = [
            (
                row[embeddings_target_column],
                row[text_target_column],
                self._relatedness_fn(query_embedding, row[embeddings_target_column])
            )
            for _, row in df.iterrows()
        ]

        strings_and_relatednesses.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'query': query,
            'query_embedding': query_embedding,
            'results': strings_and_relatednesses[:top_k]
        }

    # count tokens < GPT model >
    def count_tokens(self, prompt: str):
        '''
        Count tokens
        It counts the number of tokens in the data.

        Args:
            prompt (str): The input prompt for the GPT model.

        Returns:
            int: Number of tokens.
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

        Args:
            prompt (str): The input prompt for the GPT model.

        Returns:
            float: GPT Model estimated cost.
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
    def _get_completion_response_id(self, response: Dict):
        '''
        Get completion response id.

        Args:
            response (Dict): GPT Model response.
        
        Returns:
            str: GPT Model response id.
        '''
        return response['id']

    # get completion response usage
    def _get_completion_response_usage(self, response: Dict):
        '''
        Get completion response usage.

        Args:
            response (Dict): GPT Model response.
        
        Returns:
            dict: GPT Model response usage.
        '''
        return response['usage']
    
    # get completion response role & message
    def _get_completion_response_role_and_message(self, response: Dict):
        '''
        Get completion response role & message.

        Args:
            response (Dict): GPT Model response.
        
        Returns:
            Tuple[str, str]: A tuple where the first element is the response role
            and the second element is the response message.
        '''
        role = response['choices'][0]['message']['role']
        message = response['choices'][0]['message']['content']

        return role, message
    
    # generate unique id
    def _generate_unique_id(self):
        '''
        Generate unique id for SQL database.
        
        Returns:
            str: SQL unique id.
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
        Insert system prompt into sql database.

        Args:
            prompt (str): The input prompt for the GPT model.

        Returns:
            None
        '''
        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.env_file_path)

        # insert prompt into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID, 'system-init', 'system', prompt
        )
    
    # insert user prompt into sql database
    def insert_user_prompt_into_sql_database(self, response: Dict, prompt: str):
        '''
        Insert user prompt into sql database.

        Args:
            response (Dict): GPT Model response.
            prompt (str): The input prompt for the GPT model.
        
        Returns:
            None
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
    def insert_completion_response_into_sql_database(self, response: Dict):
        '''
        Insert completion response into sql database.

        Args:
            response (Dict): GPT Model response.
        
        Returns:
            None
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
        if not self.SQL_UNIQUE_ID_INSERTED:
            sql_manager.insert_data_to_chat_gpt_index(
                self.SQL_UNIQUE_ID,
                created_at
            )

            # set SQL_UNIQUE_ID_INSERTED to True
            self.SQL_UNIQUE_ID_INSERTED = True

        # insert response into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID,
            chat_id,
            role,
            message
        )
    
    # get GPT model completion when adding a system role
    def get_model_completion_using_system_role(self, messages: List[Dict],
        verbose: bool = True, **kwargs):
        '''
        Get GPT model completion.

        Args:
            messages (List[Dict]): A list of message objects. Each object \
                should be a dictionary containing 'role' and 'content'.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
            **kwargs: Keyword arguments for OpenAI's create completion.
        
        Returns:
            str: GPT completion response.
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        openai.api_key = self.OPENAI_API_KEY

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # get completion response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            **kwargs
        )

        # insert system prompt into sql database
        system_prompt = messages[0]['content']
        self.insert_system_prompt_into_sql_database(system_prompt)

        # insert user prompt into sql database
        user_prompt = messages[1]['content']
        self.insert_user_prompt_into_sql_database(response, user_prompt)

        # insert response into sql database
        self.insert_completion_response_into_sql_database(response)

        # display main values
        if verbose:
            print('Response id: ', self._get_completion_response_id(response))
            for key, value in self._get_completion_response_usage(response).items():
                print(f'{key}: {value}')
        
        return response['choices'][0].message['content']

    # get GPT model completion
    def get_model_completion(self, prompt: str,
        messages: Optional[Union[List, Dict]] = None, temperature: float = 0,
        verbose: bool = True):
        '''
        Get GPT model completion.

        Args:
            prompt (str): The input prompt for the GPT model.
            messages (Union[List, Dict], optional): A list or dictionary of \
                messages. If it's a list, it should be a list of message objects. \
                If it's a dictionary, it should contain 'ref_id' and 'messages'.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
            
        Returns:
            str: GPT completion response.
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
                {'role': 'user', 'content': prompt}
            ]
        else:
            if type(messages) == dict:
                '''
                Since messages are provided, we assume that the SQL_UNIQUE_ID
                has already been inserted into the database.

                Pass ref_id to SQL_UNIQUE_ID.
                Set SQL_UNIQUE_ID_INSERTED to True.
                '''
                self.SQL_UNIQUE_ID = messages['ref_id']
                self.SQL_UNIQUE_ID_INSERTED = True

                # build messages
                messages = messages['messages'] + [
                    {'role': 'user', 'content': prompt}
                ]
            else:
                pass

        # get completion response
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
    def interactive_completion(self, prompt: Optional[str] = None,
        messages: Optional[Dict] = None, temperature: float = 0,
        verbose: bool = False):
        '''
        Interactive completion. Interact with the GPT model using the command line.

        Args:
            prompt (str, optional): The input prompt for the GPT model.
            messages (List[Dict], optional): A list of message objects. Each object \
                should be a dictionary containing 'role' and 'content'.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
        
        Returns:
            None
        '''
        # Check that at least one of prompt or messages is provided
        if prompt is None and messages is None:
            raise ValueError('Either prompt or messages must be provided.')
        
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

            '''
            Since messages are provided, we assume that the SQL_UNIQUE_ID
            has already been inserted into the database.

            Pass ref_id to SQL_UNIQUE_ID.
            Set SQL_UNIQUE_ID_INSERTED to True.
            '''
            self.SQL_UNIQUE_ID = messages['ref_id']
            self.SQL_UNIQUE_ID_INSERTED = True

            # build messages
            messages = messages['messages']

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
                verbose=verbose
            )

            # accumulate messages
            msg = {'role': 'assistant', 'content': gpt_response}
            messages.append(msg)

            print (f'{model}: ', gpt_response)
    
    # Analyze sentence details
    def analyze_sentence_details(self, sentence: str, temperature: float = 0):
        '''
        Analyzes the provided sentence to identify its primary language,
        the central request or question, and the main theme or subject matter
        associated with that request.

        Args:
            sentence (str): The input sentence to be analyzed.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
        
        Returns:
            dict: A dictionary containing keys "Language", "Input request", and
            "Subject or topics" with their respective identified values.
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')
        
        openai.api_key = self.OPENAI_API_KEY

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')
        
        model = self.OPENAI_GPT_MODEL

        # generate system message role
        system_role = '''
        As a Large Language Model, you specialize in dissecting sentences to unearth
        the core components within them. When presented with a sentence:
        1. Determine the primary language in which the sentence is written.
        2. Extract the central request, input, or question that requires assistance.
        3. Identify the main topic/s or subject/s associated with the central request.

        Compile your findings into a JSON response, highlighting these three
        essential aspects.

        Example input: "¿Cuáles son los beneficios de la energía solar?"
        Expected output:
        ```
        {
            "Language": "Spanish",
            "Input request": "¿Cuáles son los beneficios",
            "Subject or topics": "energía solar"
        }
        ```
        '''
        # build messages
        messages = [
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': sentence}
        ]

        # get completion response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response['choices'][0].message['content']
