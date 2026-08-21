# -*- coding: utf-8 -*-

# ===============================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: conversation.py
# Description: Reading fields off a completion response and persisting the
#   exchange to the SQLite conversation log.
# ===============================================================================

# import modules
import datetime

# type hints
from openai.types.chat import ChatCompletion

# import database manager
from osintgpt.databases import SQLDatabaseManager

# import utils
from osintgpt.utils import create_unique_id

# ConversationLogMixin class
class ConversationLogMixin(object):
    '''
    Response accessors and the conversation log they feed.
    '''
    # get completion response id
    def _get_completion_response_id(self, response: ChatCompletion):
        '''
        Get completion response id.

        Args:
            response (ChatCompletion): GPT Model response.

        Returns:
            str: GPT Model response id.
        '''
        return response.id

    # get completion response usage
    def _get_completion_response_usage(self, response: ChatCompletion):
        '''
        Get completion response usage.

        Args:
            response (ChatCompletion): GPT Model response.

        Returns:
            dict: GPT Model response usage.
        '''
        if response.usage is None:
            return {}

        return response.usage.model_dump(exclude_none=True)

    # get completion response role & message
    def _get_completion_response_role_and_message(self, response: ChatCompletion):
        '''
        Get completion response role & message.

        Args:
            response (ChatCompletion): GPT Model response.

        Returns:
            Tuple[str, str]: A tuple where the first element is the response role
            and the second element is the response message.
        '''
        role = response.choices[0].message.role
        message = response.choices[0].message.content

        return role, message

    # generate unique id
    def _generate_unique_id(self):
        '''
        Generate unique id for SQL database.
        
        Returns:
            str: SQL unique id.
        '''
        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.settings)

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
        sql_manager = SQLDatabaseManager(self.settings)

        # insert prompt into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID, 'system-init', 'system', prompt
        )

    # insert user prompt into sql database
    def insert_user_prompt_into_sql_database(self, response: ChatCompletion,
        prompt: str):
        '''
        Insert user prompt into sql database.

        Args:
            response (ChatCompletion): GPT Model response.
            prompt (str): The input prompt for the GPT model.
        
        Returns:
            None
        '''
        # get response id
        chat_id = self._get_completion_response_id(response)

        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.settings)

        # insert prompt into sql table > chat_gpt_conversations
        sql_manager.insert_data_to_chat_gpt_conversations(
            self.SQL_UNIQUE_ID, chat_id, 'user', prompt
        )

    # insert completion response into sql database
    def insert_completion_response_into_sql_database(self, response: ChatCompletion):
        '''
        Insert completion response into sql database.

        Args:
            response (ChatCompletion): GPT Model response.

        Returns:
            None
        '''
        # get response id
        chat_id = self._get_completion_response_id(response)
        role, message = self._get_completion_response_role_and_message(response)

        # convert timestamp to %Y-%m-%d %H:%M:%S format
        created_at = datetime.datetime.fromtimestamp(
            response.created
        ).strftime('%Y-%m-%d %H:%M:%S')

        # SQL database manager instance
        sql_manager = SQLDatabaseManager(self.settings)

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
