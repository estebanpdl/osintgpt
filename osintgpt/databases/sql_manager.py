# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: sql_manager.py
# Description: SQLDatabaseManager provides an abstracted interface for interacting
#   with various SQL databases. The class encapsulates database connection,
#   table creation, and data insertion operations.
# =================================================================================

# import modules
import sqlite3

# import submodules
from sqlite3 import Error

# type hints
from typing import List, Optional, Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

# SQLDatabaseManager class
class SQLDatabaseManager(object):
    '''
    SQLDatabaseManager class.

    This class provides an abstracted interface for interacting with various SQL
    databases.
    '''
    def __init__(self, config: Union[Settings, str]):
        '''
        Initializes the instance of the class.

        Args:
            config (Union[Settings, str]): Settings, or a path to a .env file \
                (deprecated).

        Raises:
            MissingEnvironmentVariableError: If 'sql_db_file_path' has no value.
        '''
        # settings
        self.settings = resolve_settings(config).require('sql_db_file_path')

        # set database file path
        self.db_file = self.settings.sql_db_file_path

        # set database connection
        self.conn = self.create_connection(self.db_file)

        # create main chat gpt index table
        self._create_chat_gpt_index_table()

        # create chat gpt conversations table
        self._create_chat_gpt_conversations_table()
    
    # create connection
    def create_connection(self, db_file: str):
        '''
        Create SQL connection.

        Args:
            db_file (str): Database file path.

        Returns:
            sqlite3.Connection: Database connection.
        '''
        # set connection
        conn = None

        # try to connect to database
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print (e)
        
        # return connection
        return conn

    # get connection'
    def get_connection(self):
        '''
        Get SQL connection.

        Returns:
            sqlite3.Connection: Database connection.
        '''
        # return connection
        return self.conn
    
    # create main chat gpt index table
    def _create_chat_gpt_index_table(self):
        '''
        Create table in SQL database.

        Returns:
            None
        '''
        # set cursor
        cursor = self.conn.cursor()

        # try to create table
        try:
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS chat_gpt_index (
                    id text NOT NULL PRIMARY KEY,
                    created_at VARCHAR (20) NOT NULL
                );
                '''
            )

            # commit changes
            self.conn.commit()
        
        except Error as e:
            print (f"The error '{e}' occurred")
    
    # create chat gpt conversations table
    def _create_chat_gpt_conversations_table(self):
        '''
        Create chat gpt conversations table.

        Returns:
            None
        '''
        # set cursor
        cursor = self.conn.cursor()

        # try to create table
        try:
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS chat_gpt_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_id text NOT NULL,
                    chat_id text NOT NULL,
                    role text NOT NULL,
                    message text NOT NULL,
                    FOREIGN KEY(ref_id) REFERENCES chat_gpt_index(id)
                );
                '''
            )

            # commit changes
            self.conn.commit()
        
        except Error as e:
            print (f"The error '{e}' occurred")
    
    # insert chat gpt data
    def insert_data_to_chat_gpt_index(self, id: str, created_at: str):
        '''
        Insert data to chat gpt index table.

        Args:
            id (str): Conversation id.
            created_at (str): Conversation date.
        
        Returns:
            None
        '''
        # set cursor
        cursor = self.conn.cursor()

        # try to insert data
        try:
            cursor.execute(
                '''
                INSERT INTO chat_gpt_index (id, created_at)
                VALUES (?, ?)
                ''',
                (id, created_at)
            )

            # commit changes
            self.conn.commit()
        
        except Error as e:
            print (f"The error '{e}' occurred")
            self.conn.rollback()
    
    # insert chat gpt conversations
    def insert_data_to_chat_gpt_conversations(self, ref_id: str, chat_id: str,
        role: str, message: str):
        '''
        Insert data to chat gpt conversations table.

        Args:
            ref_id (str): Conversation id.
            chat_id (str): OpenAI's GPT response chat id.
            role (str): OpenAI's GPT role (e.g., user, assistant, system).
            message (str): OpenAI's GPT response message.
        
        Returns:
            None
        '''
        # set cursor
        cursor = self.conn.cursor()

        # try to insert data
        try:
            cursor.execute(
                '''
                INSERT INTO chat_gpt_conversations (ref_id, chat_id, role, message)
                VALUES (?, ?, ?, ?)
                ''',
                (ref_id, chat_id, role, message)
            )

            # commit changes
            self.conn.commit()
        
        except Error as e:
            print (f"The error '{e}' occurred")
            self.conn.rollback()
    
    # load messages from chat gpt conversations table
    def load_messages_from_chat_gpt_conversations(self, ref_id: str):
        '''
        Load messages from chat gpt conversations table.

        Args:
            ref_id (str): Conversation id.
        
        Returns:
            dict: Messages object from SQL database matching a conversation ref id.

            example:
            obj = {
                'ref_id': ref_id,
                'messages': [
                    {
                        'role': role,
                        'content': message
                    }
                ]
            }
        '''
        # set cursor
        cursor = self.conn.cursor()

        # set messages
        messages = []

        # try to load messages
        try:
            cursor.execute(
                '''
                SELECT role, message FROM chat_gpt_conversations
                WHERE ref_id = ?
                ''',
                (ref_id,)
            )

            # fetch messages
            messages = cursor.fetchall()

            # convert messages to dict -> {role: role, content: message}
            messages = [
                {
                    'role': message[0], 'content': message[1]
                } for message in messages
            ]

            # commit changes
            self.conn.commit()

            # return messages
            obj = {
                'ref_id': ref_id,
                'messages': messages
            }
            return obj

        except Error as e:
            print (f"The error '{e}' occurred")
            self.conn.rollback()
        
        # return messages
        obj = {
            'ref_id': ref_id,
            'messages': messages
        }
        return obj
