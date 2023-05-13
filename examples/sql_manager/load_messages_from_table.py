# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.databases import SQLDatabaseManager

# Init
text = f'''
Init program at {time.ctime()}

Example -> SQLDatabaseManager -> load messages from table
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
SQLDatabaseManager connection
'''
sql_manager = SQLDatabaseManager(env_file_path)

# load messages from table
ref_id = '13f30bff0eda41209334f98595053c09' # change this value to a valid ref_id
messages = sql_manager.load_messages_from_chat_gpt_conversations(
    ref_id=ref_id
)

# display messages
print ('')
print ('Messages:')
print (messages)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
