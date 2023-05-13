# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT
from osintgpt.databases import SQLDatabaseManager

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> model completion from messages
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIGPT connection
'''
gpt = OpenAIGPT(env_file_path)

'''
SQLDatabaseManager connection
'''
sql_manager = SQLDatabaseManager(env_file_path)

# load messages from table
ref_id = '6e917619dc6a40b18385467fc56487c6' # change this value to a valid ref_id
messages = sql_manager.load_messages_from_chat_gpt_conversations(
    ref_id=ref_id
)

'''
messages output:

messages = {
    'ref_id': ref_id,
    'messages': [
        {
            role: role,
            content: message
        },
        ...
    ]
}
'''

# model completion
gpt.interactive_completion(messages=messages)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
