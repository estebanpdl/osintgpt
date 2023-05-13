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
ref_id = '6bf3f96122c94170ad0d90fa5358ccbd' # change this value to a valid ref_id
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

# build prompt
prompt = f'''
Based on the topics highlighted, please add a character name linked to each topic.
'''

# model completion
result = gpt.get_model_completion(prompt, messages=messages)
print ('')
print ('')
print ('Model completion:')
print ('')
print (result)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
