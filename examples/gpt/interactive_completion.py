# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Testing -> OpenAIGPT -> model completion
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIGPT connection
'''
gpt = OpenAIGPT(env_file_path)


'''
Qdrant connection
'''
qdrant = Qdrant(env_file_path)
query = 'Sheldon explores a new theory on quantum physics'
collection_name = 'big_bang_theory'

response = gpt.search_results_from_vector(
    query, qdrant, top_k=25, collection_name=collection_name
)

content = ''
for i, res in enumerate(response):
    # add string to content and give it a new line
    content += f'{res.payload["text_data"]}\n'

# build prompt
prompt = f'''
You are a TV series critic and skilled TV show analyzer. You will analyze inputs \
from one prestigious user. Your task is to convince the user to watch the TV show \
that you are analyzing.

Make sure to clarify all questions that the user may have about the TV show.
You respond in a short, very conversational friendly style.

Consider the following text delimited by triple backticks as the TV show summary.

Text: ```{content}```
'''

# interactive completion: role system
print ('interactive completion: role system.')
print ('')
gpt.interactive_completion(prompt)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
