# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.llms import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> interactive completion
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
    vector_engine=qdrant,
    query=query,
    top_k=5,
    collection_name=collection_name
)

# content
content = ''

# get results
results = response['results']

# print results
for res in results:
    # add string to content and give it a new line
    text = res.payload['text_data']
    score = res.score
    print (f'> {text} -> {score}')
    content += f'{text}\n'

# build prompt
prompt = f'''
You are a knowledgeable critic and skillful analyzer of TV series. Your role \
aims to analyze inputs from a prestigious user, with the goal to persuade them \
to watch the TV show under analysis. The Analyzer addresses all user's queries \
about the TV show in a concise, friendly, conversational style, while maintaining \
strict focus on the provided text delimited by triple backticks.

Any question or comment out of the scope is responded with, 'Information is not \
provided in the context'.

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
