# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> Estimated Prompt Cost
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
    query, qdrant, top_k=50, collection_name=collection_name
)

content = ''
for i, res in enumerate(response):
    # add string to content and give it a new line
    content += f'{res.payload["text_data"]}\n'

# build prompt
prompt = f'''
Summarize the text delimited by triple backticks in one paragraph.
Determine five topics that are being discussed in the same text

Text: ```{content}```
'''

tokens = gpt.count_tokens(prompt)
cost = gpt.estimated_prompt_cost(prompt)

# display results
print (f'Prompt tokens: {tokens}')
print (f'Prompt cost: {cost}')
print ('')

# disclaimer
print ('The estimated prompt cost will be higher when using model completion.')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
