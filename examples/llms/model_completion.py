# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.llms import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> model completion
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

# display new lines
print ('')
print ('')

# build prompt
prompt = f'''
Summarize the text delimited by triple backticks in one paragraph. Next, identify
five topics that are being discussed in the same text.

Text: ```{content}```

Your response must follow this format:

1. Paragraph.
2. The topics separated by comma.
'''

# model completion
result = gpt.get_model_completion(prompt)
print ('')
print ('Model completion:\n')
print (result)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
