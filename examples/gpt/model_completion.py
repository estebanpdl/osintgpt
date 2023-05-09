# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Testing -> OpenAIGPT -> search results from vector
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
Determine five topics that are being discussed in the same text.

Text: ```{content}```

Follow the next format:

Paragraph.
Topics separated by comma.
Finally, do (3*2)+1.
'''

# model completion
result = gpt.get_model_completion(prompt)
print ('')
print ('Model completion:')
print (result)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
