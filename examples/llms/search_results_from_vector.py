# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.llms import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> search results from vector
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

# search results from vector
response = gpt.search_results_from_vector(
    vector_engine=qdrant,
    query=query,
    top_k=2,
    collection_name=collection_name
)

# get results
results = response['results']

# print results
for res in results:
    # add string to content and give it a new line
    content = res.payload['text_data']
    score = res.score
    print (f'> {content} -> {score}')


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
