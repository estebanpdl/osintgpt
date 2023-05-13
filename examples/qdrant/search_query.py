# -*- coding: utf-8 -*-

"""
"""

# import modules
import time
import pandas as pd

# import osintgpt modules
from osintgpt.vector_store import Qdrant
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# Init
text = f'''
Init program at {time.ctime()}

Example -> search query
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIEmbeddingGenerator
'''
embedding_generator = OpenAIEmbeddingGenerator(env_file_path)



'''
Generate_embeddings
'''
# query text
query = 'Sheldon explores a new theory on quantum physics'
embedding = embedding_generator.generate_embedding(query)

print ('')
print (query)
print ('Finding similar text in collection based on query...')
print ('')
print ('')

'''
Qdrant
'''
qdrant = Qdrant(env_file_path)

'''
Search query
'''
collection_name = 'big_bang_theory'
results = qdrant.search_query(
    embedded_query=embedding,
    collection_name=collection_name,
    top_k=10
)

# print results
using_score = False
if using_score:
    for i, res in enumerate(results):
        print(f'{i + 1}. {res.payload["text_data"]} (Score: {round(res.score, 3)})')
else:
    content = ''
    for i, res in enumerate(results):
        # add text data to content and give it a new line
        content += f'{res.payload["text_data"]}\n'
    
    # display content
    print (content)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
