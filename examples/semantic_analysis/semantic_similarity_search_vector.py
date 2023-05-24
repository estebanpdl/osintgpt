# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.semantic_operations import SemanticOperations
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example -> SemanticOperations -> Semantic similarity search: Vector
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
SemanticOperations connection
It will initialize the OpenAIGPT class
'''
operations = SemanticOperations(env_file_path)

'''
Qdrant connection
'''
qdrant = Qdrant(env_file_path)
query = 'Sheldon explores a new theory on quantum physics'
collection_name = 'big_bang_theory'

# recursive search
response = operations.semantic_similarity_search(
    query=query,
    vector_engine=qdrant,
    payload_ref_text_key='text_data',
    score_threshold=0.85,
    score_based_on_initial_query=True,
    collection_name=collection_name
)

# display results
print (response)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
