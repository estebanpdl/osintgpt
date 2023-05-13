# -*- coding: utf-8 -*-

"""
This example script demonstrates how to use the Pinecone class from the osintgpt
project to create a new index in Pinecone's vector search service. The script
initializes the Pinecone class with the required configuration, connects to the
Pinecone service, and creates a new index with the specified index name, dimension,
and metric.
"""

# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Pinecone

# Init
text = f'''
Init program at {time.ctime()}

Example -> Pinecone create_index
'''
print (text)

# pinecone config -> env file path
env_file_path = '../config/.env'
pinecone = Pinecone(env_file_path)
client = pinecone.get_client()

# create index
index_name = 'big_bang_theory'
dimension = 768
metric = 'cosine'
pinecone.create_index(index_name, dimension, metric)
print ('Index created!')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
