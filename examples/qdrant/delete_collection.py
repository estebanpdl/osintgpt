# -*- coding: utf-8 -*-

"""
"""

# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# import exceptions
from qdrant_client.http.exceptions import UnexpectedResponse

# Init
text = f'''
Init program at {time.ctime()}

Testing Qdrant -> delete collection
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)

# get vectors
collection_name = 'test_dataset_embed_beta'
qdrant.delete_collection(collection_name=collection_name)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
