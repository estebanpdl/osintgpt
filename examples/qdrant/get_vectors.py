# -*- coding: utf-8 -*-

"""
This example script demonstrates how to use the Qdrant class from the osintgpt
project to retrieve vectors from a specified collection stored in the Qdrant
service. The script initializes the Qdrant class with the required configuration,
connects to the Qdrant service, and attempts to retrieve the vectors from the
specified collection.
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

Example Qdrant -> get vectors
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)

# get vectors
collection_name = 'big_bang_theory'
try:
    vectors = qdrant.get_vectors(collection_name=collection_name)
    print (vectors)
except UnexpectedResponse:
    print ('> Collection does not exist')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
