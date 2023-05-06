# -*- coding: utf-8 -*-

"""
This example script showcases how to use the Qdrant class from the osintgpt project
to count the number of vectors in a Qdrant collection. The script initializes the
Qdrant class with the required configuration, connects to the Qdrant service, and
attempts to count the vectors in the specified collection. If the collection does
not exist, it prints a message indicating that the collection is not present.
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

Testing Qdrant -> count vectors
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)

# count vectors
collection_name = 'big_bang_theory'
try:
    count = qdrant.count_vectors(collection_name=collection_name)
    print (count)
except UnexpectedResponse:
    print ('> Collection does not exist')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
