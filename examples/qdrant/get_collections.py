# -*- coding: utf-8 -*-

"""
This example script demonstrates how to use the Qdrant class from the osintgpt
project to retrieve a list of all collections stored in the Qdrant service.
The script initializes the Qdrant class with the required configuration, connects
to the Qdrant service, and retrieves the list of collections. 
"""

# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example Qdrant -> get collections
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)
collections = qdrant.get_collections()
print (collections)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
