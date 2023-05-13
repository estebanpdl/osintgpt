# -*- coding: utf-8 -*-

"""
This example script demonstrates how to use the Qdrant class from the osintgpt
project to establish a connection to the Qdrant service. The script initializes
the Qdrant class with the required configuration, and then obtains a Qdrant client
object by calling the get_client method. 
"""

# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example Qdrant -> connection
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)
client = qdrant.get_client()
print (client)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
