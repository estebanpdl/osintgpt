# -*- coding: utf-8 -*-

"""
"""

# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Example Qdrant -> get collection info
'''
print (text)

# qdrant config -> env file path
env_file_path = '../config/.env'
qdrant = Qdrant(env_file_path)

collection_name = 'big_bang_theory'
info = qdrant.get_collection_info(collection_name=collection_name)
print (info)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)