# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# import exceptions
from qdrant_client.http.exceptions import UnexpectedResponse

# Init
text = f'''
Init program at {time.ctime()}

Testing Qdrant -> get vectors
'''
print (text)

# qdrant config -> env file path
env_file_path = '../../config/.env'
qdrant = Qdrant(env_file_path)

# get vectors
collection_name = 'narratives'
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
