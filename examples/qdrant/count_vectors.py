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

# test class
env_file_path = '.env'

qdrant = Qdrant(env_file_path)

collection_name = 'narratives'

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
