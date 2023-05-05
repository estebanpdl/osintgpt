# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Pinecone

# Init
text = f'''
Init program at {time.ctime()}

Testing Pinecone -> create_index
'''
print (text)

# pinecone config -> env file path
env_file_path = '../../config/.env'
pinecone = Pinecone(env_file_path)
client = pinecone.get_client()

# create index
index_name = 'testindex'
dimension = 768
metric = 'cosine'
pinecone.create_index(index_name, dimension, metric)
print ('Index created!')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
