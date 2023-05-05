# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Testing Qdrant -> get collections
'''
print (text)

# test class
env_file_path = '.env'

qdrant = Qdrant(env_file_path)
collections = qdrant.get_collections()
print (collections)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
