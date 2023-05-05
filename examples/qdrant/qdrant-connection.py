# import modules
import time

# import osintgpt modules
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Testing Qdrant -> connection
'''
print (text)

# test class
env_file_path = '.env'

qdrant = Qdrant(env_file_path)
client = qdrant.get_client()
print (client)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
