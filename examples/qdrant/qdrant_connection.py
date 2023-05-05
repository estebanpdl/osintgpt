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

# qdrant config -> env file path
env_file_path = '../../config/.env'
qdrant = Qdrant(env_file_path)
client = qdrant.get_client()
print (client)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
