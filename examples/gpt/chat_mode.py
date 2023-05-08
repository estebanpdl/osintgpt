# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT
from osintgpt.vector_store import Qdrant

# Init
text = f'''
Init program at {time.ctime()}

Testing -> OpenAIGPT
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIGPT connection
'''
gpt = OpenAIGPT(env_file_path)


'''
Qdrant connection
'''
qdrant = Qdrant(env_file_path)


# load embeddings from csv file
path = '../data/embeddings.csv'
df = gpt.load_embeddings_from_csv(
    file_path=path,
    columns=['embeddings'],
    encoding='utf-8',
    sep=',',
    low_memory=False
)


strings, relatednesses = gpt.strings_ranked_by_relatedness(
    'Sheldon buys a new comic book', df, text_target_column='text_data'
)

for string, relatedness in zip(strings, relatednesses):
    print(string, relatedness)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
