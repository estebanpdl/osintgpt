# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> Semantic similarity search: Dataframe
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIGPT connection
'''
gpt = OpenAIGPT(env_file_path)

# load embeddings from csv file
path = '../data/embeddings.csv'
df = gpt.load_embeddings_from_csv(
    file_path=path,
    columns=['embeddings'],
    encoding='utf-8',
    sep=',',
    low_memory=False
)

query = 'Sheldon explores a new theory on quantum physics'

# recursive search
response = gpt.semantic_similarity_search(
    query=query,
    df=df,
    payload_ref_text_key='text_data',
    score_threshold=0.80,
    top_k=10,
    score_based_on_initial_query=True
)

# display results
print (response)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
