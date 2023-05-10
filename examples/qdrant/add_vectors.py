# -*- coding: utf-8 -*-

"""
"""

# import modules
import time
import pandas as pd

# import osintgpt modules
from osintgpt.vector_store import Qdrant
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# Init
text = f'''
Init program at {time.ctime()}

Testing -> add vectors to Qdrant collection
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIEmbeddingGenerator
'''
embedding_generator = OpenAIEmbeddingGenerator(env_file_path)

'''
Read csv file and process dataset
Load text
'''
# read dataset
path = '../data/big_bang_theory_imdb.csv'
data = pd.read_csv(path, encoding='utf-8', low_memory=False)

# process dataset
dataset = data[data['Plot'].notnull()].copy()

# collect a random sample
sample = dataset.sample(n=50).copy()

# collect text data
text_data = sample['Plot'].tolist()

# load text to embedding generator
embedding_generator.load_text(text_data)

# count tokens and calculate estimated cost
n_tokens = embedding_generator.count_tokens()
estimated_cost = embedding_generator.calculate_embeddings_estimated_cost()

# show values
print (f'Number of tokens: {n_tokens}')
print (f'Estimated cost: {estimated_cost}')
print ('')

# get embeddings
embeddings = embedding_generator.embeddings

# create dataframe
obj = {
    'text_data': text_data,
    'embeddings': embeddings
}

df = pd.DataFrame(obj)


'''
Qdrant
'''
qdrant = Qdrant(env_file_path)

# vector config
vector_size = len(df['embeddings'][0])
payload = df.to_dict(orient='records')
collection_name = 'big_bang_theory'

'''
Create collections
'''
qdrant.create_collection(collection_name, vector_size)

'''
Add vectors
'''
qdrant.add_vectors(
    collection_name=collection_name,
    vectors=embeddings,
    payload=payload
)

'''
Count vectors
'''
count = qdrant.count_vectors(collection_name)
print(f'Collection: {collection_name} -> vectors {count}')
print ('')


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
