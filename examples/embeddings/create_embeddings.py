# -*- coding: utf-8 -*-

"""
"""

# import modules
import time
import pandas as pd

# import osintgpt modules
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# Init
text = f'''
Init program at {time.ctime()}

Example -> creating embeddings
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
sample = dataset.sample(n=1).copy()

# collect text data
text_data = sample['Plot'].tolist()

# load text to embedding generator
embedding_generator.load_text(text_data)

# count tokens and calculate estimated cost
n_tokens = embedding_generator.count_tokens()

# show values
print (f'Number of tokens: {n_tokens}')
print ('')

# get embeddings
embeddings = embedding_generator.embeddings

# display sample
print ('Sample -> Text:')
print (text_data[0])

print ('Sample -> Embeddings:')
print (embeddings[0][:10])
print ('')
print ('')


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
