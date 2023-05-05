# import modules
import time
import pandas as pd

# import osintgpt modules
from osintgpt.embedding import OpenAIEmbeddingGenerator

# Init
text = f'''
Init program at {time.ctime()}

Testing OpenAIEmbeddingGenerator -> count tokens and calculate estimated cost
Estimated cost based on model: text-embedding-ada-002
'''
print (text)

# openai config -> env file path
env_file_path = '../../config/.env'

'''

Test: OpenAIEmbeddingGenerator
'''
embedding_generator = OpenAIEmbeddingGenerator(env_file_path)


'''

Read csv file and process dataset
Load text

'''
# read dataset
path = '../../data/big_bang_theory_imdb.csv'
data = pd.read_csv(path, encoding='utf-8', low_memory=False)

# process dataset
dataset = data[data['Plot'].notnull()].copy()

# collect text data
text_data = dataset['Plot'].tolist()

# load text
embedding_generator.load_text(text_data)

# count tokens and calculate estimated cost
n_tokens = embedding_generator.count_tokens()
estimated_cost = embedding_generator.calculate_embeddings_estimated_cost()

# show values
print (f'Number of tokens: {n_tokens}')
print (f'Estimated cost: {estimated_cost}')
print ('')

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
