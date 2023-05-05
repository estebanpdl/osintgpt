# import modules
import time
import pandas as pd

# import osintgpt modules
from osintgpt.embedding import OpenAIEmbeddingGenerator
from osintgpt.vector_store import Qdrant


# Init
text = f'''
Init program at {time.ctime()}
'''
print (text)


# openai config
args = {
    'env_file_path': '../config/.env'
}


'''

Test: OpenAIEmbeddingGenerator
'''
embedding_generator = OpenAIEmbeddingGenerator(**args)


# test values
key = embedding_generator.get_openai_api_key()
model = embedding_generator.get_openai_gpt_model()

# print values
print (f'API Key: {key}')
print (f'GPT Model: {model}')
print ('')

'''

Test: load_text

'''
# read dataset
path = './data/msgs_dataset.csv'
data = pd.read_csv(path, encoding='utf-8', low_memory=False)

# process dataset
dataset = data[data['message'].notnull()].copy()
dataset = dataset[dataset['date'] >= '2023-03-01'].copy()

# keep text data > 10 words
dataset['long_message'] = dataset['message'].apply(
    lambda x: x if len(x.split()) > 10 else None
)

# filter dataset
dataset = dataset[dataset['long_message'].notnull()].copy()

# collect a random sample
sample = dataset.sample(n=250, random_state=1)

# collect text data
text_data = sample['long_message'].tolist()

# test -> load text
embedding_generator.load_text(text_data)

n_tokens = embedding_generator.count_tokens()
estimated_cost = embedding_generator.calculate_embeddings_estimated_cost()

# show values
print (f'Number of tokens: {n_tokens}')
print (f'Estimated cost: {estimated_cost}')
print ('')