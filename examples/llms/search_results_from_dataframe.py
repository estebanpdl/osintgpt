# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.llms import OpenAIGPT

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> search results from dataframe
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

query = 'Explain in one paragraph relationship between Sheldon and Penny'
response = gpt.search_results_from_dataframe(
    df,
    query=query,
    text_target_column='text_data',
    top_k=2,
    extract_sentence_details=True
)

# get results
results = response['results']

print (f'Query: {query}')
for embeddings, string, score in results:
    # add string to content and give it a new line
    print (f'> {string} -> Score: {score}')


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
