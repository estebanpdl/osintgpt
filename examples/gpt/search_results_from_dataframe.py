# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.gpt import OpenAIGPT

# Init
text = f'''
Init program at {time.ctime()}

Testing -> OpenAIGPT -> search results from dataframe
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

# search results from dataframe
strings, relatednesses = gpt.search_results_from_dataframe(
    'Sheldon explores a new theory on quantum physics', df, text_target_column='text_data', top_k=10
)

content = ''
for string, relatedness in zip(strings, relatednesses):
    # add string to content and give it a new line
    content += string + '\n'

# print content
print (content)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
