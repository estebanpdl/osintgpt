# -*- coding: utf-8 -*-

# import modules
import time
import pandas as pd

# import submodules
from dotenv import load_dotenv

# import osintgpt modules
from osintgpt.semantic_operations import SemanticOperations
from osintgpt.utils import count_tokens

# Init
text = f'''
Init program at {time.ctime()}

Example -> SemanticOperations -> Summarize content
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
SemanticOperations connection
OpenAIGPT connections
'''
operations = SemanticOperations(env_file_path)


# load text data
path = '../data/big_bang_theory_imdb.csv'
df = pd.read_csv(
    path,
    encoding='utf-8'
)

# Grab content that needs to be summarized
content = df['Plot'].tolist()[:int(df['Plot'].shape[0] / 5)]
content = ' '.join(content)

# Count tokens
model = 'gpt-3.5-turbo'
content_tokens = count_tokens(content, model)
print (f'Number of tokens in content: {content_tokens}')

# Summarize content
user_prompt = 'Genera un resumen del siguiente texto'
response = operations.summarize_content(
    user_prompt=user_prompt,
    context=content,
    max_tokens=300,
    temperature=0
)

# display results
print ('')
print ('')
print ('Summarize content:')
print (response)


# End
text = f'''

End program at {time.ctime()}
'''
print (text)
