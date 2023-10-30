# -*- coding: utf-8 -*-

# import modules
import time

# import osintgpt modules
from osintgpt.llms import OpenAIGPT

# Init
text = f'''
Init program at {time.ctime()}

Example -> OpenAIGPT -> analyze sentence details
'''
print (text)

# config -> env file path
env_file_path = '../config/.env'

'''
OpenAIGPT connection
'''
gpt = OpenAIGPT(env_file_path)


# build sentence
sentence = 'Which are the main narratives dicussed around the protests?'

response = gpt.analyze_sentence_details(
    sentence=sentence
)

# display new lines
print ('')
print ('')
print ('Model completion:\n')
print (response)

# End
text = f'''

End program at {time.ctime()}
'''
print (text)
