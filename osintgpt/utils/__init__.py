# A directory containing utility functions

# import modules
import uuid
import tiktoken

# type hints
from typing import List

# create unique id using uuid4
def create_unique_id(ids: List = []) -> str:
    '''
    create unique id using uuid4

    Returns:
        unique_id (str): unique id
    '''
    while True:
        # create unique id
        unique_id = str(uuid.uuid4()).replace('-', '')

        # check if unique id already exists
        if unique_id not in ids:
            break

    # return unique id
    return unique_id

# count tokens < GPT model >
def count_tokens(prompt: str, model: str) -> int:
    '''
    Count tokens
    It counts the number of tokens in the data.

    Args:
        prompt (str): The input prompt for the GPT model.
        model (str): The GPT model

    Returns:
        int: Number of tokens.
    '''
    encoding = tiktoken.encoding_for_model(model)

    # count tokens
    tokens = encoding.encode(prompt)
    num_tokens = len(tokens)

    return num_tokens
