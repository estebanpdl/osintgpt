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

# fallback encoding for models tiktoken does not know
DEFAULT_ENCODING = 'o200k_base'

# resolve the encoding for a model
def encoding_for_model(model: str):
    '''
    Get the tiktoken encoding a model uses.

    Models released after the installed tiktoken are unknown to it, which
    raises rather than returning something usable. Falling back keeps counting
    approximate instead of fatal.

    Args:
        model (str): Model name.

    Returns:
        tiktoken.Encoding: The model's encoding, or the default fallback.
    '''
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(DEFAULT_ENCODING)

# count tokens < GPT model >
def count_tokens(prompt: str, model: str) -> int:
    '''
    Count tokens
    It counts the number of tokens in the data.

    Args:
        prompt (str): The input prompt for the GPT model.
        model (str): The model the tokens will be sent to. Encodings differ \
            between models, so counting for the wrong one gives a wrong number.

    Returns:
        int: Number of tokens.
    '''
    encoding = encoding_for_model(model)

    # count tokens
    tokens = encoding.encode(prompt)
    num_tokens = len(tokens)

    return num_tokens
