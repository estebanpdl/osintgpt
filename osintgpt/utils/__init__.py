# A directory containing utility functions

# import modules
import uuid

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
