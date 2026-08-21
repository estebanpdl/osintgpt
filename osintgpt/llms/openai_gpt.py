# -*- coding: utf-8 -*-

# ===============================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: openai_gpt.py
# Description: OpenAIGPT provides an interface to OpenAI's chat models, assembled
#   from the search, conversation-log and completion behaviours.
# ===============================================================================

# import submodules
from openai import OpenAI

# type hints
from typing import Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

# import osintgpt pricing
from osintgpt.pricing import estimate_cost

# import utils
from osintgpt.utils import count_tokens as count_model_tokens

# import behaviours
from .completions import CompletionsMixin
from .conversation import ConversationLogMixin
from .search import SearchMixin

# OpenAIGPT class
class OpenAIGPT(SearchMixin, ConversationLogMixin, CompletionsMixin):
    '''
    OpenAIGPT class
    '''
    def __init__(self, config: Union[Settings, str]):
        '''
        Initializes the instance of the class.

        Args:
            config (Union[Settings, str]): Settings, or a path to a .env file \
                (deprecated).

        Raises:
            MissingEnvironmentVariableError: If either 'openai_api_key' or \
                'openai_gpt_model' has no value.
        '''
        # settings
        self.settings = resolve_settings(config).require(
            'openai_api_key', 'openai_gpt_model'
        )

        self.OPENAI_API_KEY = self.settings.openai_api_key
        self.OPENAI_GPT_MODEL = self.settings.openai_gpt_model

        # client
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)

        # set SQL unique id
        self.SQL_UNIQUE_ID = self._generate_unique_id()
        self.SQL_UNIQUE_ID_INSERTED = False

    # get openai api key
    def get_openai_api_key(self):
        '''
        Get OpenAI API key.

        Returns:
            str: OpenAI API key.
        '''
        return self.OPENAI_API_KEY

    # count tokens < GPT model >
    def count_tokens(self, prompt: str):
        '''
        Count tokens
        It counts the number of tokens in the data.

        Args:
            prompt (str): The input prompt for the GPT model.

        Returns:
            int: Number of tokens.
        '''
        return count_model_tokens(prompt, self.OPENAI_GPT_MODEL)

    # calculate completion response usage cost
    def estimated_prompt_cost(self, prompt: str):
        '''
        It calculates the estimated cost of sending a prompt to the configured
        model, based on the number of tokens.

        Args:
            prompt (str): The input prompt for the GPT model.

        Returns:
            Optional[float]: Estimated USD, or None when the model carries no \
                price in the table. None means unknown, not free.
        '''
        return estimate_cost(
            self.OPENAI_GPT_MODEL,
            self.count_tokens(prompt)
        )
