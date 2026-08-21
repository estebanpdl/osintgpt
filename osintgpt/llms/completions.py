# -*- coding: utf-8 -*-

# ===============================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: completions.py
# Description: Chat completions: single calls, an interactive loop, and the
#   query reformulation used before embedding a search.
# ===============================================================================

# type hints
from typing import Union, Optional, List, Dict

# CompletionsMixin class
class CompletionsMixin(object):
    '''
    Requests to the chat model, each logged to the conversation store.
    '''
    # get GPT model completion when adding a system role
    def get_model_completion_using_system_role(self, messages: List[Dict],
        verbose: bool = True, **kwargs):
        '''
        Get GPT model completion.

        Args:
            messages (List[Dict]): A list of message objects. Each object \
                should be a dictionary containing 'role' and 'content'.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
            **kwargs: Keyword arguments for OpenAI's create completion.
        
        Returns:
            str: GPT completion response.
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # get completion response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

        # insert system prompt into sql database
        system_prompt = messages[0]['content']
        self.insert_system_prompt_into_sql_database(system_prompt)

        # insert user prompt into sql database
        user_prompt = messages[1]['content']
        self.insert_user_prompt_into_sql_database(response, user_prompt)

        # insert response into sql database
        self.insert_completion_response_into_sql_database(response)

        # display main values
        if verbose:
            print('Response id: ', self._get_completion_response_id(response))
            for key, value in self._get_completion_response_usage(response).items():
                print(f'{key}: {value}')

        return response.choices[0].message.content

    # get GPT model completion
    def get_model_completion(self, prompt: str,
        messages: Optional[Union[List, Dict]] = None, temperature: float = 0,
        verbose: bool = True):
        '''
        Get GPT model completion.

        Args:
            prompt (str): The input prompt for the GPT model.
            messages (Union[List, Dict], optional): A list or dictionary of \
                messages. If it's a list, it should be a list of message objects. \
                If it's a dictionary, it should contain 'ref_id' and 'messages'.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
            
        Returns:
            str: GPT completion response.
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # build messages
        if messages is None:
            messages = [
                {'role': 'user', 'content': prompt}
            ]
        else:
            if type(messages) == dict:
                '''
                Since messages are provided, we assume that the SQL_UNIQUE_ID
                has already been inserted into the database.

                Pass ref_id to SQL_UNIQUE_ID.
                Set SQL_UNIQUE_ID_INSERTED to True.
                '''
                self.SQL_UNIQUE_ID = messages['ref_id']
                self.SQL_UNIQUE_ID_INSERTED = True

                # build messages
                messages = messages['messages'] + [
                    {'role': 'user', 'content': prompt}
                ]
            else:
                pass

        # get completion response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        # insert user prompt into sql database
        self.insert_user_prompt_into_sql_database(response, prompt)

        # insert response into sql database
        self.insert_completion_response_into_sql_database(response)

        # display main values
        if verbose:
            print('Response id: ', self._get_completion_response_id(response))
            for key, value in self._get_completion_response_usage(response).items():
                print(f'{key}: {value}')

        return response.choices[0].message.content

    # interactive completion: role system
    def interactive_completion(self, prompt: Optional[str] = None,
        messages: Optional[Dict] = None, temperature: float = 0,
        verbose: bool = False):
        '''
        Interactive completion. Interact with the GPT model using the command line.

        Args:
            prompt (str, optional): The input prompt for the GPT model.
            messages (List[Dict], optional): A list of message objects. Each object \
                should be a dictionary containing 'role' and 'content'.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
            verbose (bool, optional): If set to True, additional details about the \
                request and response will be printed.
        
        Returns:
            None
        '''
        # Check that at least one of prompt or messages is provided
        if prompt is None and messages is None:
            raise ValueError('Either prompt or messages must be provided.')
        
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # build messages
        if messages is None:
            messages = [
                {'role': 'system', 'content': prompt}
            ]

            # insert system prompt into sql database
            self.insert_system_prompt_into_sql_database(prompt)
        else:

            '''
            Since messages are provided, we assume that the SQL_UNIQUE_ID
            has already been inserted into the database.

            Pass ref_id to SQL_UNIQUE_ID.
            Set SQL_UNIQUE_ID_INSERTED to True.
            '''
            self.SQL_UNIQUE_ID = messages['ref_id']
            self.SQL_UNIQUE_ID_INSERTED = True

            # build messages
            messages = messages['messages']

        # interactive chat mode
        print ('Interactive chat mode with GPT. Type "exit" to quit.')
        print ('')
        while True:
            user_input = input('You: ')
            if user_input == 'exit':
                print ('')
                print ('Exiting interactive chat mode...')
                break
            
            # accumulate messages
            msg = {'role': 'user', 'content': user_input}
            messages.append(msg)
            
            # get completion
            gpt_response = self.get_model_completion(
                user_input,
                messages=messages,
                temperature=temperature,
                verbose=verbose
            )

            # accumulate messages
            msg = {'role': 'assistant', 'content': gpt_response}
            messages.append(msg)

            print (f'{model}: ', gpt_response)

    # Analyze sentence details
    def analyze_sentence_details(self, sentence: str, temperature: float = 0):
        '''
        Analyzes the provided sentence to identify its primary language,
        the central request or question, and the main theme or subject matter
        associated with that request.

        Args:
            sentence (str): The input sentence to be analyzed.
            temperature (float, optional): Controls the randomness of the model's \
                output. The higher the value, the more random the output will be. \
                If not provided, the output will be deterministic.
        
        Returns:
            dict: A dictionary containing keys "Language", "Input request", and
            "Subject or topics" with their respective identified values.
        '''
        # set api key
        if not self.OPENAI_API_KEY:
            raise ValueError('No OpenAI API key provided. Please provide one.')

        # get model
        if not self.OPENAI_GPT_MODEL:
            raise ValueError('No OpenAI GPT model provided. Please provide one.')

        model = self.OPENAI_GPT_MODEL

        # generate system message role
        system_role = '''
        As a Large Language Model, you specialize in dissecting sentences to unearth
        the core components within them. When presented with a sentence:
        1. Determine the primary language in which the sentence is written.
        2. Extract the central request, input, or question that requires assistance.
        3. Identify the main topic/s or subject/s associated with the central request.

        Compile your findings into a JSON response, highlighting these three
        essential aspects.

        Example input: "¿Cuáles son los beneficios de la energía solar?"
        Expected output:
        ```
        {
            "Language": "Spanish",
            "Input request": "¿Cuáles son los beneficios",
            "Subject or topics": "energía solar"
        }
        ```
        '''
        # build messages
        messages = [
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': sentence}
        ]

        # get completion response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content
