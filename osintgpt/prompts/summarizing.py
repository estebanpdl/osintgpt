# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: summarizing.py
# Description: Prompts for summarization tasks. The text lives in templates/;
#   these name it, so a caller does not have to know a template name.
# =================================================================================

from .templates import static_prompt


# Basic summarization
def basic_summarization() -> str:
    '''
    Guides the model to summarize provided content.

    Returns:
        str: A descriptive prompt for summarization.
    '''
    return static_prompt('summarize')


# Topic modeling and bigrams report
def topic_modeling_summarization() -> str:
    '''
    Guides the model to report themes and prominent bigrams.

    Returns:
        str: A descriptive prompt for topic modeling.
    '''
    return static_prompt('topic_modeling')
