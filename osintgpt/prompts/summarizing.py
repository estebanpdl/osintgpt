# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: summarizing.py
# Description: This module contains a collection of prompts specifically crafted 
#              for summarization tasks. These prompts aid in guiding GPT-based 
#              models to generate concise and relevant summaries.
# =================================================================================

# import modules

# Basic summarization
def basic_summarization():
    '''
    Provides a comprehensive prompt for summarization tasks.

    This function returns a pre-defined prompt to guide the Large Language
    Model in summarizing provided content. It instructs the model on the 
    approach to follow, ensuring an accurate and relevant summary based on 
    the user's content.

    Returns:
        str: A descriptive prompt for summarization.
    '''
    return '''
    As a Large Language Model, you are equipped to process and summarize vast
    amounts of information. When the user provides a content, your task is to
    provide an accurate summary, introducing the core ideas, the essence of the
    content, and the main events and keywords included within the content.
    
    Please, respond always after following the below procedure:
    1. Recognize the format or structure of the user's content.
    2. Identify main topics, events, and key phrases in content.
    3. Focus only on the provided content; do not introduce external information.

    Use such topics, events, and phrases to summarize the content. The length in
    terms of paragraphs is at your discretion, but ensure the content is
    comprehensive and inclusive of relevant events. Be sure to include such events.
    Always respond in the language in which the user made the request.
    '''

# Topic modeling and bigrams report
def topic_modeling_summarization():
    '''
    Provides a comprehensive prompt for topic modeling and bigrams report tasks.

    This function returns a pre-defined prompt guiding the Large Language Model 
    in performing topic modeling on the provided content. It instructs the model 
    to identify and report on the main themes and significant bigrams present in the 
    content to give a structured overview of the topics discussed.

    Returns:
        str: A descriptive prompt for topic modeling and bigrams report.
    '''
    return '''
    Analyze the provided content using topic modeling techniques. Your objective is 
    to extract the main themes or topics from the content, giving a clear 
    understanding of the subject matters discussed within. 

    Additionally, identify prominent bigrams (two-word phrases) that can offer 
    insight into recurring patterns or significant points of discussion.

    Structure your response as a basic report, highlighting:
    1. The top themes or topics you've identified.
    2. Noteworthy bigrams and their relevance.
    3. Any observations or narrative patterns that can help understand the content.
    4. Always respond in the language in which the user made the request.
    '''
