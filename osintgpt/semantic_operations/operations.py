# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
# 
# File: operations.py
# Description: The operations.py file includes a SemanticOperations class designed
#   to offer advanced methods for semantic analysis and text processing. This class
#   focuses on narrative analysis, primarily focusing on discerning and extracting
#   semantic patterns within large text corpora.
# =================================================================================

# import modules
import os
import pandas as pd

# import submodules
from dotenv import load_dotenv

# type hints
from typing import Union, Optional, List, Dict

# import osintgpt vector stores
from osintgpt.llms import OpenAIGPT

# import osintgpt vector stores
from osintgpt.vector_store import BaseVectorEngine

# SemanticOperations class
class SemanticOperations(object):
    '''
    SemanticOperations class
    '''
    def __init__(self, env_file_path: str, llm: str = 'openai'):
        '''
        Initializes the instance of the class.

        Args:
            env_file_path (str): Path to the file containing environment variables.
            model (str): Name of the model to use.
        
        Raises:
            MissingEnvironmentVariableError: If either 'OPENAI_API_KEY' or \
                'OPENAI_GPT_MODEL' is not found in the environment variables.
        '''
        self.llm = OpenAIGPT(env_file_path)
    
    # Semantic similarity search
    def semantic_similarity_search(self, query: str,
        vector_engine: Optional[BaseVectorEngine] = None,
        df: Optional[pd.DataFrame] = None, payload_ref_text_key: str = 'text',
        payload_ref_embeddings_key: str = 'embeddings', top_k: int = 5,
        depth: int = 50, score_threshold: float = 0.85,
        score_based_on_initial_query: bool = False, **kwargs):
        '''
        Semantic Similarity Search

        This function performs a semantic similarity search to find the most similar
        results based on a given query. It first retrieves the most similar
        results to the provided query. Then, it uses the top result as the
        new query and retrieves the most similar results again. This process is
        repeated until the specified depth is reached or until the similarity score
        drops below the defined score threshold. After the second search, the
        function will take the second-most similar result, given that the first
        most similar result is the same as the new query.

        Args:
            query (str): The initial query for the search process.
            vector_engine (Optional[BaseVectorEngine]): An instance of the vector \
                engine used to retrieve similar results. If None, method assumes \
                that a dataframe is provided.
            df (Optional[pd.DataFrame]): A dataframe to be used for searching. If \
                None, method assumes that a vector engine is provided.
            payload_ref_text_key (str): The key in the payload that contains the \
                text to be used for searching.
            payload_ref_embeddings_key (str): The key in the payload that contains \
                the embeddings to be used for searching.
            top_k (int): Top k results to be retrieved.
            depth (int): Depth. The number of times the search process is repeated \
                recursively.
            score_threshold (float): The minimum similarity score. If the \
                similarity score of results drops below this threshold, the \
                recursive search will stop.
            score_based_on_initial_query (bool): If True, the similarity score \
                will be based on the initial query. If False, the similarity \
                score will be based on the subsequent search results.
            **kwargs: Additional keyword arguments for vector engine search query \
                or dataframe search query.
        
        Returns:
            List[Dict]: A list of dictionaries each containing the result string \
                and its similarity score.
        '''
        # check if vector engine or dataframe is provided
        if vector_engine is None and df is None:
            raise ValueError('Either vector engine or dataframe must be provided.')
        
        # if vector engine is provided
        depth_init = depth
        embeddings = None

        # response
        response = []

        # set to track seen documents
        seen_documents = set()

        # search results from vector engine
        if vector_engine is not None:
            while depth > 0:
                # get either query or embeddings
                query = query if embeddings is None else None
                embeddings = embeddings if query is None else None

                # search results
                search_results = self.llm.search_results_from_vector(
                    vector_engine=vector_engine,
                    query=query,
                    embeddings=embeddings,
                    top_k=top_k,
                    **kwargs
                )

                '''
                extract results
                '''
                results = search_results['results']
                depth_has_decreased = depth % depth_init == 0
                item = 0 if depth_has_decreased else 1

                # get query embedding < intial query >
                if score_based_on_initial_query and item == 0:
                    query_embedding = search_results['query_embedding']

                # get embeddings
                embeddings = results[item].payload[
                    payload_ref_embeddings_key
                ]

                # get document < text >
                document = results[item].payload[payload_ref_text_key]

                # get score and check if it is above threshold
                if score_based_on_initial_query:
                    score = self.llm._relatedness_fn(
                        query_embedding,
                        embeddings
                    )
                else:
                    score = results[item].score

                if score < score_threshold:
                    break
                
                if document not in seen_documents:
                    # add to seen documents
                    seen_documents.add(document)

                    # append result
                    response.append(
                        {
                            'document': document,
                            'score': score
                        }
                    )
                else:
                    for i in range(item + 1, len(results)):
                        document = results[i].payload[payload_ref_text_key]
                        if document not in seen_documents:
                            # add to seen documents
                            seen_documents.add(document)

                            # extract score and embeddings from search results
                            if score_based_on_initial_query:
                                score = self.llm._relatedness_fn(
                                    query_embedding,
                                    results[i].payload[payload_ref_embeddings_key]
                                )
                            else:
                                score = results[i].score
                            
                            # get new embeddings
                            embeddings = results[i].payload[
                                payload_ref_embeddings_key
                            ]
                            break
                    else:
                        break
                    
                    if score < score_threshold: 
                        break
                    
                    # append result
                    response.append(
                        {
                            'document': document,
                            'score': score
                        }
                    )
                
                # reduce depth
                depth -= 1
        
        # if dataframe is provided
        else:
            while depth > 0:
                # get either query or embeddings
                query = query if embeddings is None else None
                embeddings = embeddings if query is None else None

                # search results
                search_results = self.llm.search_results_from_dataframe(
                    df=df,
                    query=query,
                    embeddings=embeddings,
                    top_k=top_k,
                    text_target_column=payload_ref_text_key
                )

                '''
                extract results
                '''
                results = search_results['results']
                depth_has_decreased = depth % depth_init == 0
                item = 0 if depth_has_decreased else 1

                # get query embedding < intial query >
                if score_based_on_initial_query and item == 0:
                    query_embedding = search_results['query_embedding']
                
                # get embeddings
                embeddings = results[item][0]

                # get document < text >
                document = results[item][1]

                # get score and check if it is above threshold
                if score_based_on_initial_query:
                    score = self.llm._relatedness_fn(
                        query_embedding,
                        embeddings
                    )
                else:
                    score = results[item][2]

                if score < score_threshold:
                    break
                
                if document not in seen_documents:
                    # add to seen documents
                    seen_documents.add(document)

                    # append result
                    response.append(
                        {
                            'document': document,
                            'score': score
                        }
                    )
                else:
                    for i in range(item + 1, len(results)):
                        document = results[i][1]
                        if document not in seen_documents:
                            # add to seen documents
                            seen_documents.add(document)

                            # extract score and embeddings from search results
                            if score_based_on_initial_query:
                                score = self.llm._relatedness_fn(
                                    query_embedding,
                                    results[i][0]
                                )
                            else:
                                score = results[i][2]
                            
                            # get new embeddings
                            embeddings = results[i][0]
                            break
                    else:
                        break
                    
                    if score < score_threshold:
                        break
                    
                    # append result
                    response.append(
                        {
                            'document': document,
                            'score': score
                        }
                    )

                # reduce depth
                depth -= 1
        
        return response
    
    # Compare two vectors (e.g., twitter vector, youtube transcripts vector, etc.)
    