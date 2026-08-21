# -*- coding: utf-8 -*-

# ===============================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: search.py
# Description: Similarity search over an in-memory DataFrame or a vector engine,
#   plus the helpers that load embeddings into memory.
# ===============================================================================

# import modules
import pandas as pd

# import submodules
from scipy import spatial
from ast import literal_eval

# type hints
from typing import Optional, List

# import osintgpt vector stores
from osintgpt.vector_store import BaseVectorEngine, Qdrant

# import osintgpt openai embeddings
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# SearchMixin class
class SearchMixin(object):
    '''
    Retrieval over embeddings the caller already holds.
    '''
    # load embeddings
    def load_embeddings_from_csv(self, file_path: str,
        columns: List, **kwargs):
        '''
        Load embeddings from csv file.

        Args:
            file_path (str): CSV file path.
            columns (List): List of columns specifying the embeddings.
            **kwargs: Keyword arguments for pandas read_csv method.
        
        Returns:
            pd.DataFrame: Pandas dataframe.
        '''
        data = pd.read_csv(file_path, **kwargs)
        for col in columns:
            data[col] = data[col].apply(literal_eval)
        
        self._embeddings = {
            col: data[col].tolist() for col in columns
        }

        return data

    # load embeddings from dataframe
    def load_embeddings_from_dataframe(self, dataframe: pd.DataFrame,
        columns: List):
        '''
        Load embeddings from dataframe.

        Args:
            dataframe (pd.DataFrame): Pandas dataframe.
            columns (List): List of columns specifying the embeddings.
        
        Returns:
            pd.DataFrame: Pandas dataframe.
        '''
        for col in columns:
            dataframe[col] = dataframe[col].apply(literal_eval)
        
        self._embeddings = {
            col: data[col].tolist() for col in columns
        }

        return dataframe

    # get embeddings
    def get_embeddings(self, column: str):
        '''
        Get embeddings from `self._embeddings` property.

        Args:
            column (str): Column name containing the embeddings.

        Returns:
            list: List of embeddings.
        '''
        return self._embeddings[column]

    # get embeddings dataframe
    def get_embeddings_dataframe(self):
        '''
        Get embeddings dataframe.

        Returns:
            pd.DataFrame: Pandas dataframe.
        '''
        if not hasattr(self, '_embeddings'):
            raise AttributeError('No embeddings loaded. Please load embeddings.')
        
        return pd.DataFrame(self._embeddings)

    # load search top k results from vector
    def search_results_from_vector(self, vector_engine: BaseVectorEngine,
        query: Optional[str] = None, embeddings: Optional[List] = None,
        top_k: int = 10, **kwargs):
        '''
        Search top k results from vector database.

        Args:
            vector_engine (BaseVectorEngine): Vector engine.
            query (Optional[str]): Query for the search process.
            embeddings (Optional[List]): List of embeddings.
            top_k (int): Top k results to be retrieved.
            **kwargs: Keyword arguments for the vector engine search query method.

        Returns:
            search_results (Dict): Dictionary containing the search results, \
                with the following keys: 'query', 'query_embedding', 'results'.
        '''
        # check if query or embeddings are provided
        if query is None and embeddings is None:
            raise ValueError('Either query or embeddings must be provided.')

        if not isinstance(vector_engine, BaseVectorEngine):
            supported_vector_engines = [
                Qdrant
            ]
            supported_vector_engine_names = ', '.join(
                [engine.__name__ for engine in supported_vector_engines]
            )

            # build message
            msg_a = 'Invalid vector engine provided'
            msg_b = 'Must be an instance of one of the following classes:'
            message = f'{msg_a}. {msg_b} {supported_vector_engine_names}.'
            raise ValueError(message)
        
        # OpenAIEmbeddingGenerator instance
        if query is not None:
            embedding_generator = OpenAIEmbeddingGenerator(self.settings)
            query_embedding = embedding_generator.generate_embedding(query)
        else:
            query_embedding = embeddings

        # search results
        search_results = vector_engine.search_query(
            query_embedding,
            top_k=top_k,
            **kwargs
        )
        
        return {
            'query': query,
            'query_embedding': query_embedding,
            'results': search_results
        }

    # relatedness function
    def _relatedness_fn(self, x, y):
        '''
        Relatedness function.

        This function is used to calculate the relatedness between two embeddings.
        It uses the cosine distance to calculate the relatedness.

        Args:
            x (List[float]): List of embeddings.
            y (List[float]): List of embeddings.
        
        Returns:
            float: Relatedness. 1.0 is most similar, 0.0 is least similar.
        '''
        return 1 - spatial.distance.cosine(x, y)

    # load search top k results from dataframe
    def search_results_from_dataframe(self, df: pd.DataFrame,
        query: Optional[str] = None, embeddings: Optional[List] = None,
        top_k: int = 10, embeddings_target_column: str = 'embeddings',
        text_target_column: str = 'text', extract_sentence_details: bool = False):
        '''
        Search top k results from dataframe.
        
        Args:
            df (pd.DataFrame): Pandas dataframe containing the embeddings. 
            query (Optional[str]): Query for the search process.
            embeddings (Optional[List]): List of embeddings.
            top_k (int): Top k results to be retrieved.
            embeddings_target_column (str): Embeddings target column.
            text_target_column (str): Text target column.
        
        Returns:
            List[Tuple[str, float]]: List of tuples containing the string and score.
        '''
        # check if query or embeddings are provided
        if query is None and embeddings is None:
            raise ValueError('Either query or embeddings must be provided.')
        
        # OpenAIEmbeddingGenerator instance
        if query is not None:
            embedding_generator = OpenAIEmbeddingGenerator(self.settings)
            if extract_sentence_details:
                '''
                This method will try to extract details from query.
                Only Subject or topics will be embbeded.
                The premise is that, based on this approach, we can improve
                    similarity results.
                '''
                response = self.analyze_sentence_details(query)
                response = literal_eval(response)
                try:
                    query = response['Subject or topics']
                except TypeError:
                    pass
            
            query_embedding = embedding_generator.generate_embedding(query)
        else:
            query_embedding = embeddings
        
        strings_and_relatednesses = [
            (
                row[embeddings_target_column],
                row[text_target_column],
                self._relatedness_fn(query_embedding, row[embeddings_target_column])
            )
            for _, row in df.iterrows()
        ]

        strings_and_relatednesses.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'query': query,
            'query_embedding': query_embedding,
            'results': strings_and_relatednesses[:top_k]
        }
