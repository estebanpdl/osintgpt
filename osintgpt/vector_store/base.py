# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: base.py
# Description: `BaseVectorEngine`` is an abstract base class that serves as a common
#   interface for different vector search engine implementations.
# =================================================================================

# import submodules
from abc import ABC, abstractmethod

# type hints
from typing import List

# BaseVectorEngine class
class BaseVectorEngine(ABC):
    '''
    BaseVectorEngine class

    The BaseVectorEngine class contains an abstract method called search_query,
    which must be implemented by any subclass inheriting from it. The method is
    responsible for searching results from the respective vector search engine
    and returning the top k similar results.
    '''
    @abstractmethod
    def search_query(self, embedded_query: List[float], top_k: int, **kwargs):
        '''
        Search query

        This method is responsible for searching results from the respective
        vector search engine and returning the top k similar results.

        Args:
            embedded_query: embedded query
            top_k: number of results to return
            **kwargs: keyword arguments for vector search engines
        '''
        pass
