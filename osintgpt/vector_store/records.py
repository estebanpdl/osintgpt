# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: records.py
# Description: What a store holds and what it hands back. A stored chunk keeps
#   everything a citation needs, so nothing has to be looked up again to say
#   where an answer came from.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Any, Dict, Optional


# StoredChunk class
@dataclass(frozen=True)
class StoredChunk:
    '''
    One embedded chunk, with the provenance a citation needs.
    '''
    # Which document it came from, as the corpus refers to it.
    ref: str
    # Position within that document, so chunks return in reading order and a
    # re-index can replace a document's chunks as a set.
    sequence: int
    text: str
    # The model that produced the vector. Filtered on every search: vectors
    # from different models are not comparable, and comparing them returns
    # confident nonsense rather than an error.
    embedding_model: str
    # The heading path the chunk sits under, empty for unstructured text.
    path: str = ''
    # Named separately from metadata because retrieval filters on them.
    timestamp: str = ''
    author: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        '''
        Returns:
            str: How this chunk names itself in an answer — the document, and \
                the section within it when there is one.
        '''
        return f'{self.ref} › {self.path}' if self.path else self.ref


# SearchResult class
@dataclass(frozen=True)
class SearchResult:
    '''
    A stored chunk and how well it matched.
    '''
    chunk: StoredChunk
    # Cosine similarity: 1.0 is identical, 0.0 unrelated. Comparable within
    # one search and meaningless between searches on different models.
    score: float

    @property
    def ref(self) -> str:
        return self.chunk.ref

    @property
    def text(self) -> str:
        return self.chunk.text
