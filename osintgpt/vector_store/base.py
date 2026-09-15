# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: base.py
# Description: What every vector store must do. The seam is wide enough that a
#   caller never learns which backend it is talking to.
# =================================================================================

# import submodules
from abc import ABC, abstractmethod
from uuid import uuid4

# type hints
from typing import Iterable, List, Optional, Sequence

from .records import SearchResult, StoredChunk


# BaseVectorEngine class
class BaseVectorEngine(ABC):
    '''
    A store of embedded chunks.

    Documents, not chunks, are the unit of change: a re-index replaces
    everything a document produced, because chunk boundaries move when its
    text does and leaving the old ones behind is how a store fills with
    passages no document still contains.
    '''

    def indexing_destination(self, root) -> dict:
        '''Override with a stable destination identity for a durable store.

        Unknown/in-memory destinations are conservatively instance-specific.
        Only a hash of the returned value is persisted in index state.
        '''
        if not hasattr(self, '_index_instance'):
            self._index_instance = uuid4().hex
        return {'instance': self._index_instance}

    def index_inventory(self) -> dict:
        '''Counts per file/model, including only contiguous chunk sequences.

        No embedding calls. SQL backends override this to avoid reading text.
        '''
        inventory = {}
        for ref in self.refs():
            chunks = self.chunks_for(ref)
            if [c.sequence for c in chunks] != list(range(len(chunks))):
                inventory[ref] = None
                continue
            models = {}
            for chunk in chunks:
                models[chunk.embedding_model] = models.get(chunk.embedding_model, 0) + 1
            inventory[ref] = models
        return inventory

    def chunks_for(self, ref: str) -> List[StoredChunk]:
        '''Read provenance without vectors, for index reconciliation.'''
        raise NotImplementedError('This store cannot verify existing index contents.')

    # replace a document's chunks
    @abstractmethod
    def upsert(
        self,
        ref: str,
        chunks: Sequence[StoredChunk],
        vectors: Sequence[Sequence[float]]
    ) -> int:
        '''
        Store the chunks of one document, replacing any it had before.

        Args:
            ref (str): The document.
            chunks (Sequence[StoredChunk]): Its chunks, in reading order.
            vectors (Sequence[Sequence[float]]): One vector per chunk, in the \
                same order.

        Raises:
            ValueError: If the counts differ. Vectors are matched to chunks by \
                position, so a mismatch would attach text to the wrong vector.

        Returns:
            int: How many chunks are now stored for the document.
        '''

    # find the closest chunks
    @abstractmethod
    def search(
        self,
        vector: Sequence[float],
        embedding_model: str,
        top_k: int = 10,
        refs: Optional[Iterable[str]] = None
    ) -> List[SearchResult]:
        '''
        Rank stored chunks against a query vector.

        Args:
            vector (Sequence[float]): The query embedding.
            embedding_model (str): Only chunks embedded by this model are \
                considered. Not optional: comparing vectors across models \
                returns confident nonsense rather than an error.
            top_k (int): How many results to return.
            refs (Iterable[str], optional): Restrict to these documents.

        Returns:
            List[SearchResult]: Best first, shorter than `top_k` when the \
                store holds fewer matching chunks.
        '''

    # find chunks containing a term
    @abstractmethod
    def match_text(
        self,
        term: str,
        embedding_model: Optional[str] = None,
        limit: int = 100,
        refs: Optional[Iterable[str]] = None
    ) -> List[StoredChunk]:
        '''
        Chunks whose text or author contains `term`, matched
        case-insensitively.

        Substring rather than token matching, because the tokens this exists
        to catch — handles, hashes, URLs, account ids — are the ones a
        tokenizer splits and an embedding blurs. `@acct_1` must be findable
        inside `contacted @acct_1 twice`.

        The author is matched too, because a question about an account has to
        reach the records it wrote and not only the ones that mention it.
        Those come last: an author match is every chunk that account wrote,
        and ahead of the text matches it would fill the limit on its own.

        Args:
            term (str): Literal text to find. Not a pattern; a caller wanting                 regex filters the results.
            embedding_model (str, optional): Restrict to one model's chunks.                 Text does not depend on the model, but a store holding two                 models' vectors holds every chunk twice.
            limit (int): Most chunks to return.
            refs (Iterable[str], optional): Restrict to these documents.

        Returns:
            List[StoredChunk]: Matching chunks, in document and reading order.
        '''

    # forget documents
    @abstractmethod
    def delete(self, refs: Iterable[str]) -> int:
        '''
        Remove everything stored for the given documents.

        Args:
            refs (Iterable[str]): Documents to forget.

        Returns:
            int: How many chunks were removed.
        '''

    # how much is stored
    @abstractmethod
    def count(self, embedding_model: Optional[str] = None) -> int:
        '''
        Args:
            embedding_model (str, optional): Count only this model's chunks.

        Returns:
            int: Stored chunks.
        '''

    # which documents are stored
    @abstractmethod
    def refs(self, embedding_model: Optional[str] = None) -> List[str]:
        '''
        Args:
            embedding_model (str, optional): Only this model's documents.

        Returns:
            List[str]: Document refs, sorted.
        '''

    # which models have chunks here
    @abstractmethod
    def models(self) -> List[str]:
        '''
        Every embedding model with chunks in the store.

        More than one means a model was switched and the old vectors are still
        here: invisible to search, which filters by model, and still occupying
        the store. Naming them is what makes the leftovers reclaimable.

        Returns:
            List[str]: Model names, sorted.
        '''

    # drop everything from other models
    @abstractmethod
    def purge_other_models(self, keep: str) -> int:
        '''
        Remove chunks embedded by any model but `keep`.

        Args:
            keep (str): The model whose chunks stay.

        Returns:
            int: How many chunks were removed.
        '''
