# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: qdrant_store.py
# Description: Qdrant behind the same interface as the default store. The
#   scale-up option, reached by changing configuration rather than callers.
# =================================================================================

# import submodules <Qdrant>
from qdrant_client.http import models as rest

# import modules
import uuid

# type hints
from typing import Iterable, List, Optional, Sequence, Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

from .base import BaseVectorEngine
from .connection import connect
from .records import SearchResult, StoredChunk

# One collection per project, named by the project rather than shared, so
# isolation stays structural here as it is with a file per project.
DEFAULT_COLLECTION = 'osintgpt'

# Namespace for deriving a point id from (collection, ref, sequence). Qdrant
# requires an integer or a UUID, and deriving one means re-indexing a document
# overwrites its own points rather than accumulating near-duplicates.
POINT_NAMESPACE = uuid.UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')

# Payload keys. Named rather than inlined because they are also filter keys,
# and a typo in a filter returns nothing rather than raising.
REF = 'ref'
SEQUENCE = 'sequence'
TEXT = 'text'
PATH = 'path'
TIMESTAMP = 'timestamp'
AUTHOR = 'author'
METADATA = 'metadata'
EMBEDDING_MODEL = 'embedding_model'

# Scrolled in pages rather than read whole: a collection large enough to want
# Qdrant is large enough that one request for everything is a bad idea.
SCROLL_PAGE = 1_000


# QdrantVectorStore class
class QdrantVectorStore(BaseVectorEngine):
    '''
    Vectors in a Qdrant collection, behind the interface the default store
    satisfies. Swapping backends is configuration, not a rewrite.
    '''
    def __init__(
        self,
        config: Union[Settings, str],
        collection: str = DEFAULT_COLLECTION,
        client=None
    ) -> None:
        '''
        Args:
            config (Union[Settings, str]): Settings, or a path to a .env file \
                (deprecated).
            collection (str): Collection to use. One per project keeps \
                isolation structural.
            client: An open client, for tests and for reusing a connection. \
                Built from settings when not given.
        '''
        self.settings = resolve_settings(config)
        self.collection = collection

        if client is None:
            self.client, self.kind = connect(self.settings)
        else:
            self.client, self.kind = client, 'injected'

    def upsert(
        self,
        ref: str,
        chunks: Sequence[StoredChunk],
        vectors: Sequence[Sequence[float]]
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(
                f'{ref}: {len(chunks)} chunks and {len(vectors)} vectors. '
                'They are matched by position, so a mismatch would attach '
                'text to the wrong vector.'
            )

        # Delete before inserting rather than relying on derived ids: a
        # document that shrank from five chunks to three would otherwise keep
        # the two it no longer has.
        if self._collection_exists():
            self.client.delete(
                collection_name=self.collection,
                points_selector=rest.FilterSelector(filter=self._by_ref([ref]))
            )

        if not chunks:
            return 0

        self._ensure_collection(len(vectors[0]))
        self.client.upsert(
            collection_name=self.collection,
            points=[
                rest.PointStruct(
                    id=self._point_id(chunk.ref, chunk.sequence),
                    vector=list(vector),
                    payload=_to_payload(chunk)
                )
                for chunk, vector in zip(chunks, vectors)
            ]
        )

        return len(chunks)

    def search(
        self,
        vector: Sequence[float],
        embedding_model: str,
        top_k: int = 10,
        refs: Optional[Iterable[str]] = None
    ) -> List[SearchResult]:
        conditions = [_match(EMBEDDING_MODEL, embedding_model)]

        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            conditions.append(
                rest.FieldCondition(key=REF, match=rest.MatchAny(any=wanted))
            )

        if not self._collection_exists():
            return []

        response = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=rest.Filter(must=conditions),
            limit=top_k,
            with_payload=True
        )

        return [
            SearchResult(
                chunk=_from_payload(point.payload), score=float(point.score)
            )
            for point in response.points
        ]

    def match_text(
        self,
        term: str,
        embedding_model: Optional[str] = None,
        limit: int = 100,
        refs: Optional[Iterable[str]] = None
    ) -> List[StoredChunk]:
        if not term or not self._collection_exists():
            return []

        conditions = []
        if embedding_model is not None:
            conditions.append(_match(EMBEDDING_MODEL, embedding_model))
        if refs is not None:
            wanted = list(refs)
            if not wanted:
                return []
            conditions.append(
                rest.FieldCondition(key=REF, match=rest.MatchAny(any=wanted))
            )

        # Qdrant's own MatchText is token-based: it would find `acct` and miss
        # `@acct_1` inside a sentence, which is precisely the token this leg
        # exists to catch. So the filter narrows and the substring test runs
        # here, over payloads read without their vectors.
        needle = term.lower()
        found: List[StoredChunk] = []
        for payload in self._scroll(
            rest.Filter(must=conditions) if conditions else None
        ):
            if needle in str(payload.get(TEXT, '')).lower():
                found.append(_from_payload(payload))
                if len(found) >= limit:
                    break

        return sorted(found, key=lambda chunk: (chunk.ref, chunk.sequence))

    def delete(self, refs: Iterable[str]) -> int:
        wanted = list(refs)
        if not wanted or not self._collection_exists():
            return 0

        # Counted first because Qdrant reports an operation's status rather
        # than how many points it touched, and the interface promises a count.
        removed = self.client.count(
            collection_name=self.collection,
            count_filter=self._by_ref(wanted),
            exact=True
        ).count

        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.FilterSelector(filter=self._by_ref(wanted))
        )

        return removed

    def count(self, embedding_model: Optional[str] = None) -> int:
        if not self._collection_exists():
            return 0

        count_filter = None
        if embedding_model is not None:
            count_filter = rest.Filter(
                must=[_match(EMBEDDING_MODEL, embedding_model)]
            )

        return self.client.count(
            collection_name=self.collection,
            count_filter=count_filter,
            exact=True
        ).count

    def refs(self, embedding_model: Optional[str] = None) -> List[str]:
        return sorted(self._distinct(REF, embedding_model))

    def models(self) -> List[str]:
        return sorted(self._distinct(EMBEDDING_MODEL, None))

    def purge_other_models(self, keep: str) -> int:
        if not self._collection_exists():
            return 0

        other = rest.Filter(must_not=[_match(EMBEDDING_MODEL, keep)])
        removed = self.client.count(
            collection_name=self.collection, count_filter=other, exact=True
        ).count

        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.FilterSelector(filter=other)
        )

        return removed

    # chunks of one document, in reading order
    def chunks_for(self, ref: str) -> List[StoredChunk]:
        '''
        Args:
            ref (str): The document.

        Returns:
            List[StoredChunk]: Its chunks, in the order they were stored.
        '''
        chunks = [
            _from_payload(payload)
            for payload in self._scroll(self._by_ref([ref]))
        ]

        return sorted(chunks, key=lambda chunk: chunk.sequence)

    def _distinct(self, key: str, embedding_model: Optional[str]) -> set:
        '''
        Qdrant has no DISTINCT, so this walks the payloads. A maintenance
        operation rather than a hot path, and paged so a large collection is
        never requested in one response.
        '''
        if not self._collection_exists():
            return set()

        scroll_filter = None
        if embedding_model is not None:
            scroll_filter = rest.Filter(
                must=[_match(EMBEDDING_MODEL, embedding_model)]
            )

        return {
            payload.get(key, '') for payload in self._scroll(scroll_filter)
        }

    def _scroll(self, scroll_filter):
        '''Every matching payload, a page at a time. Vectors are not read.'''
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            for point in points:
                yield point.payload or {}

            if offset is None:
                return

    def _ensure_collection(self, dimensions: int) -> None:
        '''
        Create the collection on first write, when the vector size is finally
        known. Payload indexes come with it: every search filters on the
        model, and most also filter on the document.
        '''
        if self._collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(
                size=dimensions, distance=rest.Distance.COSINE
            )
        )
        for key in (REF, EMBEDDING_MODEL):
            # wait=False because building an index takes longer on a real
            # server than the client's default timeout allows, and blocking
            # the first write of a project on an optimization is wrong: Qdrant
            # answers with a full scan until the index is ready.
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=key,
                field_schema=rest.PayloadSchemaType.KEYWORD,
                wait=False
            )

    def _collection_exists(self) -> bool:
        return bool(
            self.client.collection_exists(collection_name=self.collection)
        )

    def _by_ref(self, refs: Sequence[str]) -> 'rest.Filter':
        return rest.Filter(
            must=[rest.FieldCondition(
                key=REF, match=rest.MatchAny(any=list(refs))
            )]
        )

    def _point_id(self, ref: str, sequence: int) -> str:
        return str(
            uuid.uuid5(POINT_NAMESPACE, f'{self.collection}:{ref}:{sequence}')
        )


def _match(key: str, value: str) -> 'rest.FieldCondition':
    return rest.FieldCondition(key=key, match=rest.MatchValue(value=value))


def _to_payload(chunk: StoredChunk) -> dict:
    return {
        REF: chunk.ref,
        SEQUENCE: chunk.sequence,
        TEXT: chunk.text,
        PATH: chunk.path,
        TIMESTAMP: chunk.timestamp,
        AUTHOR: chunk.author,
        METADATA: dict(chunk.metadata),
        EMBEDDING_MODEL: chunk.embedding_model
    }


def _from_payload(payload: Optional[dict]) -> StoredChunk:
    payload = payload or {}

    return StoredChunk(
        ref=payload.get(REF, ''),
        sequence=int(payload.get(SEQUENCE, 0) or 0),
        text=payload.get(TEXT, ''),
        embedding_model=payload.get(EMBEDDING_MODEL, ''),
        path=payload.get(PATH, ''),
        timestamp=payload.get(TIMESTAMP, ''),
        author=payload.get(AUTHOR, ''),
        metadata=payload.get(METADATA) or {}
    )
