# base class
from .base import BaseVectorEngine

# what a store holds
from .records import SearchResult, StoredChunk

# import class methods
from .sqlite_store import BRUTE_FORCE_CEILING, SQLiteVectorStore

# choosing one from a project's settings
from .factory import BACKENDS, store_for

# PgVectorStore is deliberately not imported here: its drivers are optional,
# and importing this package must not require them.

__all__ = [
    'BACKENDS', 'BRUTE_FORCE_CEILING', 'BaseVectorEngine', 'DEFAULT_COLLECTION',
    'Qdrant', 'QdrantVectorStore', 'SQLiteVectorStore', 'SearchResult',
    'StoredChunk', 'store_for',
]


def __getattr__(name):
    '''Load the optional Qdrant backend only when its public names are used.'''
    if name in ('QdrantVectorStore', 'DEFAULT_COLLECTION'):
        from . import qdrant_store

        return getattr(qdrant_store, name)
    if name == 'Qdrant':
        from .qdrant import Qdrant

        return Qdrant
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
