# base class
from .base import BaseVectorEngine

# what a store holds
from .records import SearchResult, StoredChunk

# import class methods
from .sqlite_store import BRUTE_FORCE_CEILING, SQLiteVectorStore
from .qdrant_store import DEFAULT_COLLECTION, QdrantVectorStore
from .qdrant import Qdrant

# choosing one from a project's settings
from .factory import BACKENDS, store_for
