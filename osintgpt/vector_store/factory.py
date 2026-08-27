# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: factory.py
# Description: Choosing a project's store from its settings. The one place a
#   backend name becomes a backend, so callers never learn which one they got.
# =================================================================================

# type hints
from typing import Optional, Union

# import osintgpt config
from osintgpt.config import Settings, resolve_settings

from .base import BaseVectorEngine
from .qdrant_store import QdrantVectorStore
from .sqlite_store import SQLiteVectorStore

SQLITE = 'sqlite'
QDRANT = 'qdrant'

BACKENDS = (SQLITE, QDRANT)


# open the store a project is configured to use
def store_for(
    project, config: Optional[Union[Settings, str]] = None
) -> BaseVectorEngine:
    '''
    Build the store named by the project's `storage_backend`.

    Args:
        project (Project): The project whose store to open.
        config (Union[Settings, str], optional): Connection settings, needed \
            only by backends that reach a server.

    Raises:
        ValueError: If the project names a backend that does not exist. \
            Failing here is better than silently falling back to the default \
            and indexing a corpus somewhere nobody meant.

    Returns:
        BaseVectorEngine: An open store.
    '''
    backend = (project.settings.storage_backend or SQLITE).strip().lower()

    if backend == SQLITE:
        return SQLiteVectorStore(project.paths.store)

    if backend == QDRANT:
        return QdrantVectorStore(
            resolve_settings(config) if config is not None else Settings(),
            # The project's own slug, so two projects on one server stay
            # separate the way two project files do.
            collection=project.slug
        )

    raise ValueError(
        f'{project.slug}: unknown storage backend {backend!r}; '
        f'known backends: {", ".join(BACKENDS)}'
    )
