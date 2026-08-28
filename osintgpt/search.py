# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: search.py
# Description: Semantic search over a project, and across several. The first
#   place a stored corpus answers a question.
# =================================================================================

# import submodules
from contextlib import contextmanager

# type hints
from typing import Iterable, List, Optional, Sequence

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider

# import osintgpt fusion
from osintgpt.fusion import FusedResult, reciprocal_rank_fusion

# import osintgpt projects
from osintgpt.projects import Project
from osintgpt.projects.cross_project import (
    CrossProjectResults,
    search_projects
)
from osintgpt.projects.settings import ProjectSettings

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, SearchResult, store_for


# search one project
def search_project(
    project: Project,
    query: str,
    embedder: EmbeddingProvider,
    top_k: int = 10,
    refs: Optional[Iterable[str]] = None,
    store: Optional[BaseVectorEngine] = None
) -> List[SearchResult]:
    '''
    Find the passages closest to a question.

    Args:
        project (Project): The project to search.
        query (str): The question, embedded as written. Reformulating it is \
            the caller's decision, not this function's.
        embedder (EmbeddingProvider): Must be the model the project was \
            indexed with; the store filters on it, so a mismatch returns \
            nothing rather than nonsense.
        top_k (int): How many passages to return.
        refs (Iterable[str], optional): Restrict to these documents.
        store (BaseVectorEngine, optional): Defaults to the project's own.

    Returns:
        List[SearchResult]: Best first.
    '''
    vector = embedder.embed([query])[0]

    with _store_for(project, store) as engine:
        return engine.search(
            vector, embedding_model=embedder.model, top_k=top_k, refs=refs
        )


# run both retrieval legs and combine them
def hybrid_search(
    project: Project,
    query: str,
    embedder: EmbeddingProvider,
    generator: Optional[GenerationProvider] = None,
    top_k: int = 10,
    terms: Optional[Sequence[str]] = None,
    refs: Optional[Iterable[str]] = None,
    store: Optional[BaseVectorEngine] = None
) -> List[FusedResult]:
    '''
    Search semantically and lexically, then fuse by rank.

    The legs are complementary rather than redundant: semantic finds a passage
    that means the same thing in different words, lexical finds the exact
    identifier that means nothing to an embedding. A passage both legs return
    outranks one either found deeply.

    The lexical leg needs terms. Given a generator it derives them; given
    neither terms nor a generator it sits out, and this degrades to semantic
    search rather than failing.

    Args:
        project (Project): The project to search.
        query (str): The question, as asked.
        embedder (EmbeddingProvider): Must be the model the project was             indexed with.
        generator (GenerationProvider, optional): Derives exact-match terms.
        top_k (int): How many fused results to return.
        terms (Sequence[str], optional): Exact terms, when the caller already             has them. Skips derivation, and an empty sequence sits the             lexical leg out deliberately.
        refs (Iterable[str], optional): Restrict to these documents.
        store (BaseVectorEngine, optional): Defaults to the project's own.

    Returns:
        List[FusedResult]: Best first, each carrying the rank every leg gave it.
    '''
    from osintgpt.lexical import derive_search_terms, lexical_search

    legs = {
        'semantic': search_project(
            project, query, embedder, top_k=top_k, refs=refs, store=store
        )
    }

    if terms is None and generator is not None:
        terms = derive_search_terms(generator, query)

    if terms:
        legs['lexical'] = lexical_search(
            project, terms, embedding_model=embedder.model, refs=refs,
            store=store
        )

    return reciprocal_rank_fusion(legs, limit=top_k)


# search several projects at once
def search_across_projects(
    projects: Sequence[Project],
    query: str,
    embedder: EmbeddingProvider,
    top_k: int = 10,
    defaults: Optional[ProjectSettings] = None
) -> CrossProjectResults:
    '''
    Search projects that share an embedding model, and say which were left out.

    Projects embedded with a different model are excluded rather than merged:
    their vectors are not comparable, and the result names them so a partial
    answer cannot read as a complete one.

    Args:
        projects (Sequence[Project]): Projects to search.
        query (str): The question.
        embedder (EmbeddingProvider): The model to search with. Projects \
            using another are skipped.
        top_k (int): How many passages to return in total.
        defaults (ProjectSettings, optional): User defaults, for projects \
            that left their embedding model unset.

    Returns:
        CrossProjectResults: Merged hits, plus what was skipped and why.
    '''
    vector = embedder.embed([query])[0]

    def run(project: Project):
        with _store_for(project, None) as engine:
            results = engine.search(
                vector, embedding_model=embedder.model, top_k=top_k
            )

        return [(result.score, result) for result in results]

    return search_projects(
        projects,
        run,
        embedding_model=embedder.model,
        defaults=defaults,
        limit=top_k
    )


@contextmanager
def _store_for(project: Project, store: Optional[BaseVectorEngine]):
    '''
    A store this opened is one it closes; one passed in belongs to the caller.
    '''
    if store is not None:
        yield store

        return

    engine = store_for(project)
    try:
        yield engine
    finally:
        engine.close()
