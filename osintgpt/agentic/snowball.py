# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: snowball.py
# Description: Retrieve, make the best hit the next query, repeat. Narrative
#   and network expansion — what else is adjacent to this claim.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import List, Optional, Sequence

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider

# import osintgpt projects
from osintgpt.projects import Project

# import osintgpt search
from osintgpt.search import search_project

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, SearchResult

# Each hop costs an embedding call and a search. Enough to walk somewhere the
# first query would not have reached, few enough that a walk cannot run away.
DEFAULT_DEPTH = 5

# Below this the next hop is no longer about the same thing, and continuing
# would follow a thread that has already broken.
DEFAULT_THRESHOLD = 0.5


# Hop class
@dataclass(frozen=True)
class Hop:
    '''
    One step of the walk: what was asked, and what came back.
    '''
    depth: int
    query: str
    result: SearchResult
    # Similarity to the *initial* question rather than to the previous hop.
    # A walk can stay locally coherent while ending somewhere unrelated, and
    # this is the number that makes the drift visible.
    drift: Optional[float] = None

    @property
    def ref(self) -> str:
        return self.result.ref

    @property
    def text(self) -> str:
        return self.result.text


# Snowball class
@dataclass(frozen=True)
class Snowball:
    '''
    A walk outward from one question, and where it went.
    '''
    query: str
    hops: List[Hop] = field(default_factory=list)
    # Why the walk stopped, in words an operator can read.
    stopped: str = ''

    @property
    def refs(self) -> List[str]:
        seen = []
        for hop in self.hops:
            if hop.ref not in seen:
                seen.append(hop.ref)

        return seen

    @property
    def results(self) -> List[SearchResult]:
        return [hop.result for hop in self.hops]

    def __len__(self) -> int:
        return len(self.hops)


# walk outward from a question
def snowball(
    project: Project,
    query: str,
    embedder: EmbeddingProvider,
    depth: int = DEFAULT_DEPTH,
    threshold: float = DEFAULT_THRESHOLD,
    score_against_initial: bool = True,
    refs: Optional[Sequence[str]] = None,
    store: Optional[BaseVectorEngine] = None
) -> Snowball:
    '''
    Retrieve, take the best passage, search for *it*, and repeat.

    A single query finds what resembles the question. Making each hit the next
    query follows a thread instead: the claim adjacent to this claim, and the
    one adjacent to that. It answers a different question from ordinary
    search — "what else is near this" rather than "what matches this" — which
    is why it is a tool of its own rather than a parameter.

    Every hop is scored against the *initial* question as well as followed, so
    a walk that stays locally coherent while ending somewhere unrelated says
    so in `drift` rather than looking like a result.

    Args:
        project (Project): The project to walk.
        query (str): Where to start.
        embedder (EmbeddingProvider): Must be the model the project was \
            indexed with.
        depth (int): Most hops to take.
        threshold (float): Stop when a hop scores below this.
        score_against_initial (bool): Also score each hop against the opening \
            question, which is what makes drift visible.
        refs (Sequence[str], optional): Restrict to these documents.
        store (BaseVectorEngine, optional): Defaults to the project's own.

    Returns:
        Snowball: The hops taken, and why it stopped.
    '''
    from osintgpt.vector_store import store_for

    owned = store is None
    engine = store or store_for(project)

    try:
        return _walk(
            project, query, embedder, engine, depth, threshold,
            score_against_initial, refs
        )
    finally:
        if owned and hasattr(engine, 'close'):
            engine.close()


def _walk(project, query, embedder, engine, depth, threshold,
          score_against_initial, refs):
    initial = embedder.embed([query])[0] if score_against_initial else None

    hops: List[Hop] = []
    seen = set()
    current = query
    stopped = f'reached depth {depth}'

    for step in range(1, depth + 1):
        # Ask for several, because the best hit is usually the passage the
        # query came from. Walking to it would be a cycle, not a step.
        found = search_project(
            project, current, embedder, top_k=5, refs=refs, store=engine
        )

        nxt = next(
            (r for r in found if (r.chunk.ref, r.chunk.sequence) not in seen),
            None
        )
        if nxt is None:
            stopped = 'nothing new to walk to'
            break

        if nxt.score < threshold:
            stopped = (
                f'similarity {nxt.score:.2f} fell below {threshold:.2f}'
            )
            break

        seen.add((nxt.chunk.ref, nxt.chunk.sequence))
        hops.append(Hop(
            depth=step,
            query=current,
            result=nxt,
            drift=_drift(initial, nxt, embedder) if initial else None
        ))
        # The passage becomes the question. This is the whole idea.
        current = nxt.text

    return Snowball(query=query, hops=hops, stopped=stopped)


def _drift(initial: Sequence[float], result: SearchResult,
           embedder: EmbeddingProvider) -> float:
    '''
    Cosine of this passage against the opening question.
    '''
    import math

    vector = embedder.embed([result.text])[0]
    dot = sum(a * b for a, b in zip(initial, vector))
    norms = (
        math.sqrt(sum(a * a for a in initial))
        * math.sqrt(sum(b * b for b in vector))
    )

    return dot / norms if norms else 0.0
