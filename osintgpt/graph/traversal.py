# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: traversal.py
# Description: Walking the edge set in process. A project graph is small
#   enough that loading it whole beats querying it recursively.
# =================================================================================

# import submodules
from collections import deque
from dataclasses import dataclass, field

# type hints
from typing import Dict, List, Optional, Sequence, Tuple

from .store import Edge, GraphStore, merge_key

# How far a path search will look. Beyond this the connection is too indirect
# to mean anything an analyst would act on, and the search cost grows with the
# branching factor.
MAX_DEPTH = 4


# GraphHit class
@dataclass(frozen=True)
class GraphHit:
    '''
    One edge, carrying the document and sentence that assert it.
    '''
    edge: Edge
    # How many steps from the entity that was asked about.
    depth: int = 1

    @property
    def ref(self) -> str:
        return self.edge.ref

    @property
    def evidence(self) -> str:
        return self.edge.evidence


# GraphPath class
@dataclass(frozen=True)
class GraphPath:
    '''
    A chain of claims connecting two entities.
    '''
    edges: List[Edge] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def refs(self) -> List[str]:
        '''
        Returns:
            List[str]: Documents the path rests on, in order, deduplicated. \
                A path is only as good as its weakest link, and this is what \
                a reader checks.
        '''
        seen = []
        for edge in self.edges:
            if edge.ref not in seen:
                seen.append(edge.ref)

        return seen

    def __len__(self) -> int:
        return len(self.edges)


# edges touching an entity
def neighbors(
    store: GraphStore,
    entity: str,
    limit: int = 30,
    refs: Optional[Sequence[str]] = None
) -> List[GraphHit]:
    '''
    Every claim with this entity at either end.

    Matching is exact on the merge key, or substring: an analyst asking about
    a surname should find the person, and asking about "Nimbus" should find
    "Project Nimbus".

    Args:
        store (GraphStore): The project's graph.
        entity (str): Name to look for.
        limit (int): Most hits to return.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        List[GraphHit]: Claims touching the entity.
    '''
    key = merge_key(entity)
    if not key:
        return []

    hits = [
        GraphHit(edge=edge)
        for edge in store.edges(refs)
        if _matches(edge.source, key) or _matches(edge.target, key)
    ]

    return hits[:limit]


# the shortest chain of claims between two entities
def path_between(
    store: GraphStore,
    source: str,
    target: str,
    max_depth: int = MAX_DEPTH,
    refs: Optional[Sequence[str]] = None
) -> Optional[GraphPath]:
    '''
    Answer "how is A connected to B" with the claims that connect them.

    Breadth-first over the loaded edge set, so the chain returned is the
    shortest one — the fewest assertions a reader has to accept.

    Args:
        store (GraphStore): The project's graph.
        source (str): Where to start.
        target (str): Where to end.
        max_depth (int): Longest chain to consider.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        Optional[GraphPath]: The shortest path, or None when the graph does \
            not connect them within `max_depth`. None means the documents do \
            not assert a connection, not that there is none.
    '''
    start, goal = merge_key(source), merge_key(target)
    if not start or not goal:
        return None

    adjacency = _adjacency(store.edges(refs))

    if start == goal:
        return GraphPath()

    queue = deque([(start, [])])
    seen = {start}

    while queue:
        node, chain = queue.popleft()
        if len(chain) >= max_depth:
            continue

        for neighbor, edge in adjacency.get(node, []):
            if neighbor in seen:
                continue
            extended = chain + [edge]
            if neighbor == goal:
                return GraphPath(edges=extended)
            seen.add(neighbor)
            queue.append((neighbor, extended))

    return None


# everything within a few steps of an entity
def neighborhood(
    store: GraphStore,
    entity: str,
    depth: int = 2,
    limit: int = 60,
    refs: Optional[Sequence[str]] = None
) -> List[GraphHit]:
    '''
    Edges reachable from an entity, nearest first.

    Args:
        store (GraphStore): The project's graph.
        entity (str): Where to start.
        depth (int): How many steps out.
        limit (int): Most hits to return.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        List[GraphHit]: Claims, each carrying how far out it was found.
    '''
    key = merge_key(entity)
    if not key:
        return []

    adjacency = _adjacency(store.edges(refs))
    hits: List[GraphHit] = []
    seen_edges = set()
    frontier = {node for node in adjacency if _matches_key(node, key)}
    visited = set(frontier)

    for step in range(1, depth + 1):
        next_frontier = set()
        for node in frontier:
            for neighbor, edge in adjacency.get(node, []):
                marker = (edge.source, edge.relation, edge.target, edge.ref)
                if marker not in seen_edges:
                    seen_edges.add(marker)
                    hits.append(GraphHit(edge=edge, depth=step))
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier or len(hits) >= limit:
            break

    return hits[:limit]


def _adjacency(edges: Sequence[Edge]) -> Dict[str, List[Tuple[str, Edge]]]:
    '''
    Both directions. A claim that A funded B connects the two whichever end
    the question starts from.
    '''
    graph: Dict[str, List[Tuple[str, Edge]]] = {}
    for edge in edges:
        source, target = merge_key(edge.source), merge_key(edge.target)
        if not source or not target:
            continue
        graph.setdefault(source, []).append((target, edge))
        graph.setdefault(target, []).append((source, edge))

    return graph


def _matches(name: str, key: str) -> bool:
    return _matches_key(merge_key(name), key)


def _matches_key(candidate: str, key: str) -> bool:
    return candidate == key or key in candidate
