# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: fusion.py
# Description: Combining retrieval legs that score on different scales.
#   Reciprocal rank fusion uses position rather than score, which is the only
#   thing two legs can honestly agree on.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# import osintgpt vector store
from osintgpt.vector_store import SearchResult

# The constant from the original RRF paper. It damps the difference between
# the first few positions so a leg that ranks something second is not treated
# as having nearly missed it, and it is large enough that the tail contributes
# almost nothing. Changing it is a retrieval change, so measure it.
RRF_K = 60


# FusedResult class
@dataclass(frozen=True)
class FusedResult:
    '''
    One chunk, its fused score, and where each leg placed it.
    '''
    result: SearchResult
    score: float
    # Leg name to the 1-based rank it gave this chunk. A leg absent from this
    # mapping did not return the chunk at all, which is itself informative:
    # agreement between legs is the signal RRF rewards.
    ranks: Dict[str, int] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return self.result.ref

    @property
    def text(self) -> str:
        return self.result.text

    @property
    def legs(self) -> List[str]:
        '''
        Returns:
            List[str]: Legs that found this chunk, in the order they ranked \
                it best first.
        '''
        return sorted(self.ranks, key=lambda leg: self.ranks[leg])

    @property
    def found_by_all(self) -> bool:
        return len(self.ranks) > 1


# fuse ranked lists from several retrieval legs
def reciprocal_rank_fusion(
    legs: Mapping[str, Sequence[SearchResult]],
    k: int = RRF_K,
    limit: int = 10,
    weights: Mapping[str, float] = None
) -> List[FusedResult]:
    '''
    Merge ranked lists by position rather than by score.

    Scores from different legs are not comparable — a cosine similarity of
    0.62 and a lexical term coverage of 0.5 measure different things on
    different scales, and normalizing them would invent a relationship that
    is not there. Rank is what both legs genuinely produce, so rank is what is
    combined.

    A chunk found by two legs outranks one found deeply by either, which is
    the property that makes fusion worth doing at all.

    Args:
        legs (Mapping[str, Sequence[SearchResult]]): Results per leg, each \
            already ordered best first.
        k (int): Damping constant. Larger flattens the contribution of the \
            top positions.
        limit (int): How many fused results to return.
        weights (Mapping[str, float], optional): Per-leg multipliers, for a \
            caller that has measured one leg to be better on its corpus. \
            Absent legs weigh 1.0.

    Returns:
        List[FusedResult]: Best first.
    '''
    weights = weights or {}
    scores: Dict[Tuple[str, int], float] = {}
    ranks: Dict[Tuple[str, int], Dict[str, int]] = {}
    seen: Dict[Tuple[str, int], SearchResult] = {}

    for leg, results in legs.items():
        weight = float(weights.get(leg, 1.0))
        for position, result in enumerate(_deduplicated(results), 1):
            key = (result.chunk.ref, result.chunk.sequence)
            scores[key] = scores.get(key, 0.0) + weight / (k + position)
            ranks.setdefault(key, {})[leg] = position
            # The first leg to return a chunk owns the copy that is kept.
            # They carry the same text and provenance; only the score
            # differs, and the fused score replaces it.
            seen.setdefault(key, result)

    fused = [
        FusedResult(result=seen[key], score=score, ranks=ranks[key])
        for key, score in scores.items()
    ]

    # Ties are broken by how many legs agreed, then by document and reading
    # order, so the same inputs always produce the same output.
    fused.sort(
        key=lambda item: (
            -item.score, -len(item.ranks),
            item.result.chunk.ref, item.result.chunk.sequence
        )
    )

    return fused[:limit]


def _deduplicated(results: Sequence[SearchResult]) -> Iterable[SearchResult]:
    '''
    A leg returning the same chunk twice would have it contribute twice.
    '''
    seen = set()
    for result in results:
        key = (result.chunk.ref, result.chunk.sequence)
        if key in seen:
            continue
        seen.add(key)
        yield result
