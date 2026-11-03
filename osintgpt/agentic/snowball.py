'''Follow an unseen passage at each step of a semantic similarity path.'''

import math
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence

from osintgpt.config import Settings
from osintgpt.llm.base import EmbeddingProvider, EmbeddingPurpose
from osintgpt.llm.usage import UsageRecorder
from osintgpt.operations import OperationContext, OperationOutcome
from osintgpt.projects import Project
from osintgpt.vector_store import BaseVectorEngine, SearchResult

DEFAULT_DEPTH = 25
DEFAULT_THRESHOLD = 0.5
DEFAULT_DEDUPE = 'text'
DEFAULT_TOOL_DEPTH = 5


@dataclass(frozen=True)
class Hop:
    '''An accepted hit; drift is initial-query similarity (higher = less drift).'''

    depth: int
    query: str
    result: SearchResult
    drift: Optional[float] = None
    date: str = ''
    identical_text_refs: List[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return self.result.ref

    @property
    def text(self) -> str:
        return self.result.text


@dataclass(frozen=True)
class Snowball:
    '''A similarity path, its termination reason, and run-local usage.'''

    query: str
    hops: List[Hop] = field(default_factory=list)
    stopped: str = ''
    usage: UsageRecorder = field(default_factory=UsageRecorder)
    embedding_calls: int = 0
    date_field: str = 'timestamp'
    date_format: Optional[str] = None
    depth: int = DEFAULT_DEPTH
    threshold: float = DEFAULT_THRESHOLD
    model: str = ''
    dedupe: str = DEFAULT_DEDUPE
    unrecorded_calls: int = 0
    outcome: Optional[OperationOutcome] = None

    @property
    def refs(self) -> List[str]:
        return list(dict.fromkeys(hop.ref for hop in self.hops))

    @property
    def results(self) -> List[SearchResult]:
        return [hop.result for hop in self.hops]

    def __len__(self) -> int:
        return len(self.hops)


def snowball(
    project: Project,
    query: str,
    embedder: Optional[EmbeddingProvider] = None,
    depth: int = DEFAULT_DEPTH,
    threshold: float = DEFAULT_THRESHOLD,
    score_against_initial: bool = True,
    refs: Optional[Sequence[str]] = None,
    store: Optional[BaseVectorEngine] = None,
    *,
    home: Optional[Path] = None,
    config: Optional[Settings] = None,
    date_field: Optional[str] = None,
    date_format: Optional[str] = None,
    dedupe: str = DEFAULT_DEDUPE,
    overrides: Optional[dict] = None,
    cost_ceiling_usd: Optional[float] = None,
    on_event=None,
) -> Snowball:
    '''Retrieve, re-embed the best unseen hit as a query, and repeat.

    Depth counts accepted hits, including the first. A score equal to the
    threshold is accepted. Exhaustion stops the walk; there is no additional
    hop cap. Spending is limited only by explicitly set project/user/run
    ceilings (the lower applies). Text dedupe uses Unicode NFKC, case folding and collapsed
    whitespace. ``dedupe='chunk'`` keeps separate file/sequence identities.
    This is a similarity path, not evidence of
    causation or propagation.

    Omit embedder/store to use project settings, user defaults and connection
    configuration from ``home`` (the normal application home by default).
    Injected providers must match the index model. Caller-owned stores and
    provider recorders are left intact, including any caller-imposed policy.
    Built-in providers emit usage events; custom providers without events
    produce an explicitly incomplete cost estimate.

    Each pivot uses QUERY-purpose embedding; optional initial similarity uses
    DOCUMENT-purpose embedding of each accepted hit. No stored vectors are
    reused. Date fields name ``timestamp`` or an indexed metadata key. Dates
    never filter or reorder traversal; ``date_format`` is used only by output.
    '''
    _validate(query, depth, threshold)
    if dedupe not in ('text', 'chunk'):
        raise ValueError('Dedupe must be text or chunk.')
    query = query.strip()
    selected_date = (date_field or '').strip() or 'timestamp'
    operation = OperationContext(
        project, home=home, config=config, embedder=embedder, store=store,
        overrides=overrides, cost_ceiling_usd=cost_ceiling_usd, on_event=on_event,
    )
    with operation:
        # Validate the selected model before opening a store or making a call.
        embedder = operation.embedder
        result = _walk(
            query, embedder, operation.store, depth, threshold,
            score_against_initial, refs, selected_date, date_format, operation, dedupe,
        )
    return replace(result, outcome=operation.outcome)


def _validate(query, depth, threshold):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Enter a nonempty Snowball query.')
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError('Depth must be a positive integer.')
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError('Threshold must be a finite cosine score from -1 to 1.')


def _walk(query, embedder, engine, depth, threshold, score_against_initial,
          refs, date_field, date_format, operation, dedupe):
    calls = 0
    dimensions = None

    def embed(text, purpose):
        nonlocal calls, dimensions
        calls += 1
        vectors = operation.embed([text], purpose=purpose)
        if len(vectors) != 1 or not len(vectors[0]):
            raise ValueError('Embedding provider returned no usable vector.')
        vector = list(vectors[0])
        if not all(math.isfinite(value) for value in vector):
            raise ValueError('Embedding provider returned a nonfinite vector.')
        if dimensions is not None and len(vector) != dimensions:
            raise ValueError('Embedding dimensions changed during the Snowball run.')
        dimensions = len(vector)
        return vector

    initial = embed(query, EmbeddingPurpose.QUERY)
    current_vector = initial
    current = query
    hops = []
    seen = set()
    copies = {}
    stopped = f'reached depth {depth}'
    for step in range(1, depth + 1):
        nxt = _next_hit(engine, current_vector, embedder.model, refs, seen, copies, dedupe)
        if nxt is None:
            stopped = 'nothing new to walk to'
            break
        if not math.isfinite(nxt.score):
            raise ValueError('Store returned a nonfinite similarity score.')
        if nxt.score < threshold:
            stopped = f'similarity {nxt.score:.2f} fell below {threshold:.2f}'
            break

        drift = None
        if score_against_initial:
            vector = embed(nxt.text, EmbeddingPurpose.DOCUMENT)
            drift = _cosine(initial, vector)
        value = (nxt.chunk.timestamp if date_field == 'timestamp'
                 else nxt.chunk.metadata.get(date_field))
        hops.append(Hop(step, current, nxt, drift, '' if value is None else str(value),
                        copies[_text_key(nxt.text)]))
        seen.add(_identity(nxt, dedupe))
        if step < depth:
            current = nxt.text
            current_vector = embed(current, EmbeddingPurpose.QUERY)

    return Snowball(
        query=query, hops=hops, stopped=stopped, usage=operation.usage, embedding_calls=calls,
        date_field=date_field, date_format=date_format, depth=depth,
        threshold=threshold, model=embedder.model, dedupe=dedupe,
        unrecorded_calls=operation.unrecorded_calls,
    )


def _text_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def _identity(result, dedupe):
    return _text_key(result.text) if dedupe == 'text' else (result.ref, result.chunk.sequence)


def _next_hit(engine, vector, model, refs, seen, copies, dedupe):
    limit = max(5, len(seen) + 1)
    while True:
        found = engine.search(vector, embedding_model=model, top_k=limit, refs=refs)
        for result in found:
            sources = copies.setdefault(_text_key(result.text), [])
            citation = result.chunk.citation
            if citation not in sources:
                sources.append(citation)
        nxt = next((r for r in found if _identity(r, dedupe) not in seen), None)
        if nxt is not None or len(found) < limit:
            return nxt
        # Duplicate text can occupy any number of chunks. Grow past every
        # skipped copy rather than treating a full duplicate window as exhaustion.
        limit = max(limit * 2, len(found) + 1)


def _cosine(initial, vector):
    if len(initial) != len(vector):
        raise ValueError('Embedding dimensions changed during the Snowball run.')
    norms = math.sqrt(sum(a * a for a in initial)) * math.sqrt(sum(b * b for b in vector))
    if not norms:
        return 0.0
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(initial, vector)) / norms))
