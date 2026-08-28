# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: tools.py
# Description: The retrieval tools a model may call. Deterministic Python —
#   which tool to call, and when, is the model's decision and never encoded here.
# =================================================================================

# import modules
import logging

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Any, Dict, List, Optional, Sequence

# import osintgpt graph
from osintgpt.graph import graph_for, neighbors, path_between

# import osintgpt lexical
from osintgpt.lexical import lexical_search

# import osintgpt search
from osintgpt.search import search_project

from .support import (
    SNIPPET_CHARS,
    _claim,
    _clamp,
    _dating_note,
    _passage,
    _read,
    _resolve,
    _within_days
)

log = logging.getLogger('osintgpt.agentic')

# Content costs context; a count does not. This is what lets a survey range
# wider than a fetch — the difference the roadmap records as measurably
# improving answer quality.
SNIPPETS = 'snippets'
REFS = 'refs'

# Lines returned by one fetch_source call. A model that needs more asks again
# with the offset it was handed.
FETCH_LINES = 200

# When a `days` filter is asked for, retrieve this many times the limit first,
# because filtering after ranking would otherwise return far fewer than asked.
TIME_FILTER_OVERSAMPLE = 4


# ToolContext class
@dataclass
class ToolContext:
    '''
    Everything the tools need, so a tool function takes only its arguments.
    '''
    project: Any
    embedder: Any
    # Only the graph tool needs it, and only when the project enabled one.
    generator: Any = None
    store: Any = None

    @property
    def root(self) -> Path:
        return Path(self.project.paths.root).resolve()


# ToolResult class
@dataclass(frozen=True)
class ToolResult:
    '''
    What one call returned, and what it cost to say so.
    '''
    tool: str
    # What the model is shown. Kept separate from the trace, because the
    # trace records that a call happened and this is the call's answer.
    payload: Dict[str, Any] = field(default_factory=dict)
    count: int = 0
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error


# search by meaning
def semantic_search(
    context: ToolContext,
    query: str,
    limit: int = 8,
    days: Optional[int] = None,
    refs: Optional[Sequence[str]] = None
) -> ToolResult:
    '''
    Passages that mean something close to the query.

    Args:
        context (ToolContext): Project and providers.
        query (str): What to search for, in the model's own words.
        limit (int): Most passages to return.
        days (int, optional): Only documents timestamped within this many \
            days. The model decides what "last week" means and passes a \
            number; nothing here parses a phrase.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        ToolResult: Passages with their citations and scores.
    '''
    wanted = _clamp(limit, 1, 30)
    found = search_project(
        context.project, query, context.embedder,
        top_k=wanted * TIME_FILTER_OVERSAMPLE if days else wanted,
        refs=refs, store=context.store
    )

    found, undated = _within_days(found, days)
    found = found[:wanted]

    return ToolResult(
        tool='semantic_search',
        payload={
            'passages': [_passage(r) for r in found],
            **_dating_note(days, undated)
        },
        count=len(found)
    )


# search for literal strings
def exact_search(
    context: ToolContext,
    terms: Sequence[str],
    mode: str = SNIPPETS,
    limit: int = 20,
    days: Optional[int] = None,
    refs: Optional[Sequence[str]] = None
) -> ToolResult:
    '''
    Chunks containing the given strings, exactly as written.

    `mode="refs"` is the survey primitive: it answers *where* matches are and
    how many, with no content at all. That costs almost nothing, so a model
    can range several times wider before deciding what to actually read.

    Args:
        context (ToolContext): Project and providers.
        terms (Sequence[str]): Literal strings — handles, hashes, URLs, names.
        mode (str): `snippets` for content, `refs` for locations and counts.
        limit (int): Most results to consider.
        days (int, optional): Only documents timestamped within this many days.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        ToolResult: Snippets, or a per-document count.
    '''
    found = lexical_search(
        context.project, list(terms),
        limit=_clamp(limit, 1, 200),
        embedding_model=getattr(context.embedder, 'model', None),
        refs=refs, store=context.store
    )

    if mode == REFS:
        dated, _ = _within_days(found, days)
        counts: Dict[str, int] = {}
        for result in dated:
            counts[result.ref] = counts.get(result.ref, 0) + 1

        return ToolResult(
            tool='exact_search',
            payload={
                'mode': REFS,
                'documents': [
                    {'ref': ref, 'matches': n}
                    for ref, n in sorted(
                        counts.items(), key=lambda i: (-i[1], i[0])
                    )
                ]
            },
            count=len(counts)
        )

    kept, undated = _within_days(found, days)

    return ToolResult(
        tool='exact_search',
        payload={
            'mode': SNIPPETS,
            'passages': [_passage(r) for r in kept[:_clamp(limit, 1, 30)]],
            **_dating_note(days, undated)
        },
        count=len(kept)
    )


# follow a thread outward
def snowball_search(
    context: ToolContext,
    query: str,
    depth: int = 5,
    threshold: float = 0.5
) -> ToolResult:
    '''
    Walk from a passage to what is adjacent to it, and so on.

    Answers "what else is near this" rather than "what matches this", so it
    reaches material the opening question would not have found. Each hop is
    scored against the original question too, and that drift is returned:
    a walk can stay locally coherent while ending somewhere unrelated.

    Args:
        context (ToolContext): Project and providers.
        query (str): Where to start.
        depth (int): Most hops.
        threshold (float): Stop when similarity falls below this.

    Returns:
        ToolResult: The hops, with drift, and why the walk stopped.
    '''
    from .snowball import snowball

    walk = snowball(
        context.project, query, context.embedder,
        depth=_clamp(depth, 1, 10), threshold=threshold, store=context.store
    )

    return ToolResult(
        tool='snowball',
        payload={
            'stopped': walk.stopped,
            'hops': [
                {
                    'depth': hop.depth,
                    'citation': hop.result.chunk.citation,
                    'text': hop.text[:SNIPPET_CHARS],
                    'score': round(hop.result.score, 4),
                    'drift_from_question': (
                        round(hop.drift, 4) if hop.drift is not None else None
                    )
                }
                for hop in walk.hops
            ]
        },
        count=len(walk.hops)
    )


# ask the graph
def graph_query(
    context: ToolContext,
    entity: str,
    target: Optional[str] = None,
    limit: int = 20
) -> ToolResult:
    '''
    What the documents assert about an entity, or how two are connected.

    Every claim returned carries the document and the sentence asserting it,
    because a relationship without its evidence is not a finding.

    Args:
        context (ToolContext): Project and providers.
        entity (str): The name to ask about.
        target (str, optional): Given one, returns the chain connecting them.
        limit (int): Most claims to return.

    Returns:
        ToolResult: Claims with their evidence, or the connecting chain.
    '''
    with graph_for(context.project) as graph:
        if not graph.is_built:
            return ToolResult(
                tool='graph_query',
                payload={
                    'built': False,
                    'note': 'This project has no graph. Relationships are '
                            'not available; use the other tools.'
                }
            )

        if target:
            path = path_between(graph, entity, target)
            if path is None:
                return ToolResult(
                    tool='graph_query',
                    payload={
                        'connected': False,
                        'note': f'The documents do not assert a connection '
                                f'between {entity} and {target}. That is not '
                                f'the same as there being none.'
                    }
                )

            return ToolResult(
                tool='graph_query',
                payload={'connected': True, 'path': [_claim(e) for e in path.edges]},
                count=path.length
            )

        hits = neighbors(graph, entity, limit=_clamp(limit, 1, 60))

        return ToolResult(
            tool='graph_query',
            payload={'claims': [_claim(hit.edge) for hit in hits]},
            count=len(hits)
        )


# what documents exist
def list_documents(
    context: ToolContext,
    pattern: Optional[str] = None,
    limit: int = 100
) -> ToolResult:
    '''
    The documents in the project, optionally those whose ref contains a string.

    A survey tool: it says what there is to read before anything is read.

    Args:
        context (ToolContext): Project and providers.
        pattern (str, optional): Substring the ref must contain.
        limit (int): Most documents to list.

    Returns:
        ToolResult: Document refs and how many chunks each holds.
    '''
    from osintgpt.vector_store import store_for

    owned = context.store is None
    engine = context.store or store_for(context.project)

    try:
        refs = engine.refs(getattr(context.embedder, 'model', None))
    finally:
        if owned and hasattr(engine, 'close'):
            engine.close()

    if pattern:
        needle = pattern.lower()
        refs = [ref for ref in refs if needle in ref.lower()]

    return ToolResult(
        tool='list_documents',
        payload={'documents': refs[:_clamp(limit, 1, 500)]},
        count=len(refs)
    )


# read a document
def fetch_source(
    context: ToolContext,
    ref: str,
    offset: int = 0,
    limit: int = FETCH_LINES
) -> ToolResult:
    '''
    Read a window of a document, by line.

    Windowed because a long document either fills the context or gets cut
    without saying so. When there is more, the result carries `next_offset`,
    so the model can continue rather than guess.

    Args:
        context (ToolContext): Project and providers.
        ref (str): The document, as the corpus refers to it.
        offset (int): First line to return, zero-based.
        limit (int): Most lines to return.

    Returns:
        ToolResult: The lines, and where to continue from.
    '''
    path = _resolve(context, ref)
    if path is None:
        return ToolResult(
            tool='fetch_source',
            error=f'{ref} is not a document in this project'
        )

    try:
        text = _read(context, path, ref)
    except Exception as error:  # noqa: BLE001 — one call, not the loop
        return ToolResult(tool='fetch_source', error=str(error))

    lines = text.splitlines()
    start = max(int(offset), 0)
    end = start + _clamp(limit, 1, 1000)
    window = lines[start:end]

    payload: Dict[str, Any] = {
        'ref': ref,
        'offset': start,
        'lines': len(window),
        'total_lines': len(lines),
        'text': '\n'.join(window)
    }
    if end < len(lines):
        payload['next_offset'] = end

    return ToolResult(
        tool='fetch_source', payload=payload, count=len(window)
    )
