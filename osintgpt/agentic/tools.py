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
from osintgpt.graph.retrieval import discover_entities, query_graph
from osintgpt.graph.evidence import assertion_context

# import osintgpt lexical
from osintgpt.lexical import lexical_search

# import osintgpt search
from osintgpt.search import search_project
from .snowball import DEFAULT_TOOL_DEPTH, DEFAULT_THRESHOLD, DEFAULT_DEDUPE

from .support import (
    SNIPPET_CHARS,
    _clamp,
    _passage,
    _read,
    _resolve
)

log = logging.getLogger('osintgpt.agentic')

# Content costs context; a count does not. This is what lets a survey range
# wider than a fetch, which is where most of the answer quality comes from.
SNIPPETS = 'snippets'
REFS = 'refs'

# Lines returned by one fetch_source call. A model that needs more asks again
# with the offset it was handed.
FETCH_LINES = 200


# ToolContext class
@dataclass
class ToolContext:
    '''
    Everything the tools need, so a tool function takes only its arguments.
    '''
    project: Any
    embedder: Any
    generator: Any = None
    store: Any = None
    # The corpus and the files it covers, built on first use. Checking a ref
    # and reading a document both need them, and a corpus does not change
    # while one question is being answered — so neither should re-walk it.
    _corpus: Any = field(default=None, repr=False)
    _covered: Any = field(default=None, repr=False)

    @property
    def root(self) -> Path:
        return Path(self.project.paths.root).resolve()

    @property
    def corpus(self) -> Any:
        '''
        Returns:
            Corpus: The project's registered sources.
        '''
        if self._corpus is None:
            from osintgpt.ingestion import Corpus

            self._corpus = Corpus.load(self.project.paths.sources)

        return self._corpus

    @property
    def covered(self) -> set:
        '''
        Returns:
            set: Every file the registered corpus covers, resolved. The same \
                set `index_project` walks, so what can be read is exactly \
                what could have been indexed.
        '''
        if self._covered is None:
            self._covered = {
                path.resolve() for path in self.corpus.files(self.root)
            }

        return self._covered


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
    # What `count` counts, singular. Set by the tool because only the tool
    # knows: the same tool returns documents in one mode and passages in
    # another, so naming the unit from the tool name would be wrong half the
    # time. A trace reading "88 results" cannot be told from "88 passages".
    unit: str = 'result'
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error


# search by meaning
def semantic_search(
    context: ToolContext,
    query: str,
    limit: int = 8,
    refs: Optional[Sequence[str]] = None
) -> ToolResult:
    '''
    Passages that mean something close to the query.

    Args:
        context (ToolContext): Project and providers.
        query (str): What to search for, in the model's own words.
        limit (int): Most passages to return.
        refs (Sequence[str], optional): Restrict to these documents.

    Returns:
        ToolResult: Passages with their citations and scores.
    '''
    found = search_project(
        context.project, query, context.embedder,
        top_k=_clamp(limit, 1, 30), refs=refs, store=context.store
    )

    return ToolResult(
        tool='semantic_search',
        payload={'passages': [_passage(r) for r in found]},
        count=len(found),
        unit='passage'
    )


# search for literal strings
def exact_search(
    context: ToolContext,
    terms: Sequence[str],
    mode: str = SNIPPETS,
    limit: int = 20,
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
        counts: Dict[str, int] = {}
        # Which terms hit each document, in the order they were searched. A
        # count says how loud a document is; the terms say what it is loud
        # about, and two documents with the same count can differ entirely.
        matched: Dict[str, List[str]] = {}
        for result in found:
            counts[result.ref] = counts.get(result.ref, 0) + 1
            seen = matched.setdefault(result.ref, [])
            for term in result.terms:
                if term not in seen:
                    seen.append(term)

        return ToolResult(
            tool='exact_search',
            payload={
                'mode': REFS,
                'documents': [
                    {
                        'ref': ref,
                        'matches': n,
                        **({'terms': matched[ref]} if matched.get(ref) else {})
                    }
                    for ref, n in sorted(
                        counts.items(), key=lambda i: (-i[1], i[0])
                    )
                ]
            },
            count=len(counts),
            unit='document'
        )

    return ToolResult(
        tool='exact_search',
        payload={
            'mode': SNIPPETS,
            'passages': [_passage(r) for r in found[:_clamp(limit, 1, 30)]]
        },
        count=len(found),
        unit='passage'
    )


# follow a thread outward
def snowball_search(
    context: ToolContext,
    query: str,
    depth: int = DEFAULT_TOOL_DEPTH,
    threshold: float = DEFAULT_THRESHOLD,
    refs: Optional[Sequence[str]] = None,
    date_field: Optional[str] = None,
    date_format: Optional[str] = None,
    score_against_initial: bool = True,
    dedupe: str = DEFAULT_DEDUPE
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
        depth=depth, threshold=threshold, store=context.store, refs=refs,
        date_field=date_field, date_format=date_format,
        score_against_initial=score_against_initial, dedupe=dedupe
    )
    from .snowball_output import snowball_data

    data = snowball_data(walk)

    return ToolResult(
        tool='snowball',
        payload={
            **{key: value for key, value in data.items() if key != 'hops'},
            'stopped': walk.stopped,
            'hops': [
                {
                    'depth': hop.depth,
                    'query': hop.query,
                    'ref': hop.result.ref,
                    'document_ref': hop.result.chunk.document_ref,
                    'sequence': hop.result.chunk.sequence,
                    'citation': hop.result.chunk.citation,
                    'text': hop.text[:SNIPPET_CHARS],
                    'score': round(hop.result.score, 4),
                    'date': hop.date,
                    'identical_text_refs': list(hop.identical_text_refs),
                    'similarity_to_query': hop.drift,
                    **(
                        {'timestamp': hop.result.chunk.timestamp}
                        if hop.result.chunk.timestamp else {}
                    ),
                    **(
                        {'author': hop.result.chunk.author}
                        if hop.result.chunk.author else {}
                    ),
                    'drift_from_question': (
                        round(hop.drift, 4) if hop.drift is not None else None
                    )
                }
                for hop in walk.hops
            ]
        },
        count=len(walk.hops),
        unit='hop'
    )


def graph_query(context: ToolContext, entity: str, target: Optional[str] = None, limit: int = 20,
                direction: str = 'connectivity', depth: Optional[int] = None,
                max_bytes: int = 24000, refs=None) -> ToolResult:
    return _graph_result('graph_query', lambda: query_graph(
        context.project, entity, target, limit=limit, direction=direction, depth=depth,
        max_bytes=max_bytes, refs=refs), 'path' if target else 'claims', 'step' if target else 'claim')


def graph_entities(context: ToolContext, query: str, limit: int = 20, max_bytes: int = 24000, refs=None) -> ToolResult:
    return _graph_result('graph_entities', lambda: discover_entities(
        context.project, query, limit=limit, max_bytes=max_bytes, refs=refs), 'candidates', 'entity')


def graph_evidence(context: ToolContext, assertion_id: str, quote_index: int = 0, occurrence: int = 0,
                   context_chars: int = 800, offset: int = 0, limit_chars: int = 4000,
                   max_bytes: int = 24000) -> ToolResult:
    return _graph_result('graph_evidence', lambda: assertion_context(
        context.project, assertion_id, quote_index=quote_index, occurrence=occurrence,
        context_chars=context_chars, offset=offset, limit_chars=limit_chars, max_bytes=max_bytes), 'text', 'character')


def _graph_result(name, operation, collection, unit):
    try:
        payload = operation()
    except ValueError as error:
        return ToolResult(tool=name, error=str(error))
    return ToolResult(tool=name, payload=payload, count=len(payload.get(collection, [])), unit=unit)


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
        count=len(refs),
        unit='document'
    )


# read a document
def fetch_source(
    context: ToolContext,
    ref: str,
    offset: int = 0,
    limit: int = FETCH_LINES,
    source_version: str = ''
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
        text, provenance = _read(context, path, ref, source_version)
    except Exception as error:  # noqa: BLE001 — one call, not the loop
        return ToolResult(tool='fetch_source', error=str(error))

    lines = text.splitlines()
    start = max(int(offset), 0)
    end = start + _clamp(limit, 1, 1000)
    window = lines[start:end]
    original_lines = text.splitlines(keepends=True)
    char_start = sum(len(line) for line in original_lines[:start])
    char_end = char_start + len(''.join(original_lines[start:end]).rstrip('\r\n'))
    provenance['records'] = [record for record in provenance['records']
                             if record['start'] < char_end and record['end'] > char_start]
    for record in provenance['records']:
        record['locations'] = [span for span in record['locations']
                               if span['start'] + record['start'] < char_end
                               and span['end'] + record['start'] > char_start]

    payload: Dict[str, Any] = {
        'ref': ref,
        'offset': start,
        'lines': len(window),
        'total_lines': len(lines),
        'text': '\n'.join(window),
        'provenance': provenance
    }
    if end < len(lines):
        payload['next_offset'] = end

    return ToolResult(
        tool='fetch_source', payload=payload, count=len(window),
        unit='line'
    )
