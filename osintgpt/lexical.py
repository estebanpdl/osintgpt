# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: lexical.py
# Description: Exact search over a project's stored text. The leg that catches
#   what embeddings blur — handles, hashes, URLs, account ids, error codes.
# =================================================================================

# import modules
import json
import logging
import re

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Iterable, List, Optional, Sequence

# import osintgpt llm
from osintgpt.llm.base import GenerationProvider

# import osintgpt projects
from osintgpt.projects import Project

# import osintgpt prompts
from osintgpt.prompts import prompt

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, SearchResult, StoredChunk

log = logging.getLogger('osintgpt.lexical')

# Enough terms to cover a question with several identifiers in it, few enough
# that a model padding its list cannot swamp the results.
MAX_TERMS = 8

# A single character matches everything and ranks nothing.
MIN_TERM_LENGTH = 2

# Chunks scanned per term before giving up. A term this common is not
# selective, and returning the first thousand of it helps nobody.
DEFAULT_LIMIT = 200


# LexicalHit class
@dataclass(frozen=True)
class LexicalHit:
    '''
    One chunk and the terms that were found in it.
    '''
    chunk: StoredChunk
    terms: List[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return self.chunk.ref

    @property
    def text(self) -> str:
        return self.chunk.text

    # how well this chunk answers the term set
    def score(self, total_terms: int) -> float:
        '''
        Args:
            total_terms (int): How many terms were searched for.

        Returns:
            float: Share of the searched terms this chunk contains. Coverage \
                rather than frequency: a chunk mentioning three of the four \
                things asked about beats one repeating a single term twenty \
                times.
        '''
        return len(self.terms) / total_terms if total_terms else 0.0


# ask the model which literal strings to search for
def derive_search_terms(
    generator: GenerationProvider,
    query: str,
    max_terms: int = MAX_TERMS
) -> List[str]:
    '''
    Choose exact-match terms from a question.

    The terms come from the model rather than from a stopword list, because
    every stopword list is bound to one language and this tool is not. A list
    that strips English function words leaves a question in Turkish or Arabic
    mostly intact, and the terms it yields are noise.

    Fails soft: on a model error or an unparseable reply the lexical leg sits
    out, and the semantic leg still answers.

    Args:
        generator (GenerationProvider): Chooses the terms.
        query (str): The question, as asked.
        max_terms (int): Ceiling on how many to return.

    Returns:
        List[str]: Literal strings to search for, deduplicated, possibly \
            empty when the question names nothing exact.
    '''
    try:
        reply = generator.generate(
            prompt('search_terms', max_terms=max_terms), query
        )
    except Exception as error:  # noqa: BLE001 — a leg sitting out, not a stop
        log.warning('term derivation failed; skipping the lexical leg: %s',
                    error)

        return []

    terms = _parse_terms(reply)
    if not terms:
        log.info('no exact terms in %r; the lexical leg has nothing to do',
                 query[:80])

    return terms[:max_terms]


# search a project's stored text for literal strings
def lexical_search(
    project: Project,
    terms: Sequence[str],
    limit: int = DEFAULT_LIMIT,
    embedding_model: Optional[str] = None,
    refs: Optional[Iterable[str]] = None,
    store: Optional[BaseVectorEngine] = None
) -> List[SearchResult]:
    '''
    Find chunks containing any of the given terms, ranked by how many.

    Runs over the stored text rather than the files on disk. The store holds
    what was indexed, with its provenance already attached, so both retrieval
    legs see one corpus — and re-reading the sources per query would re-extract
    every PDF, which is seconds per document.

    Args:
        project (Project): The project to search.
        terms (Sequence[str]): Literal strings, typically from \
            `derive_search_terms`.
        limit (int): Chunks to scan per term.
        embedding_model (str, optional): Restrict to one model's chunks, \
            which a store holding two models' vectors needs to avoid \
            returning every chunk twice.
        refs (Iterable[str], optional): Restrict to these documents.
        store (BaseVectorEngine, optional): Defaults to the project's own.

    Returns:
        List[SearchResult]: Best first, scored by term coverage.
    '''
    usable = _usable(terms)
    if not usable:
        return []

    from osintgpt.vector_store import store_for

    owned = store is None
    engine = store or store_for(project)

    try:
        found = _collect(engine, usable, limit, embedding_model, refs)
    finally:
        if owned and hasattr(engine, 'close'):
            engine.close()

    hits = sorted(
        found.values(),
        key=lambda hit: (-len(hit.terms), hit.chunk.ref, hit.chunk.sequence)
    )

    return [
        SearchResult(chunk=hit.chunk, score=hit.score(len(usable)))
        for hit in hits
    ]


def _collect(engine, terms, limit, embedding_model, refs) -> dict:
    '''
    One pass per term, gathered by chunk. A chunk found by three terms is one
    result carrying three, not three results.
    '''
    found = {}
    for term in terms:
        for chunk in engine.match_text(
            term, embedding_model=embedding_model, limit=limit, refs=refs
        ):
            key = (chunk.ref, chunk.sequence)
            if key in found:
                found[key].terms.append(term)
            else:
                found[key] = LexicalHit(chunk=chunk, terms=[term])

    return found


def _usable(terms: Sequence[str]) -> List[str]:
    '''
    Deduplicated, case-folded, and long enough to be selective.
    '''
    seen: List[str] = []
    for term in terms:
        cleaned = (term or '').strip()
        if len(cleaned) < MIN_TERM_LENGTH:
            continue
        if cleaned.lower() not in [s.lower() for s in seen]:
            seen.append(cleaned)

    return seen


def _parse_terms(reply: str) -> List[str]:
    '''
    Read a JSON array out of a model reply.

    Models wrap JSON in prose or a code fence often enough that finding the
    array is worth doing, and a reply that yields nothing costs the leg rather
    than the query.
    '''
    text = (reply or '').strip()

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []

    if not isinstance(parsed, list):
        return []

    return [
        item.strip() for item in parsed
        if isinstance(item, str) and item.strip()
    ]
