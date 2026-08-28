# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: extraction.py
# Description: Reading entities and relationships out of a whole document.
#   Whole, because chunk boundaries break the coreference the extraction needs.
# =================================================================================

# import modules
import json
import logging
import re

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import List, Sequence

# import osintgpt llm
from osintgpt.llm.base import GenerationProvider

# import osintgpt prompts
from osintgpt.prompts import prompt

from .store import Edge, Entity, merge_key

log = logging.getLogger('osintgpt.graph')

# A document longer than this is windowed. Large enough that most documents
# arrive whole — which is the point, since a window boundary is exactly where
# a name stops resolving — and small enough to sit inside any current model's
# context alongside the instructions.
WINDOW_CHARS = 24_000

# Overlap between windows, so a sentence spanning the cut is asserted in one
# of them rather than truncated in both.
WINDOW_OVERLAP = 1_000


# Extraction class
@dataclass(frozen=True)
class Extraction:
    '''
    What one document yielded.
    '''
    ref: str
    entities: List[Entity] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    # Set when the model could not be read. One document failing is not the
    # pass failing.
    problem: str = ''

    @property
    def ok(self) -> bool:
        return not self.problem


# read one document's entities and relationships
def extract_document(
    generator: GenerationProvider,
    ref: str,
    text: str,
    window_chars: int = WINDOW_CHARS
) -> Extraction:
    '''
    Extract from a whole document, windowing only when it is too long.

    Chunking is a retrieval artifact. Extracting from chunks breaks
    coreference — "the group" and its name three paragraphs earlier land in
    different calls — so the document goes in whole, and where it cannot, each
    window carries the names already found so the same thing does not become
    two nodes.

    Args:
        generator (GenerationProvider): Reads the document.
        ref (str): The document, as the corpus refers to it.
        text (str): Its full text.
        window_chars (int): Longest text sent in one call.

    Returns:
        Extraction: Entities and edges, or the problem that prevented them.
    '''
    body = (text or '').strip()
    if not body:
        return Extraction(ref=ref)

    windows = _windows(body, window_chars)
    entities: List[Entity] = []
    edges: List[Edge] = []
    known: List[str] = []

    for index, window in enumerate(windows, 1):
        try:
            reply = generator.generate(
                prompt(
                    'graph_extraction',
                    ref=ref,
                    text=window,
                    known_entities=known,
                    part=f'{index} of {len(windows)}' if len(windows) > 1 else ''
                ),
                'Extract the entities and relationships.'
            )
        except Exception as error:  # noqa: BLE001 — one document, not the pass
            log.warning('%s: extraction call failed: %s', ref, error)

            return Extraction(ref=ref, problem=str(error))

        found_entities, found_edges = _parse(reply, ref)
        entities.extend(found_entities)
        edges.extend(found_edges)

        for entity in found_entities:
            if entity.name not in known:
                known.append(entity.name)

    return Extraction(
        ref=ref, entities=_merged(entities), edges=_evidenced(edges, ref)
    )


def _windows(text: str, window_chars: int) -> List[str]:
    '''
    One window unless the document is too long, then overlapping slices.
    '''
    if len(text) <= window_chars:
        return [text]

    step = max(window_chars - WINDOW_OVERLAP, window_chars // 2)

    return [
        text[start:start + window_chars]
        for start in range(0, len(text), step)
        if text[start:start + window_chars].strip()
    ]


def _parse(reply: str, ref: str):
    '''
    Read the JSON object out of a reply, tolerating prose around it.
    '''
    match = re.search(r'\{.*\}', (reply or '').strip(), re.DOTALL)
    if not match:
        log.warning('%s: no JSON in the extraction reply', ref)

        return [], []

    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        log.warning('%s: unparseable extraction reply', ref)

        return [], []

    if not isinstance(parsed, dict):
        return [], []

    entities = [
        Entity(
            key=merge_key(str(item.get('name', ''))),
            name=str(item.get('name', '')).strip(),
            type=str(item.get('type', '')).strip(),
            mentions=1
        )
        for item in _rows(parsed.get('entities'))
        if str(item.get('name', '')).strip()
    ]

    edges = [
        Edge(
            source=str(item.get('source', '')).strip(),
            target=str(item.get('target', '')).strip(),
            relation=str(item.get('relation', '')).strip(),
            ref=ref,
            evidence=str(item.get('evidence', '')).strip()
        )
        for item in _rows(parsed.get('edges'))
    ]

    return entities, edges


def _rows(value) -> List[dict]:
    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, dict)]


def _merged(entities: Sequence[Entity]) -> List[Entity]:
    '''
    One row per key, keeping the first spelling and summing mentions.
    '''
    merged = {}
    for entity in entities:
        if not entity.key:
            continue
        held = merged.get(entity.key)
        if held is None:
            merged[entity.key] = entity
        else:
            merged[entity.key] = Entity(
                key=held.key, name=held.name,
                type=held.type or entity.type,
                mentions=held.mentions + entity.mentions
            )

    return list(merged.values())


def _evidenced(edges: Sequence[Edge], ref: str) -> List[Edge]:
    '''
    Only edges that carry both endpoints, a relation, and a quotable sentence.

    An edge without its evidence is a model assertion rather than a sourced
    claim, and the difference is the whole reason the graph is admissible in
    OSINT work. Dropping it silently is right: the model was asked not to
    produce one, and keeping it would put an unsourced claim in a store whose
    contract says there are none.
    '''
    kept = []
    dropped = 0
    for edge in edges:
        if edge.source and edge.target and edge.relation and edge.evidence:
            kept.append(edge)
        else:
            dropped += 1

    if dropped:
        log.info('%s: dropped %d edge(s) with no evidence', ref, dropped)

    return kept
