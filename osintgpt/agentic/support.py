# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: support.py
# Description: What guards and shapes the tools' inputs and outputs — the
#   project boundary and the payloads a model is shown.
# =================================================================================

# import modules
import logging

# import submodules
from pathlib import Path

# type hints
from typing import Any, Dict, Optional

log = logging.getLogger('osintgpt.agentic')

# How much of a passage a snippet carries. Enough to judge relevance, short
# enough that twenty of them still leave room to think.
SNIPPET_CHARS = 700


def _resolve(context, ref: str) -> Optional[Path]:
    '''
    A ref to a file the corpus covers, or None.

    The boundary is the registered corpus, not the project directory. Material
    normally lives outside the project — a case folder on another drive, a
    shared collection — and the index stores an absolute ref for it, so
    confining refs to the project root refuses every document such a project
    has. What makes a path readable is that a source registered it.

    That is a narrower test than a prefix check, not a looser one: membership
    of the set `index_project` walks. A ref climbing out with `..`, an
    absolute path nobody registered, and a symlink pointing away all fail it,
    because none of them is a file the corpus covers. Resolving first is what
    makes the comparison real rather than one a `..` walks straight through.
    '''
    from osintgpt.ingestion.source_view import reference_parts

    try:
        candidate, _ = reference_parts(ref, context.root, context.covered)
    except (OSError, ValueError):
        return None

    if candidate is None:
        log.warning('refused a ref the corpus does not cover: %r', ref)

        return None

    return candidate if candidate.is_file() else None


def _read(context, path: Path, ref: str, source_version=''):
    from osintgpt.ingestion.source_view import read_source
    return read_source(context.project, path, ref, source_version)


def _passage(result) -> Dict[str, Any]:
    chunk = result.chunk

    return {
        'citation': chunk.citation,
        'ref': chunk.ref,
        'text': result.text[:SNIPPET_CHARS],
        'score': round(result.score, 4),
        'document_ref': chunk.document_ref or chunk.ref,
        **({'provenance': chunk.metadata['_osintgpt_provenance']}
           if '_osintgpt_provenance' in chunk.metadata else {}),
        **({'timestamp': chunk.timestamp} if chunk.timestamp else {}),
        **({'author': chunk.author} if chunk.author else {}),
        # Which terms put this chunk in the result, so the model reads why it
        # ranked rather than inferring it from position.
        **({'terms': list(result.terms)} if getattr(result, 'terms', ()) else {})
    }


def _clamp(value, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low

    return max(low, min(number, high))
