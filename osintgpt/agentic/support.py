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
    try:
        candidate = (context.root / ref).resolve()
    except (OSError, ValueError):
        return None

    if candidate not in context.covered:
        log.warning('refused a ref the corpus does not cover: %r', ref)

        return None

    return candidate if candidate.is_file() else None


def _read(context, path: Path, ref: str) -> str:
    '''
    The document's text, through the same loaders that indexed it, so a PDF
    reads as the markdown the index holds rather than as bytes.
    '''
    from osintgpt.ingestion import load_documents

    documents = load_documents(
        path, context.corpus.mapping_for(path, context.root)
    )

    return '\n\n'.join(d.text for d in documents if d.text)


def _passage(result) -> Dict[str, Any]:
    chunk = result.chunk

    return {
        'citation': chunk.citation,
        'ref': chunk.ref,
        'text': result.text[:SNIPPET_CHARS],
        'score': round(result.score, 4),
        **({'timestamp': chunk.timestamp} if chunk.timestamp else {}),
        **({'author': chunk.author} if chunk.author else {}),
        # Which terms put this chunk in the result, so the model reads why it
        # ranked rather than inferring it from position.
        **({'terms': list(result.terms)} if getattr(result, 'terms', ()) else {})
    }


def _claim(edge) -> Dict[str, str]:
    return {
        'source': edge.source,
        'relation': edge.relation,
        'target': edge.target,
        'ref': edge.ref,
        'evidence': edge.evidence
    }


def _clamp(value, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low

    return max(low, min(number, high))
