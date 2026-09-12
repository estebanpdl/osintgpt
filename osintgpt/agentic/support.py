# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: support.py
# Description: What guards and shapes the tools' inputs and outputs — the
#   project boundary, the time filter, and the payloads a model is shown.
# =================================================================================

# import modules
import logging

# import submodules
from datetime import datetime, timedelta
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


def _within_days(results, days):
    """
    Split results into those recent enough and count what could not be dated.

    A document whose timestamp cannot be read is **kept**, not dropped.
    Hiding material because its date was unparseable is a worse failure than
    a filter that is slightly loose: the analyst never learns the document
    exists. The count travels with the result so the model can say the filter
    was partial.
    """
    if not days:
        return list(results), 0

    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=max(int(days), 0))
    kept, undated = [], 0

    for result in results:
        moment = _moment(result.chunk.timestamp)
        if moment is None:
            undated += 1
            kept.append(result)
        elif moment >= cutoff:
            kept.append(result)

    return kept, undated


def _moment(timestamp: str):
    """
    A stored timestamp as a datetime, or None.

    ISO forms only, and deliberately no regex: date formats are
    language-bound, and guessing at one would quietly mis-order a corpus
    written in a convention nobody here anticipated.
    """
    text = (timestamp or '').strip()
    if not text:
        return None

    from datetime import datetime

    for candidate in (text, text[:19], text[:10]):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue

    return None


def _dating_note(days, undated: int) -> Dict[str, Any]:
    """
    Says when a time filter could not be applied to everything it saw.
    """
    if not days or not undated:
        return {}

    return {
        'filtered_days': days,
        'undated_documents': undated,
        'note': f'{undated} passage(s) had no readable timestamp and were '
                'kept rather than hidden, so this filter is partial.'
    }


def _passage(result) -> Dict[str, Any]:
    chunk = result.chunk

    return {
        'citation': chunk.citation,
        'ref': chunk.ref,
        'text': result.text[:SNIPPET_CHARS],
        'score': round(result.score, 4),
        **({'timestamp': chunk.timestamp} if chunk.timestamp else {}),
        **({'author': chunk.author} if chunk.author else {})
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
