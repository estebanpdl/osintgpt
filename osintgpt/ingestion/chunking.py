# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: chunking.py
# Description: Splitting a document into retrieval-sized pieces. A pure
#   function over text, which is what makes it cheap to judge against a corpus.
# =================================================================================

# import modules
import re

# type hints
from typing import List

# Large enough to hold an argument, small enough that a hit points somewhere
# specific. Characters rather than tokens: the boundary has to be decided
# before a tokenizer is chosen, and it must not differ per embedding model.
MAX_CHARS = 1500

_HEADING = re.compile(r'^#{1,6}\s')

# Paragraph breaks tolerant of trailing whitespace and Windows line endings,
# which real documents carry and a bare '\n\n' split misses.
_PARAGRAPH = re.compile(r'\r?\n[ \t]*\r?\n')


# split a document into chunks
def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    '''
    Split text into retrieval chunks.

    Sections start at markdown headings; a section over the cap is re-split on
    paragraph boundaries, and a paragraph over the cap is cut to length so that
    a document with no structure at all still chunks.

    Splitting on structure rather than a sliding window means a chunk tends to
    be about one thing, which is what makes a hit interpretable.

    Args:
        text (str): Document text.
        max_chars (int): Ceiling on a chunk, in characters.

    Returns:
        List[str]: Chunks in document order, stripped, never empty strings.
    '''
    chunks: List[str] = []
    for section in _sections(text):
        if len(section) <= max_chars:
            chunks.append(section)
            continue
        chunks.extend(_split_section(section, max_chars))

    return chunks


# split on markdown headings
def _sections(text: str) -> List[str]:
    sections: List[str] = []
    current: List[str] = []

    for line in (text or '').splitlines():
        # A heading opens a section rather than joining the previous one, so a
        # chunk carries the heading that introduces it.
        if _HEADING.match(line) and current:
            sections.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append('\n'.join(current))

    return [section for section in (s.strip() for s in sections) if section]


# re-split an oversized section
def _split_section(section: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    window = ''

    for paragraph in _PARAGRAPH.split(section):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if window and len(window) + len(paragraph) + 2 > max_chars:
            chunks.append(window)
            window = ''

        if len(paragraph) > max_chars:
            if window:
                chunks.append(window)
                window = ''
            chunks.extend(_hard_split(paragraph, max_chars))
            continue

        window = f'{window}\n\n{paragraph}' if window else paragraph

    if window:
        chunks.append(window)

    return chunks


# last resort for text with no usable boundary
def _hard_split(paragraph: str, max_chars: int) -> List[str]:
    '''
    Cut at the last whitespace before the cap so a chunk rarely ends mid-word.

    Falls back to cutting at the cap for text with no whitespace at all, which
    is what a long identifier or a language without spaces looks like.
    '''
    pieces: List[str] = []
    remaining = paragraph

    while len(remaining) > max_chars:
        cut = remaining.rfind(' ', 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].lstrip()

    if remaining:
        pieces.append(remaining)

    return pieces
