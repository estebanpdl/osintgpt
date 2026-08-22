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
from typing import Iterator, List, Tuple

# Large enough to hold an argument, small enough that a hit points somewhere
# specific. Characters rather than tokens: the boundary has to be decided
# before a tokenizer is chosen, and it must not differ per embedding model.
MAX_CHARS = 1500

_HEADING = re.compile(r'^#{1,6}\s')

# Paragraph breaks tolerant of trailing whitespace and Windows line endings,
# which real documents carry and a bare '\n\n' split misses.
_PARAGRAPH = re.compile(r'\r?\n[ \t]*\r?\n')

# A table row, and the rule beneath a header. Anything made only of pipes,
# dashes, colons and spaces is the rule rather than data.
_TABLE_ROW = re.compile(r'^\s*\|')
_TABLE_RULE = re.compile(r'^\s*\|[\s:|-]*-[\s:|-]*$')

# Sentence terminators across the scripts an OSINT corpus arrives in. Scripts
# that space their words need a break after the mark, or 3.14 becomes a
# boundary. Full-width marks never carry one, so requiring it would leave text
# in those scripts with no sentence boundaries at all.
_SPACED_END = r'[.!?۔।؟…](?=[\s"\'”’)\]]|$)'
_FULL_WIDTH_END = r'[。！？]'
_SENTENCE_END = re.compile(f'(?:{_SPACED_END})|(?:{_FULL_WIDTH_END})')

TABLE = 'table'
PROSE = 'prose'


# split a document into chunks
def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    '''
    Split text into retrieval chunks.

    Sections start at markdown headings; a section over the cap is re-split on
    paragraph and table boundaries, and text with no usable boundary is cut at
    a sentence end so that a document with no structure at all still chunks.

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

    def flush() -> None:
        nonlocal window
        if window:
            chunks.append(window)
            window = ''

    for kind, block in _blocks(section):
        if window and len(window) + len(block) + 2 > max_chars:
            flush()

        if len(block) <= max_chars:
            window = f'{window}\n\n{block}' if window else block
            continue

        flush()
        if kind == TABLE:
            chunks.extend(_split_table(block, max_chars))
        else:
            chunks.extend(_hard_split(block, max_chars))

    flush()

    return chunks


# walk a section as tables and paragraphs
def _blocks(section: str) -> Iterator[Tuple[str, str]]:
    '''
    Yield (kind, text) for each block, keeping a table's rows together.

    A table is one block because its rows mean nothing without the header that
    names their columns; splitting between them produces chunks that read as
    data and carry none.
    '''
    lines = section.splitlines()
    index = 0

    while index < len(lines):
        if _TABLE_ROW.match(lines[index]):
            start = index
            while index < len(lines) and _TABLE_ROW.match(lines[index]):
                index += 1
            table = '\n'.join(lines[start:index]).strip()
            if table:
                yield TABLE, table
            continue

        start = index
        while index < len(lines) and not _TABLE_ROW.match(lines[index]):
            index += 1
        prose = '\n'.join(lines[start:index])
        for paragraph in _PARAGRAPH.split(prose):
            paragraph = paragraph.strip()
            if paragraph:
                yield PROSE, paragraph


# split a table too large for one chunk
def _split_table(table: str, max_chars: int) -> List[str]:
    '''
    Break a table into row groups, repeating the header in each.

    The repetition is the point: a group of rows without its header is a chunk
    nobody can read, so the header costs a few characters in every piece and
    buys every piece its meaning.
    '''
    lines = table.splitlines()

    header: List[str] = []
    if len(lines) > 1 and _TABLE_RULE.match(lines[1]):
        header = lines[:2]
        lines = lines[2:]

    preamble = '\n'.join(header)
    room = max_chars - len(preamble) - 1 if header else max_chars

    # A header that leaves no room for rows is not a header worth repeating.
    if room < 80:
        preamble, room = '', max_chars
        lines = header + lines
        header = []

    pieces: List[str] = []
    group: List[str] = []
    length = 0

    for row in lines:
        if group and length + len(row) + 1 > room:
            pieces.append(_join_table(preamble, group))
            group, length = [], 0
        # A single row wider than the cap is cut rather than dropped.
        if len(row) > room:
            if group:
                pieces.append(_join_table(preamble, group))
                group, length = [], 0
            for piece in _hard_split(row, room):
                pieces.append(_join_table(preamble, [piece]))
            continue
        group.append(row)
        length += len(row) + 1

    if group:
        pieces.append(_join_table(preamble, group))

    return pieces


def _join_table(preamble: str, rows: List[str]) -> str:
    body = '\n'.join(rows)

    return f'{preamble}\n{body}' if preamble else body


# last resort for text with no usable boundary
def _hard_split(paragraph: str, max_chars: int) -> List[str]:
    '''
    Cut at the last sentence end before the cap, then the last whitespace,
    then the cap itself — so a chunk reads as a finished thought where the text
    allows one, and never ends mid-word where it does not.

    Falls through to a blunt cut for text with no whitespace at all, which is
    what a long identifier or a script that does not space its words looks
    like.
    '''
    pieces: List[str] = []
    remaining = paragraph

    while len(remaining) > max_chars:
        window = remaining[:max_chars + 1]

        cut = 0
        ends = list(_SENTENCE_END.finditer(window))
        if ends:
            cut = ends[-1].end()
        if cut <= 0:
            cut = window.rfind(' ')
        if cut <= 0:
            cut = max_chars

        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].lstrip()

    if remaining:
        pieces.append(remaining)

    return pieces
