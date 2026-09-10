# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: chunking.py
# Description: Splitting a document into units of meaning. A pure function over
#   text, which is what makes it cheap to judge against a corpus.
# =================================================================================

# import modules
import re

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Iterator, List, Tuple

# Large enough to hold an argument, small enough that a hit points somewhere
# specific. Characters rather than tokens: the boundary has to be decided
# before a tokenizer is chosen, and it must not differ per embedding model.
MAX_CHARS = 1500

# Separates the headings a chunk sits under. Rare enough in prose that its
# presence marks the line as osintgpt's rather than the document's.
BREADCRUMB_SEPARATOR = ' › '

# A path that would leave less than this for content is not worth its cost, so
# the chunk goes out without one rather than being crowded by its own label.
MIN_CONTENT_ROOM = 300

_HEADING = re.compile(r'^(#{1,6})\s')

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


# Chunk class
@dataclass(frozen=True)
class Chunk:
    '''
    One retrieval unit and the section path it sits under. The path stays a
    field rather than only a line of prose, so a citation and a store can read
    it without parsing the text back.
    '''
    text: str
    path: str = ''

    @property
    def rendered(self) -> str:
        '''The chunk as it is embedded, path included.'''
        return f'{self.path}\n\n{self.text}' if self.path else self.text

    def __len__(self) -> int:
        return len(self.rendered)


# _Section class
@dataclass
class _Section:
    """
    One heading and everything beneath it, addressed by line range so the
    source is never reconstructed: a chunk is a slice of the document.
    """
    level: int
    heading: str
    start: int
    end: int
    children: List["_Section"] = field(default_factory=list)


# split a document into chunks
def chunk_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    Split text into retrieval chunks.

    Where a document has headings, the largest section that fits becomes one
    chunk and carries the path of headings above it, so a passage arrives with
    the context that frames it. Where it has none -- a transcript, a scraped
    page, extracted PDF text -- none of that applies and the text falls through
    to tables, paragraphs and sentence ends exactly as it would otherwise.

    Args:
        text (str): Document text.
        max_chars (int): Ceiling on a chunk, in characters, path included.

    Returns:
        List[str]: Chunks in document order, stripped, never empty strings.
    """
    return [chunk.rendered for chunk in chunk_document(text, max_chars)]


# split a document into chunks, keeping each one's path
def chunk_document(text: str, max_chars: int = MAX_CHARS) -> List['Chunk']:
    '''
    As `chunk_text`, but each chunk keeps its section path as a separate field
    rather than only as a line of prose in front of it.

    Args:
        text (str): Document text.
        max_chars (int): Ceiling on a chunk, in characters, path included.

    Returns:
        List[Chunk]: Chunks in document order.
    '''
    lines = (text or '').splitlines()
    chunks: List[Chunk] = []
    _emit(_outline(lines), lines, (), max_chars, chunks)

    return chunks


# build the heading tree
def _outline(lines: List[str]) -> _Section:
    """
    A tree of sections by heading level.

    A document that skips a level or uses only one is ordinary rather than
    malformed: a heading attaches to the nearest shallower one. A document with
    no headings yields a single root covering everything, which is what makes
    the tree free for text that has no structure to find.
    """
    root = _Section(level=0, heading="", start=0, end=len(lines))
    stack = [root]

    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is None:
            continue

        level = len(match.group(1))
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop().end = index

        section = _Section(
            level=level, heading=line.strip(), start=index, end=len(lines)
        )
        stack[-1].children.append(section)
        stack.append(section)

    while len(stack) > 1:
        stack.pop().end = len(lines)

    return root


# walk the tree, taking the largest section that fits
def _emit(
    section: _Section,
    lines: List[str],
    ancestors: Tuple[str, ...],
    max_chars: int,
    chunks: List[Chunk]
) -> None:
    text = '\n'.join(lines[section.start:section.end]).strip()
    if not text:
        return

    path = _breadcrumb(ancestors, max_chars)
    room = max_chars - len(path) - 2 if path else max_chars

    if len(text) <= room:
        chunks.append(Chunk(text=text, path=path))
        return

    # Too large to stand alone: its own opening becomes chunks, then each
    # child is tried in turn.
    below = ancestors + ((section.heading,) if section.heading else ())
    deeper = _breadcrumb(below, max_chars)

    own_end = section.children[0].start if section.children else section.end
    own = '\n'.join(lines[section.start:own_end]).strip()
    if own:
        # Only the first piece opens with the heading itself. The rest take it
        # into their path, or a section split into five leaves four chunks
        # with nothing saying where they sit. Room is measured against the
        # longer path so every piece fits either way.
        own_room = max_chars - len(deeper) - 2 if deeper else max_chars
        pieces = [own] if len(own) <= own_room else _split_section(own, own_room)
        chunks.extend(
            Chunk(text=piece, path=path if index == 0 else deeper)
            for index, piece in enumerate(pieces)
        )

    for child in section.children:
        _emit(child, lines, below, max_chars, chunks)


# the path of headings a chunk sits under
def _breadcrumb(ancestors: Tuple[str, ...], max_chars: int) -> str:
    names = [
        heading.lstrip("#").strip()
        for heading in ancestors
        if heading.lstrip("#").strip()
    ]
    if not names:
        return ""

    path = BREADCRUMB_SEPARATOR.join(names)

    # A path is context, not content. One that would crowd out the passage it
    # describes is dropped rather than trimmed into something misleading.
    return path if max_chars - len(path) >= MIN_CONTENT_ROOM else ""


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
