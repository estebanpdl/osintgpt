# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: text.py
# Description: Formats that are already prose. The whole file is one document,
#   so nothing has to be chosen before it can be indexed.
# =================================================================================

# import modules
import re

# import submodules
from html.parser import HTMLParser
from pathlib import Path

# type hints
from typing import Dict, List, Optional, Tuple, Union

from .documents import Document, FieldMapping

# Extensions read as prose without any configuration.
TEXT_SUFFIXES = {'.txt', '.md', '.markdown', '.rst', '.log'}
HTML_SUFFIXES = {'.html', '.htm'}

# Formats that conventionally open with a metadata block.
FRONTMATTER_SUFFIXES = {'.md', '.markdown'}

# Frontmatter keys read as a document's own timestamp and author when a source
# names none. Conventional rather than exhaustive: a project whose documents
# use other keys names them explicitly, and nothing is inferred beyond this
# list — a guessed timestamp is worse than an absent one, because a filter
# built on it fails silently.
TIMESTAMP_KEYS = ('date', 'created', 'created_at', 'published', 'published_at')
AUTHOR_KEYS = ('author', 'authors', 'by', 'prepared_by')

# A fenced block at the very top of a file. Nothing else in the document is
# treated this way: a rule further down is a rule.
_FRONTMATTER = re.compile(
    r'^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n', re.DOTALL
)

# Flat 'key: value' lines. Indented lines and list items belong to a nested
# value this deliberately does not try to reconstruct.
_FRONTMATTER_FIELD = re.compile(r'^([A-Za-z0-9_.-]+):[ \t]*(.*)$')

# Elements whose text is markup machinery rather than content.
_SKIPPED = {'script', 'style', 'head', 'meta', 'link', 'noscript'}

# Elements that end a line of prose, so extracted text keeps its paragraphing
# instead of running together into one block.
_BREAKS = {
    'p', 'div', 'br', 'li', 'tr', 'section', 'article', 'header', 'footer',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'pre'
}


# _TextExtractor class
class _TextExtractor(HTMLParser):
    '''
    Pulls readable text out of HTML using the standard library, which keeps a
    parser dependency out of the core install.
    '''
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIPPED:
            self._skipping += 1
        elif tag in _BREAKS:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in _SKIPPED and self._skipping:
            self._skipping -= 1
        elif tag in _BREAKS:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self._skipping and data.strip():
            self.parts.append(data.strip())

    @property
    def text(self) -> str:
        joined = ' '.join(self.parts)

        # Collapse the runs of blank lines the break tags produced, without
        # touching the paragraph breaks that carry structure.
        lines = [line.strip() for line in joined.split('\n')]

        return '\n\n'.join(line for line in lines if line)


# separate a leading metadata block from the body
def split_frontmatter(raw: str) -> Tuple[Dict[str, str], str]:
    '''
    Take a leading '---' block off a document and read it as metadata.

    Frontmatter describes a document rather than saying anything; embedding it
    puts field names into the vector, and the fields it carries — type,
    version, who prepared it — are exactly what belongs in a citation instead.

    Only flat 'key: value' lines are read. A block that cannot be read is
    still removed from the body, because it is metadata either way.

    Args:
        raw (str): File contents.

    Returns:
        Tuple[Dict[str, str], str]: The fields found, and the remaining body.
    '''
    match = _FRONTMATTER.match(raw)
    if match is None:
        return {}, raw

    fields: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if line[:1] in (' ', '\t', '-') or not line.strip():
            continue
        field = _FRONTMATTER_FIELD.match(line)
        if field is None:
            continue
        key = field.group(1)
        value = field.group(2).strip().strip('"').strip("'")
        if value:
            fields[key] = value

    return fields, raw[match.end():]


# the value of the first key a mapping has
def _first_of(fields: Dict[str, str], keys) -> str:
    for key in keys:
        value = str(fields.get(key, '')).strip()
        if value:
            return value

    return ''


# read a prose file as one document
def load_text(
    path: Union[str, Path], mapping: Optional[FieldMapping] = None
) -> List[Document]:
    '''
    Read a whole file as a single document.

    A prose document has no columns to map, but it can still carry when it was
    written and who wrote it. Those come from frontmatter — named by the source
    where it says so, and otherwise from the conventional keys — because
    retrieval filters on them, and a question about April should not land on a
    mapped spreadsheet while silently missing every markdown file beside it.

    Args:
        path (Union[str, Path]): File to read.
        mapping (FieldMapping, optional): Names which frontmatter fields are             the timestamp and author, when the conventional keys are wrong.

    Returns:
        List[Document]: One document, or none when the file holds no text.
    '''
    path = Path(path)
    mapping = mapping or FieldMapping()
    # utf-8-sig rather than utf-8: an editor's byte order mark would
    # otherwise sit in front of the first character, which stops a
    # leading metadata block from being recognised and rides into the
    # first chunk. Harmless on a file that has none.
    raw = path.read_text(encoding='utf-8-sig', errors='replace')
    metadata: Dict[str, str] = {}

    if path.suffix.lower() in FRONTMATTER_SUFFIXES:
        metadata, raw = split_frontmatter(raw)

    if path.suffix.lower() in HTML_SUFFIXES:
        extractor = _TextExtractor()
        extractor.feed(raw)
        raw = extractor.text

    text = raw.strip()
    if not text:
        return []

    timestamp = (
        str(metadata.get(mapping.timestamp, '')).strip() if mapping.timestamp
        else _first_of(metadata, TIMESTAMP_KEYS)
    )
    author = (
        str(metadata.get(mapping.author, '')).strip() if mapping.author
        else _first_of(metadata, AUTHOR_KEYS)
    )

    return [Document(
        ref=path.as_posix(),
        text=text,
        metadata=metadata,
        timestamp=timestamp,
        author=author
    )]
