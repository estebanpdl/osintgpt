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
from typing import Dict, List, Tuple, Union

from .documents import Document

# Extensions read as prose without any configuration.
TEXT_SUFFIXES = {'.txt', '.md', '.markdown', '.rst', '.log'}
HTML_SUFFIXES = {'.html', '.htm'}

# Formats that conventionally open with a metadata block.
FRONTMATTER_SUFFIXES = {'.md', '.markdown'}

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


# read a prose file as one document
def load_text(path: Union[str, Path]) -> List[Document]:
    '''
    Read a whole file as a single document.

    Args:
        path (Union[str, Path]): File to read.

    Returns:
        List[Document]: One document, or none when the file holds no text.
    '''
    path = Path(path)
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

    return [Document(ref=path.as_posix(), text=text, metadata=metadata)]
