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

# import submodules
from html.parser import HTMLParser
from pathlib import Path

# type hints
from typing import List, Union

from .documents import Document

# Extensions read as prose without any configuration.
TEXT_SUFFIXES = {'.txt', '.md', '.markdown', '.rst', '.log'}
HTML_SUFFIXES = {'.html', '.htm'}

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
    raw = path.read_text(encoding='utf-8', errors='replace')

    if path.suffix.lower() in HTML_SUFFIXES:
        extractor = _TextExtractor()
        extractor.feed(raw)
        raw = extractor.text

    text = raw.strip()
    if not text:
        return []

    return [Document(ref=path.as_posix(), text=text)]
