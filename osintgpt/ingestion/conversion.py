# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: conversion.py
# Description: One file as markdown, and which reader produced it. The same
#   readers ingestion uses, so what is shown is what would be indexed.
# =================================================================================

# import submodules
from dataclasses import dataclass
from pathlib import Path

# type hints
from typing import List, Optional, Union

from .documents import Document, FieldMapping
from .fallback import FALLBACK_SUFFIXES
from .loaders import STRUCTURED_SUFFIXES, load_documents
from .pdf import Transcriber, extract_pdf
from .text import HTML_SUFFIXES, TEXT_SUFFIXES

PDF = 'pdf'
DOCX = 'docx'
TEXT = 'text'
TABULAR = 'tabular'
MARKITDOWN = 'markitdown'


# Conversion class
@dataclass(frozen=True)
class Conversion:
    '''
    One file as markdown, with what it took to read it.
    '''
    path: Path
    reader: str
    markdown: str
    documents: int = 0
    pages: int = 0
    # Pages a vision model read, and pages nothing could read. Both are zero
    # for every format but PDF.
    transcribed_pages: int = 0
    empty_pages: int = 0


# which reader handles this file
def reader_for(path: Union[str, Path]) -> str:
    '''
    Args:
        path (Union[str, Path]): File to check.

    Returns:
        str: Name of the reader, or an empty string when nothing reads it.
    '''
    suffix = Path(path).suffix.lower()

    if suffix == '.pdf':
        return PDF
    if suffix == '.docx':
        return DOCX
    if suffix in STRUCTURED_SUFFIXES:
        return TABULAR
    if suffix in TEXT_SUFFIXES | HTML_SUFFIXES:
        return TEXT
    if suffix in FALLBACK_SUFFIXES:
        return MARKITDOWN

    return ''


# read one file as markdown
def to_markdown(
    path: Union[str, Path],
    mapping: Optional[FieldMapping] = None,
    transcriber: Optional[Transcriber] = None
) -> Conversion:
    '''
    Convert a file to markdown without indexing or embedding anything.

    Without a transcriber nothing is sent anywhere: a scanned page becomes the
    same named gap ingestion would leave, which is free and says so.

    Args:
        path (Union[str, Path]): File to convert.
        mapping (FieldMapping, optional): Required for structured formats.
        transcriber (Transcriber, optional): Reads scanned PDF pages.

    Raises:
        ValueError: If the file is an image or has no reader.
        UnmappedSourceError: If a structured file arrives without a mapping.

    Returns:
        Conversion: The markdown and how it was produced.
    '''
    path = Path(path)
    reader = reader_for(path)

    if reader == PDF:
        extraction = extract_pdf(path, transcriber)
        markdown = extraction.markdown.strip()

        return Conversion(
            path=path,
            reader=PDF,
            markdown=markdown,
            documents=1 if markdown else 0,
            pages=len(extraction.pages),
            transcribed_pages=extraction.transcribed_pages,
            empty_pages=extraction.empty_pages
        )

    # Everything else goes through the loader rather than its reader directly,
    # so an image and an unreadable extension are refused in one place with
    # the message ingestion already uses.
    documents = load_documents(path, mapping)
    markdown = (
        _sections(documents) if reader == TABULAR
        else '\n\n'.join(document.text.strip() for document in documents)
    )

    return Conversion(
        path=path,
        reader=reader,
        markdown=markdown.strip(),
        documents=len(documents)
    )


def _sections(documents: List[Document]) -> str:
    '''
    A structured file as one section per record.

    A record is a document of its own once indexed, so its ref and the fields
    retrieval filters on head it: what is read here is the unit the corpus
    will hold, rather than a table flattened back together.
    '''
    blocks: List[str] = []

    for document in documents:
        heading = [f'## {document.ref}']
        if document.timestamp:
            heading.append(f'- timestamp: {document.timestamp}')
        if document.author:
            heading.append(f'- author: {document.author}')

        blocks.append('\n'.join(heading) + '\n\n' + document.text.strip())

    return '\n\n'.join(blocks)
