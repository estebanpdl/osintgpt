# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: loaders.py
# Description: One entry point over every format, so nothing downstream has to
#   know whether a document came from prose or from row 40,127 of a sheet.
# =================================================================================

# import submodules
from pathlib import Path

# type hints
from typing import List, Optional, Union

from .documents import Document, FieldMapping
from .fallback import FALLBACK_SUFFIXES, can_convert, convert
from .images import IMAGE_SUFFIXES, is_image
from .office import extract_docx
from .pdf import Transcriber, extract_pdf
from .tabular import load_records
from .text import HTML_SUFFIXES, TEXT_SUFFIXES, load_text

# Formats carrying many fields, only some of which are content. These need a
# mapping; prose does not.
STRUCTURED_SUFFIXES = {'.csv', '.xlsx', '.xlsm', '.json', '.jsonl', '.ndjson'}

# Formats whose text has to be recovered before it can be read: a PDF stores
# glyphs and positions, a .docx stores XML. Both become markdown first.
DOCUMENT_SUFFIXES = {'.pdf', '.docx'}

SUPPORTED_SUFFIXES = (
    TEXT_SUFFIXES | HTML_SUFFIXES | STRUCTURED_SUFFIXES | DOCUMENT_SUFFIXES
)

# What osintgpt can read at all, including formats only the optional converter
# reaches and images, which are embedded rather than read. Kept apart from
# SUPPORTED_SUFFIXES so a dry run can distinguish "read by a reader chosen for
# it" from "converted as a last resort" from "not text at all".
READABLE_SUFFIXES = SUPPORTED_SUFFIXES | FALLBACK_SUFFIXES | IMAGE_SUFFIXES


# does osintgpt read this file
def is_supported(path: Union[str, Path]) -> bool:
    '''
    Args:
        path (Union[str, Path]): File to check.

    Returns:
        bool: True when a loader exists for the extension.
    '''
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


# does this file need a field mapping
def needs_mapping(path: Union[str, Path]) -> bool:
    '''
    Args:
        path (Union[str, Path]): File to check.

    Returns:
        bool: True for formats whose content fields must be named.
    '''
    return Path(path).suffix.lower() in STRUCTURED_SUFFIXES


# read a file as documents
def load_documents(
    path: Union[str, Path],
    mapping: Optional[FieldMapping] = None,
    transcriber: Optional[Transcriber] = None
) -> List[Document]:
    '''
    Read any supported file into documents.

    Prose yields one document; a structured file yields one per record, which
    keeps chunking and embedding identical for both.

    Args:
        path (Union[str, Path]): File to read.
        mapping (FieldMapping, optional): Required for structured formats.
        transcriber (Transcriber, optional): Reads scanned PDF pages. Without             one, those pages degrade to a named gap rather than failing.

    Raises:
        ValueError: If the extension has no loader.
        UnmappedSourceError: If a structured file arrives without a mapping.

    Returns:
        List[Document]: The documents, in file order.
    '''
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in STRUCTURED_SUFFIXES:
        return list(load_records(path, mapping or FieldMapping()))

    if suffix == '.pdf':
        extraction = extract_pdf(path, transcriber)
        text = extraction.markdown.strip()

        return [Document(ref=path.as_posix(), text=text)] if text else []

    if suffix == '.docx':
        text = extract_docx(path).strip()

        return [Document(ref=path.as_posix(), text=text)] if text else []

    if suffix in TEXT_SUFFIXES | HTML_SUFFIXES:
        return load_text(path, mapping)

    if is_image(path):
        # An image carries no text to load. It is corpus, and it is indexed
        # through the embedding provider rather than through a reader, so
        # this path has nothing to return rather than nothing to say.
        raise ValueError(
            f'{path.name} is an image; images are embedded directly and have '
            'no text to load'
        )

    # Last resort, and only for formats with no reader of their own. Where a
    # reader exists it is better: it was chosen for that format, and a general
    # converter would undo decisions taken for a reason.
    if can_convert(path):
        text = convert(path)

        return [Document(ref=path.as_posix(), text=text)] if text else []

    readable = ', '.join(sorted(READABLE_SUFFIXES))

    raise ValueError(
        f'{path.name}: osintgpt has no loader for {suffix!r}; '
        f'readable: {readable}'
    )
