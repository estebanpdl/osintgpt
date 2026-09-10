# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: pdf.py
# Description: PDF to markdown at ingestion. Born-digital pages give up their
#   text directly; pages that yield almost none are images of text, and only
#   those are worth sending to a vision model.
# =================================================================================

# import modules
import logging
import re

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Callable, List, Optional, Union

from .chunking import _SENTENCE_END

log = logging.getLogger('osintgpt.ingestion')

# Below this many characters a page is an image of text rather than text: a
# scan, a slide, a figure. The number is a signal, not a measurement — a page
# holding only a caption is legitimately this short, and transcribing it costs
# a call rather than losing anything.
MIN_PAGE_CHARS = 40

# ~144 dpi. Readable by a vision model without sending an enormous payload for
# every page.
RENDER_SCALE = 2.0

# Marks a page whose text could not be recovered. Kept in the output on
# purpose: a document with a gap that says so is auditable, and one that
# silently omits a page is not.
PLACEHOLDER = '[Page {page}: no extractable text — scanned or image-only]'
FAILED = '[Page {page}: transcription failed]'

# Takes rendered page bytes and returns transcribed markdown. Injected rather
# than constructed here, so extraction is testable without a provider and the
# ingestion model stays the caller's choice.
# Takes the rendered page and its 1-based number. The number is passed rather
# than counted by the transcriber, because only the extractor knows which
# pages were skipped as already readable.
Transcriber = Callable[[bytes, int], str]

_PDF_EXTRACTION_NOISE = re.compile(
    r'[\x0c\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]'
)
_LAYOUT_PADDING = re.compile(r'[ \t]+')


# PageExtraction class
@dataclass(frozen=True)
class PageExtraction:
    '''
    One page's text and where it came from.
    '''
    number: int
    text: str
    # True when the text came from a vision model rather than the PDF itself.
    transcribed: bool = False
    # True when the page had no recoverable text at all.
    empty: bool = False


# PdfExtraction class
@dataclass(frozen=True)
class PdfExtraction:
    '''
    A whole PDF as markdown, with what it cost to get there.
    '''
    pages: List[PageExtraction] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return join_page_texts([page.text for page in self.pages])

    @property
    def transcribed_pages(self) -> int:
        return sum(1 for page in self.pages if page.transcribed)

    @property
    def empty_pages(self) -> int:
        return sum(1 for page in self.pages if page.empty)

    @property
    def needs_vision(self) -> int:
        '''
        Pages a vision model would be asked to read.

        Reported so a dry run can say what ingestion will cost before anyone
        pays for it: this is the expensive half of reading a PDF.
        '''
        return sum(1 for page in self.pages if page.transcribed or page.empty)


def join_page_texts(page_texts: List[str]) -> str:
    '''Join PDF pages without inventing paragraph breaks inside sentences.'''
    pages = [text for text in page_texts if text]
    if not pages:
        return ''

    joined = pages[0]
    for previous, current in zip(pages, pages[1:]):
        separator = ' ' if _continues_sentence(previous, current) else '\n\n'
        joined += separator + current

    return joined


def _continues_sentence(previous: str, current: str) -> bool:
    previous = previous.rstrip()
    current = current.lstrip()
    if not previous or not current:
        return False

    ends_sentence = any(
        match.end() == len(previous)
        for match in _SENTENCE_END.finditer(previous)
    )
    first = current[0]
    starts_continuation = (
        first.islower()
        or first.isalpha() and not first.isupper() and not first.istitle()
    )

    return not ends_sentence and starts_continuation


def _clean_extracted_text(text: str) -> str:
    collapsed = _LAYOUT_PADDING.sub(' ', text)
    return _PDF_EXTRACTION_NOISE.sub('', collapsed).strip()


def _extract_page_text(page) -> str:
    try:
        return page.extract_text(extraction_mode='layout') or ''
    except TypeError:
        return page.extract_text() or ''
    except KeyError as error:
        # Layout mode indexes /Contents directly, which blank pages may omit.
        if error.args != ('/Contents',):
            raise

        return page.extract_text() or ''


# read the text a PDF carries directly
def extract_page_texts(path: Union[str, Path]) -> List[str]:
    '''
    Per-page text as the PDF itself stores it.

    Args:
        path (Union[str, Path]): PDF to read.

    Raises:
        ImportError: If pypdf is not installed.

    Returns:
        List[str]: One string per page, stripped, empty where the page holds \
            no extractable text.
    '''
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError(
            "reading PDFs needs the 'pypdf' package, which osintgpt "
            'requires: reinstall with pip install --force-reinstall osintgpt'
        ) from error

    reader = PdfReader(str(path))

    return [
        _clean_extracted_text(_extract_page_text(page))
        for page in reader.pages
    ]


# rasterize one page
def render_page(path: Union[str, Path], index: int) -> bytes:
    '''
    Render a page to PNG bytes for a vision model.

    Args:
        path (Union[str, Path]): PDF to read.
        index (int): Zero-based page index.

    Raises:
        ImportError: If pypdfium2 is not installed.

    Returns:
        bytes: The page as a PNG.
    '''
    try:
        import pypdfium2
        # pypdfium2 renders to a raw bitmap and leaves encoding to Pillow, so
        # a missing Pillow surfaces here rather than as a failed page.
        from PIL import Image  # noqa: F401
    except ImportError as error:
        raise ImportError(
            'transcribing scanned pages needs pypdfium2 and pillow, which '
            'osintgpt requires: reinstall with pip install '
            '--force-reinstall osintgpt'
        ) from error

    import io

    document = pypdfium2.PdfDocument(str(path))
    try:
        bitmap = document[index].render(scale=RENDER_SCALE)
        buffer = io.BytesIO()
        bitmap.to_pil().save(buffer, format='PNG')

        return buffer.getvalue()
    finally:
        document.close()


# read a PDF as markdown
def extract_pdf(
    path: Union[str, Path],
    transcriber: Optional[Transcriber] = None,
    min_page_chars: int = MIN_PAGE_CHARS
) -> PdfExtraction:
    '''
    Read a PDF, transcribing only the pages that need it.

    Transcription produces markdown rather than an image embedding because the
    result feeds every retrieval leg — exact search and graph extraction
    included — and works with any embedding model. An embedded page image
    would be reachable only by the semantic leg, and only with a multimodal
    model.

    Args:
        path (Union[str, Path]): PDF to read.
        transcriber (Transcriber, optional): Turns rendered page bytes into \
            markdown. Without one, pages needing it degrade to a placeholder \
            rather than failing the document.
        min_page_chars (int): Below this, a page is treated as an image of \
            text.

    Returns:
        PdfExtraction: Per-page results and the markdown they compose.
    '''
    path = Path(path)
    pages: List[PageExtraction] = []

    for index, text in enumerate(extract_page_texts(path)):
        number = index + 1

        if len(text) >= min_page_chars:
            pages.append(PageExtraction(number=number, text=text))
            continue

        if transcriber is None:
            # An honest gap. The page is named so a reader knows something is
            # missing and what would recover it.
            pages.append(PageExtraction(
                number=number,
                text=PLACEHOLDER.format(page=number),
                empty=True
            ))
            continue

        try:
            transcribed = transcriber(render_page(path, index), number).strip()
        except Exception as error:  # noqa: BLE001 — one page, not the document
            log.warning('page %d of %s could not be transcribed: %s',
                        number, path.name, error)
            pages.append(PageExtraction(
                number=number, text=FAILED.format(page=number), empty=True
            ))
            continue

        # A page that transcribes to nothing was probably blank, and saying so
        # is better than an empty stretch in the middle of a document.
        if not transcribed:
            pages.append(PageExtraction(
                number=number,
                text=PLACEHOLDER.format(page=number),
                empty=True
            ))
            continue

        pages.append(PageExtraction(
            number=number, text=transcribed, transcribed=True
        ))

    return PdfExtraction(pages=pages)
