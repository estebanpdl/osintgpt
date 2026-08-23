# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_pdf.py
# Description: Reading PDFs, and the decision that matters — which pages are
#   text and which are images of text, since only the second kind costs money.
# =================================================================================

# import modules
import pytest

# import osintgpt ingestion
from osintgpt.ingestion import load_documents
from osintgpt.ingestion.pdf import (
    MIN_PAGE_CHARS,
    PdfExtraction,
    extract_pdf,
    extract_page_texts,
    render_page
)

pypdfium2 = pytest.importorskip('pypdfium2')

PARAGRAPH = (
    'Assessed material of a length a born-digital page would carry, well '
    'above the threshold that separates text from an image of text.'
)


def write_pdf(path, pages):
    '''
    A real PDF, built rather than fixtured.

    Each page holds the text it is given; a page given none is genuinely
    empty, which is what a scan looks like to a text extractor.
    '''
    document = pypdfium2.PdfDocument.new()
    for text in pages:
        page = document.new_page(595, 842)
        if text:
            font = document.add_font(
                pypdfium2.raw.FPDF_LoadStandardFont(document.raw, b'Helvetica'),
                type=pypdfium2.raw.FPDF_FONT_TYPE1,
                is_cid=False
            ) if hasattr(document, 'add_font') else None
            if font is not None:
                page.insert_text(text, pos_x=50, pos_y=750, font_size=11,
                                 font=font)
        page.gen_content()
    document.save(str(path))
    document.close()

    return path


@pytest.fixture
def text_pdf(tmp_path):
    return write_pdf(tmp_path / 'report.pdf', [PARAGRAPH, PARAGRAPH])


@pytest.fixture
def scanned_pdf(tmp_path):
    '''Pages with no extractable text — what a scan yields.'''
    return write_pdf(tmp_path / 'scan.pdf', ['', ''])


class TestPageExtraction:
    def test_a_pdf_yields_one_string_per_page(self, scanned_pdf):
        assert len(extract_page_texts(scanned_pdf)) == 2

    def test_pages_render_to_png_bytes(self, scanned_pdf):
        rendered = render_page(scanned_pdf, 0)

        assert rendered.startswith(b'\x89PNG')
        assert len(rendered) > 1_000


class TestWithoutATranscriber:
    def test_a_page_with_no_text_becomes_a_named_gap(self, scanned_pdf):
        '''
        A document with a gap that says so is auditable; one that silently
        omits a page is not.
        '''
        extraction = extract_pdf(scanned_pdf)

        assert 'Page 1' in extraction.markdown
        assert 'scanned or image-only' in extraction.markdown

    def test_the_document_still_reads(self, scanned_pdf):
        extraction = extract_pdf(scanned_pdf)

        assert extraction.markdown
        assert extraction.empty_pages == 2
        assert extraction.transcribed_pages == 0

    def test_it_reports_what_vision_would_cost(self, scanned_pdf):
        '''
        The expensive half of reading a PDF, countable before paying for it.
        '''
        assert extract_pdf(scanned_pdf).needs_vision == 2


class TestWithATranscriber:
    def test_only_pages_needing_it_are_sent(self, scanned_pdf):
        calls = []

        def transcriber(image):
            calls.append(image)

            return 'Transcribed page content.'

        extract_pdf(scanned_pdf, transcriber)

        assert len(calls) == 2
        assert all(image.startswith(b'\x89PNG') for image in calls)

    def test_transcribed_text_reaches_the_markdown(self, scanned_pdf):
        extraction = extract_pdf(scanned_pdf, lambda image: 'Recovered text.')

        assert 'Recovered text.' in extraction.markdown
        assert extraction.transcribed_pages == 2
        assert extraction.empty_pages == 0

    def test_a_failing_page_does_not_fail_the_document(self, scanned_pdf):
        def transcriber(image):
            raise RuntimeError('the provider refused')

        extraction = extract_pdf(scanned_pdf, transcriber)

        assert 'transcription failed' in extraction.markdown
        assert extraction.empty_pages == 2

    def test_a_page_that_transcribes_to_nothing_is_named(self, scanned_pdf):
        extraction = extract_pdf(scanned_pdf, lambda image: '   ')

        assert 'Page 1' in extraction.markdown
        assert extraction.transcribed_pages == 0


class TestThreshold:
    def test_the_threshold_is_configurable(self, scanned_pdf):
        '''
        The character count that separates text from an image of text is a
        signal, not a measurement, so a corpus can move it.
        '''
        calls = []
        extract_pdf(
            scanned_pdf, lambda image: calls.append(image) or 'text',
            min_page_chars=0
        )

        assert calls == []

    def test_the_default_is_low_enough_to_be_a_signal(self):
        assert 0 < MIN_PAGE_CHARS < 200


class TestThroughTheLoader:
    def test_a_pdf_becomes_one_document(self, scanned_pdf):
        documents = load_documents(
            scanned_pdf, transcriber=lambda image: 'Recovered text.'
        )

        assert len(documents) == 1
        assert 'Recovered text.' in documents[0].text

    def test_the_ref_is_the_path(self, scanned_pdf):
        documents = load_documents(scanned_pdf)

        assert documents[0].ref == scanned_pdf.as_posix()

    def test_a_pdf_needs_no_field_mapping(self, scanned_pdf):
        from osintgpt.ingestion.loaders import needs_mapping

        assert needs_mapping(scanned_pdf) is False

    def test_pdfs_are_supported(self):
        from osintgpt.ingestion import SUPPORTED_SUFFIXES

        assert '.pdf' in SUPPORTED_SUFFIXES


class TestEmptyExtraction:
    def test_a_pdf_with_no_pages_yields_no_documents(self, tmp_path):
        path = write_pdf(tmp_path / 'blank.pdf', [])

        assert load_documents(path) == []

    def test_an_extraction_with_no_pages_has_empty_markdown(self):
        assert PdfExtraction().markdown == ''
