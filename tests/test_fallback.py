# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_fallback.py
# Description: The last-resort converter, and the boundary that keeps it from
#   displacing a reader chosen for a format.
# =================================================================================

# import modules
import pytest

# import osintgpt ingestion
from osintgpt.ingestion import (
    FALLBACK_SUFFIXES,
    STRUCTURED_SUFFIXES,
    SUPPORTED_SUFFIXES,
    load_documents
)
from osintgpt.ingestion.fallback import can_convert

pytest.importorskip('markitdown')
pptx = pytest.importorskip('pptx')


@pytest.fixture
def deck(tmp_path):
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = 'Assessment Overview'
    slide.placeholders[1].text = 'First finding\nSecond finding'
    path = tmp_path / 'deck.pptx'
    presentation.save(path)

    return path


class TestBoundary:
    def test_it_never_claims_a_format_with_a_reader(self):
        '''
        Where a reader exists it is better: it was chosen for that format, and
        a general converter would undo decisions taken for a reason.
        '''
        assert not (FALLBACK_SUFFIXES & SUPPORTED_SUFFIXES)

    def test_structured_formats_are_not_convertible(self):
        '''
        A CSV through a converter becomes one enormous markdown table, which
        is the opposite of naming which fields carry content.
        '''
        assert not (FALLBACK_SUFFIXES & STRUCTURED_SUFFIXES)
        assert can_convert('records.csv') is False

    @pytest.mark.parametrize('name', ['a.pdf', 'a.docx', 'a.md', 'a.json'])
    def test_formats_with_readers_are_refused(self, name):
        assert can_convert(name) is False

    def test_an_unsupported_format_is_offered(self):
        assert can_convert('deck.pptx') is True


class TestConversion:
    def test_a_slide_deck_becomes_a_document(self, deck):
        documents = load_documents(deck)

        assert len(documents) == 1
        assert 'Assessment Overview' in documents[0].text
        assert 'First finding' in documents[0].text

    def test_the_ref_is_the_path(self, deck):
        assert load_documents(deck)[0].ref == deck.as_posix()

    def test_a_format_nothing_reads_still_raises(self, tmp_path):
        path = tmp_path / 'archive.zip'
        path.write_bytes(b'not a document')

        with pytest.raises(ValueError) as excinfo:
            load_documents(path)

        assert '.zip' in str(excinfo.value)
