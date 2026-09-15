# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_conversion.py
# Description: One file as markdown, and the promise that producing it costs
#   nothing unless a page is actually sent to a model.
# =================================================================================

import pytest

# import osintgpt ingestion
from osintgpt.ingestion import Conversion, to_markdown
from osintgpt.ingestion.conversion import (
    DOCX,
    MARKITDOWN,
    PDF,
    TABULAR,
    TEXT,
    reader_for
)
from osintgpt.ingestion.documents import FieldMapping
from osintgpt.ingestion.tabular import UnmappedSourceError


@pytest.fixture
def records(tmp_path):
    path = tmp_path / 'registros.csv'
    path.write_text(
        'record_id,body,fecha\n'
        'r1,primer registro del expediente,2024-01-01\n'
        'r2,segundo registro del expediente,2024-01-02\n',
        encoding='utf-8'
    )

    return path


class TestReaders:
    def test_every_readable_family_names_its_reader(self, tmp_path):
        expected = {
            'a.pdf': PDF, 'a.docx': DOCX, 'a.csv': TABULAR, 'a.xlsx': TABULAR,
            'a.json': TABULAR, 'a.md': TEXT, 'a.txt': TEXT, 'a.html': TEXT,
            'a.pptx': MARKITDOWN, 'a.epub': MARKITDOWN
        }

        for name, reader in expected.items():
            assert reader_for(tmp_path / name) == reader, name

    def test_an_unreadable_extension_names_no_reader(self, tmp_path):
        assert reader_for(tmp_path / 'capture.bin') == ''


class TestProse:
    def test_a_markdown_file_converts_to_itself(self, tmp_path):
        path = tmp_path / 'note.md'
        path.write_text('# Title\n\nA paragraph.\n', encoding='utf-8')

        conversion = to_markdown(path)

        assert isinstance(conversion, Conversion)
        assert conversion.reader == TEXT
        assert conversion.documents == 1
        assert conversion.markdown == '# Title\n\nA paragraph.'

    def test_frontmatter_is_separated_the_way_indexing_separates_it(
        self, tmp_path
    ):
        '''
        What the operator sees has to be what would be chunked, and a metadata
        block is never chunked.
        '''
        path = tmp_path / 'note.md'
        path.write_text(
            '---\nauthor: Ana\n---\n\nThe body.\n', encoding='utf-8'
        )

        assert to_markdown(path).markdown == 'The body.'

    def test_an_empty_file_converts_to_nothing_rather_than_failing(
        self, tmp_path
    ):
        path = tmp_path / 'empty.txt'
        path.write_text('   \n', encoding='utf-8')

        conversion = to_markdown(path)

        assert conversion.markdown == ''
        assert conversion.documents == 0


class TestStructured:
    def test_a_structured_file_without_a_mapping_is_refused(self, records):
        with pytest.raises(UnmappedSourceError):
            to_markdown(records)

    def test_each_record_becomes_its_own_section(self, records):
        conversion = to_markdown(
            records,
            FieldMapping(
                content=('body',), timestamp='fecha', identity='record_id'
            )
        )

        assert conversion.reader == TABULAR
        assert conversion.documents == 2
        assert conversion.markdown.count('## ') == 2
        assert '- timestamp: 2024-01-01' in conversion.markdown
        assert 'primer registro del expediente' in conversion.markdown

    def test_a_record_heading_carries_the_ref_it_would_be_cited_by(
        self, records
    ):
        conversion = to_markdown(
            records, FieldMapping(content=('body',), identity='record_id')
        )

        assert '#r1' in conversion.markdown
        assert '#r2' in conversion.markdown


class TestRefusals:
    def test_an_image_says_why_it_has_no_markdown(self, tmp_path):
        path = tmp_path / 'scan.png'
        path.write_bytes(b'\x89PNG\r\n')

        with pytest.raises(ValueError, match='images are embedded'):
            to_markdown(path)

    def test_an_unreadable_extension_lists_what_is_readable(self, tmp_path):
        path = tmp_path / 'capture.bin'
        path.write_bytes(b'\x00\x01')

        with pytest.raises(ValueError, match='no loader'):
            to_markdown(path)


class TestScannedPages:
    '''
    The PDF branch, which is the only one that can reach a model at all.
    '''

    @pytest.fixture
    def scanned(self, tmp_path):
        pypdfium2 = pytest.importorskip('pypdfium2')
        path = tmp_path / 'scanned.pdf'
        document = pypdfium2.PdfDocument.new()
        for _ in range(2):
            document.new_page(595, 842).gen_content()
        document.save(str(path))
        document.close()

        return path

    def test_pages_nothing_can_read_are_counted_and_named(self, scanned):
        conversion = to_markdown(scanned)

        assert conversion.reader == PDF
        assert conversion.pages == 2
        assert conversion.empty_pages == 2
        assert conversion.transcribed_pages == 0
        assert 'no extractable text' in conversion.markdown

    def test_a_transcriber_is_used_only_when_given(self, scanned):
        calls = []

        def transcribe(image, page):
            calls.append(page)

            return f'Page {page} as read from the scan.'

        conversion = to_markdown(scanned, transcriber=transcribe)

        assert calls == [1, 2]
        assert conversion.transcribed_pages == 2
        assert conversion.empty_pages == 0
        assert 'as read from the scan' in conversion.markdown
