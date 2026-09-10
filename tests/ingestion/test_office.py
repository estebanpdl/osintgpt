# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_office.py
# Description: Word documents. The structure a writer applied is what chunking
#   later acts on, so what matters is whether it survives the conversion.
# =================================================================================

# import modules
import pytest

# import osintgpt ingestion
from osintgpt.ingestion import chunk_document, extract_docx, load_documents

docx = pytest.importorskip('docx')

BODY = 'A paragraph of assessed material carrying enough text to be content.'


@pytest.fixture
def document(tmp_path):
    def build(build_document):
        instance = docx.Document()
        build_document(instance)
        path = tmp_path / 'report.docx'
        instance.save(path)

        return path

    return build


class TestStructure:
    def test_headings_become_markdown_levels(self, document):
        def build(doc):
            doc.add_heading('Top Section', level=1)
            doc.add_paragraph(BODY)
            doc.add_heading('Nested Section', level=2)
            doc.add_paragraph(BODY)

        text = extract_docx(document(build))

        assert '# Top Section' in text
        assert '## Nested Section' in text

    def test_body_paragraphs_stay_plain(self, document):
        def build(doc):
            doc.add_paragraph(BODY)

        text = extract_docx(document(build))

        assert text == BODY

    def test_a_document_with_no_styles_yields_plain_prose(self, document):
        '''
        The unstructured case, which is most Word documents in practice.
        '''
        def build(doc):
            for _ in range(5):
                doc.add_paragraph(BODY)

        text = extract_docx(document(build))

        assert '#' not in text
        assert text.count(BODY) == 5

    def test_empty_paragraphs_are_dropped(self, document):
        def build(doc):
            doc.add_paragraph(BODY)
            doc.add_paragraph('')
            doc.add_paragraph('   ')
            doc.add_paragraph(BODY)

        assert extract_docx(document(build)).count('\n\n\n') == 0


class TestTables:
    def build_table(self, doc, name='Layer', rows=3):
        table = doc.add_table(rows=rows, cols=2)
        table.rows[0].cells[0].text = name
        table.rows[0].cells[1].text = 'Description'
        for index in range(1, rows):
            table.rows[index].cells[0].text = f'{name} {index}'
            table.rows[index].cells[1].text = f'Detail {index}'

        return table

    def test_a_mixed_document_renders_in_author_order(self, document):
        def build(doc):
            doc.add_paragraph('Actors assessed:')
            self.build_table(doc, name='Actor', rows=2)
            doc.add_paragraph('Each is assessed below.')

        assert extract_docx(document(build)) == (
            'Actors assessed:\n\n'
            '| Actor | Description |\n'
            '|---|---|\n'
            '| Actor 1 | Detail 1 |\n\n'
            'Each is assessed below.'
        )

    def test_several_tables_stay_interleaved_with_prose(self, document):
        def build(doc):
            doc.add_paragraph('Before first')
            self.build_table(doc, name='First', rows=2)
            doc.add_paragraph('Between tables')
            self.build_table(doc, name='Second', rows=2)
            doc.add_paragraph('After second')

        text = extract_docx(document(build))
        positions = [
            text.index(value)
            for value in (
                'Before first', '| First |', 'Between tables',
                '| Second |', 'After second'
            )
        ]

        assert positions == sorted(positions)

    def test_a_document_without_tables_is_unchanged(self, document):
        def build(doc):
            doc.add_paragraph('First paragraph')
            doc.add_paragraph('Second paragraph')

        assert extract_docx(document(build)) == (
            'First paragraph\n\nSecond paragraph'
        )

    def test_a_table_as_the_first_element_stays_first(self, document):
        def build(doc):
            self.build_table(doc, rows=2)
            doc.add_paragraph('Following prose')

        assert extract_docx(document(build)).startswith(
            '| Layer | Description |\n|---|---|\n| Layer 1 | Detail 1 |'
        )

    def test_a_table_becomes_pipe_rows(self, document):
        text = extract_docx(document(self.build_table))

        assert '| Layer | Description |' in text
        assert '| Layer 1 | Detail 1 |' in text

    def test_the_header_rule_is_written(self, document):
        '''
        Without the rule, chunking cannot tell a header from a data row, and
        an oversized table loses the header it should be repeating.
        '''
        text = extract_docx(document(self.build_table))
        lines = [line for line in text.splitlines() if line.startswith('|')]

        assert '---' in lines[1]

    def test_the_table_survives_chunking_as_a_unit(self, document):
        text = extract_docx(document(self.build_table))
        chunks = chunk_document(text)

        holding = [c for c in chunks if 'Layer 1' in c.text]

        assert len(holding) == 1
        assert 'Layer 2' in holding[0].text

    def test_a_split_table_repeats_its_header(self, document):
        def build(doc):
            self.build_table(doc, rows=80)

        chunks = chunk_document(extract_docx(document(build)))
        header = '| Layer | Description |\n|---|---|'

        assert len(chunks) > 1
        assert all(chunk.text.startswith(header) for chunk in chunks)

    def test_a_cell_spanning_lines_stays_on_one_row(self, document):
        def build(doc):
            table = doc.add_table(rows=1, cols=1)
            table.rows[0].cells[0].text = 'first line\nsecond line'

        text = extract_docx(document(build))

        assert '| first line second line |' in text


class TestThroughTheLoader:
    def test_a_word_document_becomes_one_document(self, document):
        def build(doc):
            doc.add_heading('Report', level=1)
            doc.add_paragraph(BODY)

        documents = load_documents(document(build))

        assert len(documents) == 1
        assert 'Report' in documents[0].text

    def test_it_needs_no_field_mapping(self, document):
        from osintgpt.ingestion.loaders import needs_mapping

        def build(doc):
            doc.add_paragraph(BODY)

        assert needs_mapping(document(build)) is False

    def test_an_empty_document_yields_nothing(self, document):
        assert load_documents(document(lambda doc: None)) == []

    def test_headings_drive_chunking_as_they_would_in_markdown(self, document):
        def build(doc):
            for index in range(4):
                doc.add_heading(f'Section {index}', level=1)
                for _ in range(8):
                    doc.add_paragraph(BODY)

        chunks = chunk_document(extract_docx(document(build)))

        assert len(chunks) >= 4
        assert all(chunk.text.startswith('# Section') for chunk in chunks[:4])

    def test_nested_headings_produce_a_path(self, document):
        '''
        A path exists only where a section sits inside another; a top-level
        section already opens with its own heading and needs none.
        '''
        def build(doc):
            doc.add_heading('Report', level=1)
            for index in range(3):
                doc.add_heading(f'Area {index}', level=2)
                for _ in range(8):
                    doc.add_paragraph(BODY)

        chunks = chunk_document(extract_docx(document(build)))

        assert any(chunk.path == 'Report' for chunk in chunks)

    def test_docx_is_supported(self):
        from osintgpt.ingestion import SUPPORTED_SUFFIXES

        assert '.docx' in SUPPORTED_SUFFIXES
