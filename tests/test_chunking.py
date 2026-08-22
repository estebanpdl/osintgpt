# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_chunking.py
# Description: Chunk boundaries. Silent wrongness here is invisible until
#   retrieval is bad, so the properties are pinned rather than the shapes.
# =================================================================================

# import modules
import pytest

# import osintgpt ingestion
from osintgpt.ingestion.chunking import MAX_CHARS, chunk_text


class TestProperties:
    '''Hold for any input, which is what makes them worth asserting.'''

    @pytest.mark.parametrize('text', [
        '',
        '   \n\n  \t ',
        'one short line',
        '# Heading\n\nBody text.',
        'word ' * 2_000,
        'x' * 5_000,
        '# A\n\n' + ('para\n\n' * 500)
    ])
    def test_no_chunk_exceeds_the_cap(self, text):
        assert all(len(chunk) <= MAX_CHARS for chunk in chunk_text(text))

    @pytest.mark.parametrize('text', [
        '',
        '   \n\n  ',
        '#\n#\n#',
        'real content'
    ])
    def test_no_chunk_is_blank(self, text):
        assert all(chunk.strip() for chunk in chunk_text(text))

    def test_every_word_survives(self):
        '''
        Chunking must not lose content. Order and boundaries can change; a
        dropped sentence is a document that cannot be retrieved.
        '''
        text = '# Title\n\n' + '\n\n'.join(
            f'Paragraph {i} with some words in it.' for i in range(200)
        )
        rejoined = ' '.join(chunk_text(text))

        for i in range(200):
            assert f'Paragraph {i} ' in rejoined

    def test_chunks_stay_in_document_order(self):
        text = '\n\n'.join(f'Section {i:03d} content.' for i in range(100))
        chunks = chunk_text(text)
        positions = [text.index(chunk.splitlines()[0]) for chunk in chunks]

        assert positions == sorted(positions)


class TestEmptyInput:
    @pytest.mark.parametrize('text', ['', '   ', '\n\n\n', '\t'])
    def test_produces_nothing(self, text):
        assert chunk_text(text) == []


class TestHeadings:
    def test_a_heading_starts_a_chunk(self):
        text = '# First\n\nAlpha.\n\n# Second\n\nBeta.'
        chunks = chunk_text(text)

        assert len(chunks) == 2
        assert chunks[0].startswith('# First')
        assert chunks[1].startswith('# Second')

    def test_a_heading_travels_with_its_body(self):
        '''
        A chunk that has lost its heading is a chunk nobody can place.
        '''
        chunks = chunk_text('## Operation Blackcore\n\nThe group met in May.')

        assert 'Operation Blackcore' in chunks[0]
        assert 'met in May' in chunks[0]

    @pytest.mark.parametrize('level', range(1, 7))
    def test_every_heading_level_splits(self, level):
        marker = '#' * level
        text = f'{marker} One\n\nAlpha.\n\n{marker} Two\n\nBeta.'

        assert len(chunk_text(text)) == 2

    def test_a_hash_without_a_space_is_not_a_heading(self):
        '''#hashtag and #1 are content, not structure.'''
        text = 'Post about #osint and #1 trending.'

        assert len(chunk_text(text)) == 1

    def test_text_before_the_first_heading_is_kept(self):
        chunks = chunk_text('Preamble text.\n\n# Later\n\nBody.')

        assert 'Preamble text.' in chunks[0]


class TestOversizedSections:
    def test_a_long_section_splits_on_paragraphs(self):
        paragraph = 'A sentence about something. ' * 10
        text = '# Long\n\n' + '\n\n'.join([paragraph] * 20)
        chunks = chunk_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= MAX_CHARS for chunk in chunks)

    def test_paragraphs_are_packed_rather_than_one_per_chunk(self):
        '''
        A chunk per paragraph would fragment context and multiply cost.
        '''
        text = '\n\n'.join(['Short paragraph.'] * 40)
        chunks = chunk_text(text)

        assert len(chunks) < 40

    def test_windows_line_endings_split_the_same(self):
        unix = '\n\n'.join(['Short paragraph.'] * 40)
        windows = unix.replace('\n', '\r\n')

        assert len(chunk_text(windows)) == len(chunk_text(unix))

    def test_blank_lines_with_whitespace_still_separate(self):
        '''
        A blank line carrying spaces is still a paragraph break. Real documents
        have them, and a bare '\\n\\n' split would run two paragraphs together.
        '''
        chunks = chunk_text('x' * (MAX_CHARS - 5) + '\n   \n' + 'tail text')

        assert len(chunks) == 2
        assert chunks[1] == 'tail text'

    def test_text_under_the_cap_is_returned_unchanged(self):
        '''
        Only oversized sections are re-split, so text that needs no splitting
        keeps its own formatting rather than being normalized.
        '''
        text = 'First paragraph.\n   \nSecond paragraph.'

        assert chunk_text(text) == [text]


class TestHardSplit:
    def test_one_enormous_paragraph_still_chunks(self):
        chunks = chunk_text('word ' * 3_000)

        assert len(chunks) > 1
        assert all(len(chunk) <= MAX_CHARS for chunk in chunks)

    def test_cuts_at_whitespace_rather_than_mid_word(self):
        chunks = chunk_text('alpha ' * 1_000)

        for chunk in chunks[:-1]:
            assert chunk.endswith('alpha')

    def test_text_with_no_whitespace_still_chunks(self):
        '''
        A long identifier, or a language that does not space its words.
        '''
        chunks = chunk_text('x' * 4_000)

        assert len(chunks) == 3
        assert ''.join(chunks) == 'x' * 4_000


class TestMultilingual:
    @pytest.mark.parametrize('text', [
        'Análisis de las elecciones. ' * 200,
        'تحليل الروايات المتعددة. ' * 200,
        'Анализ нарративов в сети. ' * 200,
        '多语言叙事分析。' * 400
    ])
    def test_non_english_text_chunks_within_the_cap(self, text):
        chunks = chunk_text(text)

        assert chunks
        assert all(len(chunk) <= MAX_CHARS for chunk in chunks)

    def test_accented_content_survives_intact(self):
        text = '# Elecciones\n\nLa campaña de desinformación en español.'
        chunks = chunk_text(text)

        assert 'campaña de desinformación' in chunks[0]


class TestConfigurableCap:
    def test_a_smaller_cap_is_respected(self):
        chunks = chunk_text('word ' * 500, max_chars=200)

        assert all(len(chunk) <= 200 for chunk in chunks)

    def test_a_larger_cap_produces_fewer_chunks(self):
        text = 'word ' * 2_000

        assert len(chunk_text(text, max_chars=3_000)) < len(
            chunk_text(text, max_chars=500)
        )
