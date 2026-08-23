# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_loaders.py
# Description: Reading files into documents, and the refusal to guess which
#   fields of a structured record carry its content.
# =================================================================================

# import modules
import json
import pytest

# import osintgpt ingestion
from osintgpt.ingestion import (
    Document,
    FieldMapping,
    UnmappedSourceError,
    describe_fields,
    load_documents,
    value_at
)
from osintgpt.ingestion.loaders import needs_mapping

# Field values chosen so no example implies a preferred language or region.
RECORDS = [
    {'id': 'a1', 'body': 'First record content.', 'handle': 'alpha',
     'at': '2026-03-01', 'score': 10},
    {'id': 'a2', 'body': 'Second record content.', 'handle': 'beta',
     'at': '2026-03-02', 'score': 20},
    {'id': 'a3', 'body': '', 'handle': 'gamma', 'at': '2026-03-03',
     'score': 30}
]

MAPPING = FieldMapping(
    content=('body',), metadata=('handle', 'at'), identity='id'
)


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / 'records.csv'
    lines = ['id,body,handle,at,score']
    lines += [
        f"{r['id']},{r['body']},{r['handle']},{r['at']},{r['score']}"
        for r in RECORDS
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')

    return path


@pytest.fixture
def json_file(tmp_path):
    path = tmp_path / 'records.json'
    path.write_text(json.dumps({'data': {'items': RECORDS}}), encoding='utf-8')

    return path


@pytest.fixture
def xlsx_file(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / 'records.xlsx'
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['id', 'body', 'handle', 'at', 'score'])
    for record in RECORDS:
        sheet.append([record[k] for k in ('id', 'body', 'handle', 'at', 'score')])
    workbook.save(path)

    return path


class TestProse:
    @pytest.mark.parametrize('name', ['a.txt', 'a.md', 'a.rst', 'a.log'])
    def test_a_whole_file_is_one_document(self, tmp_path, name):
        path = tmp_path / name
        path.write_text('# Title\n\nSome content here.', encoding='utf-8')

        documents = load_documents(path)

        assert len(documents) == 1
        assert 'Some content here.' in documents[0].text

    def test_the_ref_is_the_path(self, tmp_path):
        path = tmp_path / 'report.md'
        path.write_text('Content.', encoding='utf-8')

        assert load_documents(path)[0].ref == path.as_posix()

    def test_an_empty_file_yields_nothing(self, tmp_path):
        path = tmp_path / 'empty.txt'
        path.write_text('   \n\n  ', encoding='utf-8')

        assert load_documents(path) == []

    def test_prose_needs_no_mapping(self, tmp_path):
        path = tmp_path / 'a.txt'
        path.write_text('Content.', encoding='utf-8')

        assert needs_mapping(path) is False
        assert load_documents(path)


class TestHtml:
    def test_tags_are_stripped(self, tmp_path):
        path = tmp_path / 'page.html'
        path.write_text(
            '<html><body><p>Visible text.</p></body></html>', encoding='utf-8'
        )

        text = load_documents(path)[0].text

        assert 'Visible text.' in text
        assert '<p>' not in text

    def test_scripts_and_styles_are_dropped(self, tmp_path):
        path = tmp_path / 'page.html'
        path.write_text(
            '<html><head><style>.a{color:red}</style></head>'
            '<body><script>var x = 1;</script><p>Real content.</p></body>'
            '</html>',
            encoding='utf-8'
        )

        text = load_documents(path)[0].text

        assert 'Real content.' in text
        assert 'color:red' not in text
        assert 'var x' not in text

    def test_block_elements_keep_paragraphs_apart(self, tmp_path):
        path = tmp_path / 'page.html'
        path.write_text(
            '<p>First para.</p><p>Second para.</p>', encoding='utf-8'
        )

        text = load_documents(path)[0].text

        assert 'First para.' in text
        assert 'Second para.' in text
        assert 'First para. Second para.' not in text

    def test_entities_are_decoded(self, tmp_path):
        path = tmp_path / 'page.html'
        path.write_text('<p>caf&#233; &amp; more</p>', encoding='utf-8')

        assert 'café & more' in load_documents(path)[0].text


class TestStructuredRefusesToGuess:
    @pytest.mark.parametrize('fixture', ['csv_file', 'json_file', 'xlsx_file'])
    def test_loading_without_a_mapping_raises(self, request, fixture):
        '''
        Indexing every field would bury content under identifiers that repeat
        across every record, and the result looks populated rather than broken.
        '''
        path = request.getfixturevalue(fixture)

        with pytest.raises(UnmappedSourceError):
            load_documents(path)

    def test_the_error_lists_the_available_fields(self, csv_file):
        with pytest.raises(UnmappedSourceError) as excinfo:
            load_documents(csv_file)

        message = str(excinfo.value)

        assert 'body' in message
        assert 'handle' in message

    def test_structured_formats_are_flagged_as_needing_one(self, csv_file):
        assert needs_mapping(csv_file) is True


class TestStructuredLoading:
    @pytest.mark.parametrize('fixture', ['csv_file', 'json_file', 'xlsx_file'])
    def test_one_document_per_record_with_content(self, request, fixture):
        path = request.getfixturevalue(fixture)
        mapping = MAPPING
        if fixture == 'json_file':
            mapping = FieldMapping(
                content=('body',), metadata=('handle', 'at'), identity='id',
                records='data.items'
            )

        documents = load_documents(path, mapping)

        # The third record has empty content and is not a document.
        assert len(documents) == 2

    def test_only_content_fields_reach_the_text(self, csv_file):
        documents = load_documents(csv_file, MAPPING)

        assert documents[0].text == 'First record content.'
        assert 'alpha' not in documents[0].text
        assert '2026-03-01' not in documents[0].text

    def test_metadata_rides_along_unembedded(self, csv_file):
        documents = load_documents(csv_file, MAPPING)

        assert documents[0].metadata == {'handle': 'alpha', 'at': '2026-03-01'}

    def test_unlisted_fields_are_dropped(self, csv_file):
        documents = load_documents(csv_file, MAPPING)

        assert 'score' not in documents[0].metadata
        assert '10' not in documents[0].text

    def test_the_ref_uses_the_identity_field(self, csv_file):
        documents = load_documents(csv_file, MAPPING)

        assert documents[0].ref.endswith('#a1')
        assert documents[1].ref.endswith('#a2')

    def test_without_an_identity_field_the_ref_uses_position(self, csv_file):
        documents = load_documents(csv_file, FieldMapping(content=('body',)))

        assert documents[0].ref.endswith('#0')

    def test_several_content_fields_join(self, csv_file):
        documents = load_documents(
            csv_file, FieldMapping(content=('handle', 'body'))
        )

        assert documents[0].text == 'alpha\n\nFirst record content.'


class TestNestedRecords:
    def test_a_dotted_path_reaches_a_nested_field(self):
        record = {'user': {'name': 'someone'}, 'body': 'text'}

        assert value_at(record, 'user.name') == 'someone'

    def test_a_missing_path_is_none_rather_than_an_error(self):
        assert value_at({'a': 1}, 'a.b.c') is None
        assert value_at({'a': 1}, 'nope') is None

    def test_records_points_at_the_array(self, json_file):
        mapping = FieldMapping(content=('body',), records='data.items')

        assert len(load_documents(json_file, mapping)) == 2

    def test_a_top_level_array_needs_no_records_path(self, tmp_path):
        path = tmp_path / 'flat.json'
        path.write_text(json.dumps(RECORDS), encoding='utf-8')

        assert len(load_documents(path, FieldMapping(content=('body',)))) == 2

    def test_line_delimited_json_reads_one_record_per_line(self, tmp_path):
        path = tmp_path / 'records.jsonl'
        path.write_text(
            '\n'.join(json.dumps(r) for r in RECORDS), encoding='utf-8'
        )

        assert len(load_documents(path, FieldMapping(content=('body',)))) == 2

    def test_nested_metadata_is_addressed_by_path(self, tmp_path):
        path = tmp_path / 'nested.json'
        path.write_text(
            json.dumps([{'body': 'text', 'user': {'name': 'someone'}}]),
            encoding='utf-8'
        )
        mapping = FieldMapping(content=('body',), metadata=('user.name',))

        assert load_documents(path, mapping)[0].metadata == {
            'user.name': 'someone'
        }


class TestDescribeFields:
    def test_reports_every_field(self, csv_file):
        described = describe_fields(csv_file)

        assert set(described) == {'id', 'body', 'handle', 'at', 'score'}

    def test_average_length_distinguishes_content_from_identifiers(
        self, csv_file
    ):
        described = describe_fields(csv_file)

        assert described['body']['average_length'] > described['id'][
            'average_length'
        ]

    def test_a_unique_field_is_flagged(self, tmp_path):
        '''
        Uniqueness separates an identifier from a field that repeats, and is
        only a signal at sample size — two differing rows say nothing.
        '''
        path = tmp_path / 'repeats.csv'
        rows = '\n'.join(f'{i},shared value' for i in range(20))
        path.write_text(f'id,category\n{rows}', encoding='utf-8')

        described = describe_fields(path)

        assert described['id']['unique'] is True
        assert described['category']['unique'] is False

    def test_an_example_value_is_offered(self, csv_file):
        assert describe_fields(csv_file)['handle']['example'] == 'alpha'

    def test_nested_fields_are_offered_as_paths(self, tmp_path):
        path = tmp_path / 'nested.json'
        path.write_text(
            json.dumps([{'body': 'text', 'user': {'name': 'someone'}}]),
            encoding='utf-8'
        )

        assert 'user.name' in describe_fields(path)

    def test_it_describes_rather_than_decides(self, csv_file):
        '''
        Nothing here names a role. Choosing is the operator's, because a wrong
        guess produces an index that is populated and useless.
        '''
        described = describe_fields(csv_file)

        for report in described.values():
            assert 'content' not in report
            assert 'role' not in report


class TestUnsupported:
    def test_an_unknown_extension_lists_what_is_supported(self, tmp_path):
        path = tmp_path / 'archive.zip'
        path.write_bytes(b'not a document')

        with pytest.raises(ValueError) as excinfo:
            load_documents(path)

        message = str(excinfo.value)

        assert '.zip' in message
        assert '.csv' in message


class TestFrontmatter:
    BLOCK = (
        '---\n'
        'type: synthesis\n'
        'version: 2\n'
        'prepared_by: an analyst\n'
        '---\n'
    )

    def test_it_becomes_metadata_rather_than_text(self, tmp_path):
        '''
        Frontmatter describes a document rather than saying anything, and its
        fields are what a citation wants. Embedding it puts field names into
        the vector instead.
        '''
        path = tmp_path / 'report.md'
        path.write_text(f'{self.BLOCK}\n# Title\n\nBody text.', encoding='utf-8')

        document = load_documents(path)[0]

        assert document.metadata['type'] == 'synthesis'
        assert document.metadata['version'] == '2'
        assert 'prepared_by' not in document.text
        assert document.text.startswith('# Title')

    def test_a_byte_order_mark_does_not_hide_it(self, tmp_path):
        '''An editor's BOM sits in front of the opening rule.'''
        path = tmp_path / 'report.md'
        path.write_text(
            f'{self.BLOCK}\n# Title\n\nBody.', encoding='utf-8-sig'
        )

        document = load_documents(path)[0]

        assert document.metadata['type'] == 'synthesis'
        assert not document.text.startswith('\ufeff')

    def test_a_rule_further_down_is_left_alone(self, tmp_path):
        path = tmp_path / 'report.md'
        path.write_text('# Title\n\nBody.\n\n---\n\nMore body.', encoding='utf-8')

        document = load_documents(path)[0]

        assert document.metadata == {}
        assert '---' in document.text

    def test_an_unparseable_block_is_still_removed(self, tmp_path):
        path = tmp_path / 'report.md'
        path.write_text(
            '---\ntags:\n  - one\n  - two\n---\n\nBody text.', encoding='utf-8'
        )

        document = load_documents(path)[0]

        assert document.text == 'Body text.'

    def test_plain_text_files_are_not_searched_for_one(self, tmp_path):
        path = tmp_path / 'notes.txt'
        path.write_text(f'{self.BLOCK}\nBody.', encoding='utf-8')

        document = load_documents(path)[0]

        assert document.metadata == {}
        assert document.text.startswith('---')


class TestProseProvenance:
    '''
    A prose document has no columns to map, but retrieval still filters on
    when and who — and a question about a month should not land on a mapped
    spreadsheet while silently missing every markdown file beside it.
    '''

    def write(self, tmp_path, frontmatter):
        path = tmp_path / 'report.md'
        path.write_text(
            f'---\n{frontmatter}\n---\n\n# Title\n\nBody.', encoding='utf-8'
        )

        return path

    def test_a_conventional_date_key_is_read(self, tmp_path):
        path = self.write(tmp_path, 'date: 2026-04-22')

        assert load_documents(path)[0].timestamp == '2026-04-22'

    def test_a_conventional_author_key_is_read(self, tmp_path):
        path = self.write(tmp_path, 'author: an analyst')

        assert load_documents(path)[0].author == 'an analyst'

    @pytest.mark.parametrize('key', [
        'date', 'created', 'created_at', 'published', 'published_at'
    ])
    def test_every_conventional_timestamp_key(self, tmp_path, key):
        path = self.write(tmp_path, f'{key}: 2026-04-22')

        assert load_documents(path)[0].timestamp == '2026-04-22'

    def test_a_source_can_name_its_own_field(self, tmp_path):
        path = self.write(tmp_path, 'collected_on: 2026-04-22')
        mapping = FieldMapping(timestamp='collected_on')

        assert load_documents(path, mapping)[0].timestamp == '2026-04-22'

    def test_a_named_field_beats_a_conventional_one(self, tmp_path):
        path = self.write(
            tmp_path, 'date: 2026-01-01\ncollected_on: 2026-04-22'
        )
        mapping = FieldMapping(timestamp='collected_on')

        assert load_documents(path, mapping)[0].timestamp == '2026-04-22'

    def test_nothing_is_invented_when_absent(self, tmp_path):
        '''
        A guessed timestamp is worse than an absent one: a filter built on it
        fails silently rather than reporting nothing to filter.
        '''
        path = self.write(tmp_path, 'type: synthesis')
        document = load_documents(path)[0]

        assert document.timestamp == ''
        assert document.author == ''

    def test_a_document_with_no_frontmatter_carries_neither(self, tmp_path):
        path = tmp_path / 'plain.md'
        path.write_text('# Title\n\nBody.', encoding='utf-8')
        document = load_documents(path)[0]

        assert document.timestamp == ''
        assert document.author == ''

    def test_the_value_stays_in_metadata_too(self, tmp_path):
        '''Named separately for filtering, kept in metadata for citation.'''
        path = self.write(tmp_path, 'date: 2026-04-22')
        document = load_documents(path)[0]

        assert document.metadata['date'] == '2026-04-22'
