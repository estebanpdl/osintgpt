# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: tabular.py
# Description: Rows and records into documents. These formats carry many fields
#   and only some are content, so the mapping is required rather than guessed.
# =================================================================================

# import modules
import csv
import json

# import submodules
from dataclasses import dataclass
from pathlib import Path

# type hints
from typing import Any, Dict, Iterable, Iterator, List, Union

from .documents import (
    Document,
    FieldMapping,
    content_for,
    document_from_record,
    value_at
)

# Rows sampled when describing a file's fields. Enough to tell a body of text
# from an identifier without reading a corpus to answer a question about it.
SAMPLE_ROWS = 50

# Enough of a kept record to recognise it as the field the operator meant.
EXAMPLE_CHARS = 160


# UnmappedSourceError class
class UnmappedSourceError(ValueError):
    '''
    Raised when a structured file is loaded without being told which fields
    carry its content.
    '''


# describe what a structured file contains
def describe_fields(path: Union[str, Path]) -> Dict[str, dict]:
    '''
    Sample a file and report what each field looks like.

    Feeds the dry run, where an operator turns a description into a mapping.
    Reports what the values are; it never decides what they are for.

    Args:
        path (Union[str, Path]): File to sample.

    Returns:
        Dict[str, dict]: Per field, its average text length, how many of the \
            sampled values were filled, whether they were unique, and one \
            example.
    '''
    records = list(_sample(Path(path)))
    fields: Dict[str, dict] = {}

    for name in _field_names(records):
        values = [value_at(record, name) for record in records]
        filled = [str(v).strip() for v in values if v is not None and str(v).strip()]
        fields[name] = {
            'filled': len(filled),
            'sampled': len(values),
            'average_length': round(
                sum(len(v) for v in filled) / len(filled)
            ) if filled else 0,
            'unique': len(set(filled)) == len(filled) and len(filled) > 1,
            'example': filled[0][:120] if filled else ''
        }

    return fields


# MappingPreview class
@dataclass(frozen=True)
class MappingPreview:
    '''
    What a mapping would make of a file, counted over the whole of it.
    '''
    records: int = 0
    kept: int = 0
    # Records whose primary content field was empty.
    without_content: int = 0
    # Records that had content, but less of it than the mapping asks for.
    too_short: int = 0
    shortest: int = 0
    example: str = ''

    @property
    def dropped(self) -> int:
        return self.without_content + self.too_short

    # several files of one shape, counted as one
    @classmethod
    def merged(cls, previews: Iterable['MappingPreview']) -> 'MappingPreview':
        '''
        Args:
            previews (Iterable[MappingPreview]): Per-file counts.

        Returns:
            MappingPreview: The totals, carrying the shortest record found in \
                any of the files — the one a floor is judged against.
        '''
        records = kept = without_content = too_short = 0
        shortest = 0
        example = ''

        for preview in previews:
            records += preview.records
            kept += preview.kept
            without_content += preview.without_content
            too_short += preview.too_short
            if preview.kept and (not shortest or preview.shortest < shortest):
                shortest = preview.shortest
                example = preview.example

        return cls(
            records=records,
            kept=kept,
            without_content=without_content,
            too_short=too_short,
            shortest=shortest,
            example=example
        )

    # operator-facing summary
    @property
    def summary(self) -> str:
        '''
        Returns:
            str: One line naming what would be indexed and what a rule would \
                discard, each cause separately — a floor set too high and a \
                content field named wrongly look identical in a total.
        '''
        if not self.records:
            return 'no records'

        parts = [f'{self.kept:,} of {self.records:,} records']
        if self.without_content:
            parts.append(f'{self.without_content:,} with no content')
        if self.too_short:
            parts.append(f'{self.too_short:,} under the floor')
        if self.kept:
            parts.append(f'shortest {self.shortest:,} chars')

        return ', '.join(parts)


# test a mapping against the whole file
def preview_mapping(
    path: Union[str, Path],
    mapping: FieldMapping
) -> MappingPreview:
    '''
    Report what a mapping would keep and discard, without indexing anything.

    Reads every record. Exports are ordered, so a share read off the opening
    rows is the share for one account in one period, not for the file.

    Args:
        path (Union[str, Path]): Structured file to read.
        mapping (FieldMapping): The roles to test.

    Returns:
        MappingPreview: Counts, and the shortest record that would survive.
    '''
    path = Path(path)
    if not mapping.is_set:
        return MappingPreview()

    records = kept = without_content = too_short = 0
    shortest = 0
    example = ''

    for record in _records(path, mapping):
        records += 1

        text = content_for(record, mapping)
        if not text:
            without_content += 1
            continue
        if len(text) < mapping.min_chars:
            too_short += 1
            continue

        kept += 1
        if not shortest or len(text) < shortest:
            shortest = len(text)
            example = text[:EXAMPLE_CHARS]

    return MappingPreview(
        records=records,
        kept=kept,
        without_content=without_content,
        too_short=too_short,
        shortest=shortest,
        example=example
    )


# test a mapping against every file sharing a schema
def preview_files(
    paths: Iterable[Union[str, Path]],
    mapping: FieldMapping
) -> MappingPreview:
    '''
    Report what a mapping would keep across several files of the same shape.

    Reads all of them. Files sharing a schema do not share their contents, and
    the blank rows a floor exists to catch are not spread evenly between them.

    Args:
        paths (Iterable[Union[str, Path]]): Structured files to read.
        mapping (FieldMapping): The roles to test.

    Returns:
        MappingPreview: The totals across all of them.
    '''
    return MappingPreview.merged(
        preview_mapping(path, mapping) for path in paths
    )


# load a structured file as documents
def load_records(
    path: Union[str, Path], mapping: FieldMapping, *, statistics=None
) -> Iterator[Document]:
    '''
    Read a tabular or nested file into one document per record.

    Args:
        path (Union[str, Path]): File to read.
        mapping (FieldMapping): Which fields play which role.

    Raises:
        UnmappedSourceError: If no content field was named. Indexing every \
            field would bury the content under identifiers that repeat across \
            every record, and the result looks populated rather than broken.

    Yields:
        Document: One per record that had content.
    '''
    path = Path(path)
    if not mapping.is_set:
        raise UnmappedSourceError(
            f'{path.name} is structured data; name the field or fields '
            'carrying its content before indexing it. Available fields: '
            + ', '.join(describe_fields(path)) or '(none found)'
        )

    source_ref = path.as_posix()
    if statistics is not None:
        statistics.update(records=0, kept=0, without_content=0, too_short=0)
    for position, record in enumerate(_records(path, mapping)):
        document = document_from_record(record, mapping, source_ref, position)
        if statistics is not None:
            statistics['records'] += 1
            key = 'kept' if document is not None else 'without_content' if not content_for(record, mapping) else 'too_short'
            statistics[key] += 1
        if document is not None:
            yield document


# records from any supported structured format
def _records(path: Path, mapping: FieldMapping) -> Iterator[Any]:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return _csv_records(path)
    if suffix in ('.xlsx', '.xlsm'):
        return _excel_records(path)
    if suffix in ('.json', '.jsonl', '.ndjson'):
        return _json_records(path, mapping)

    raise ValueError(f'{path.name}: not a structured format osintgpt reads')


def _csv_records(path: Path) -> Iterator[dict]:
    # newline='' is required by csv; utf-8-sig drops the BOM that spreadsheet
    # exports carry, which would otherwise corrupt the first column's name.
    with open(path, newline='', encoding='utf-8-sig', errors='replace') as handle:
        yield from csv.DictReader(handle)


def _excel_records(path: Path) -> Iterator[dict]:
    from openpyxl import load_workbook

    # read_only streams rather than loading the sheet; data_only takes a
    # formula's last computed value instead of its source.
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            return

        names = [str(name) if name is not None else '' for name in header]
        for row in rows:
            yield {
                name: value
                for name, value in zip(names, row)
                if name
            }
    finally:
        workbook.close()


def _json_records(path: Path, mapping: FieldMapping) -> Iterator[Any]:
    text = path.read_text(encoding='utf-8', errors='replace')

    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return

    parsed = json.loads(text)
    if mapping.records:
        parsed = value_at(parsed, mapping.records)

    if isinstance(parsed, list):
        yield from parsed
    elif isinstance(parsed, dict):
        yield parsed


# a bounded sample, for describing rather than loading
def _sample(path: Path) -> Iterator[Any]:
    empty = FieldMapping()
    try:
        for index, record in enumerate(_records(path, empty)):
            if index >= SAMPLE_ROWS:
                return
            yield record
    except (ValueError, KeyError):
        return


def _field_names(records: List[Any]) -> List[str]:
    '''
    Field names in first-seen order, descending into nested records one level
    at a time so a dotted path is offered rather than an opaque object.
    '''
    names: List[str] = []

    def walk(record: Any, prefix: str = '') -> None:
        if not isinstance(record, dict):
            return
        for key, value in record.items():
            path = f'{prefix}{key}'
            if isinstance(value, dict):
                walk(value, f'{path}.')
            elif path not in names:
                names.append(path)

    for record in records:
        walk(record)

    return names
