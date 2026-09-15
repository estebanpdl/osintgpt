# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: documents.py
# Description: What a loader produces, and how an operator says which parts of
#   a structured record are content, which are metadata, and which identify it.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Any, Dict, List, Optional, Tuple

# Separates a document's file from the record inside it, so a row keeps a ref
# that survives a re-index: 'data/records.csv#4821'.
REF_SEPARATOR = '#'


# Document class
@dataclass(frozen=True)
class Document:
    '''
    One unit of corpus. A whole file for prose, one record for tabular data.
    '''
    # Stable across re-indexes, so a changed record updates rather than
    # duplicating. Derived from the identity field when one is named.
    ref: str
    # Chunked and embedded. Only this.
    text: str
    # Travels with the document into citations and lexical search, and is never
    # embedded: field names repeat across every record, so embedding them
    # would make every record look alike.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # When the record was created or collected, as the source wrote it.
    # Retrieval filters on this, so it is named rather than left among the
    # other metadata. Parsing is deliberately not done here: 03/04/2026 is two
    # different dates depending on where it was written, and guessing which
    # would be a locale assumption baked into a corpus.
    timestamp: str = ''
    # Who or what produced the record. Matched by exact search alongside the
    # text, because a question about an account has to reach the records it
    # wrote and not only the ones that mention it.
    author: str = ''


# FieldMapping class
@dataclass(frozen=True)
class FieldMapping:
    '''
    Which fields of a structured record play which role. Required for tabular
    and nested sources, because guessing wrong produces a full index that is
    quietly useless.
    '''
    # Joined in order when more than one is named, e.g. a title and a body.
    # The first is the primary: a record whose primary field is empty is not a
    # record, however much the others carry. Without that rule a blank body
    # survives on whatever identifiers sit beside it, and indexes as
    # boilerplate identical to every other blank row.
    content: Tuple[str, ...] = ()
    metadata: Tuple[str, ...] = ()
    # Field carrying when the record was made. Named rather than detected:
    # created_at, captured_at, date and at are the same field, and a heuristic
    # that picks the wrong one breaks every time filter without saying so.
    timestamp: str = ''
    # Field carrying who produced the record.
    author: str = ''
    # Field whose value makes the document ref. Empty falls back to position,
    # which is stable only while the file's row order is.
    identity: str = ''
    # Nested formats only: path to the array of records. Empty means the
    # document is itself the array, or a single record.
    records: str = ''
    # Shortest content worth indexing. A record can satisfy its primary field
    # and still say nothing — a bare handle, a lone emoji — and those embed to
    # near-identical vectors that crowd out real material. Zero keeps
    # everything: where the floor belongs depends on the source.
    min_chars: int = 0

    @classmethod
    def from_dict(cls, data: Optional[dict]):
        '''
        Build a mapping from what a project file recorded.

        Args:
            data (dict, optional): The `fields` table for one source.

        Returns:
            FieldMapping: The mapping, empty when nothing was recorded.
        '''
        data = data or {}

        return cls(
            content=tuple(_as_list(data.get('content'))),
            metadata=tuple(_as_list(data.get('metadata'))),
            timestamp=str(data.get('timestamp') or ''),
            author=str(data.get('author') or ''),
            identity=str(data.get('identity') or ''),
            records=str(data.get('records') or ''),
            min_chars=_as_count(data.get('min_chars'))
        )

    def to_dict(self) -> dict:
        '''
        Returns:
            dict: The mapping as a project file records it, omitting what was \
                not set.
        '''
        recorded: Dict[str, Any] = {}
        if self.content:
            recorded['content'] = list(self.content)
        if self.metadata:
            recorded['metadata'] = list(self.metadata)
        if self.timestamp:
            recorded['timestamp'] = self.timestamp
        if self.author:
            recorded['author'] = self.author
        if self.identity:
            recorded['identity'] = self.identity
        if self.records:
            recorded['records'] = self.records
        if self.min_chars:
            recorded['min_chars'] = self.min_chars

        return recorded

    @property
    def is_set(self) -> bool:
        return bool(self.content)


# read a possibly nested value
def value_at(record: Any, path: str) -> Any:
    '''
    Read a field by dotted path, so nested records address the same way flat
    ones do.

    Args:
        record (Any): A mapping, or something containing one.
        path (str): Field name, or a dotted path like 'user.screen_name'.

    Returns:
        Any: The value, or None when any step of the path is absent.
    '''
    current = record
    for step in path.split('.'):
        if isinstance(current, dict):
            current = current.get(step)
        else:
            return None
        if current is None:
            return None

    return current


# the text one record would contribute
def content_for(record: Any, mapping: FieldMapping) -> str:
    '''
    Join a record's content fields in the order they were named.

    Shared with the dry run, so what an operator is shown is decided by the
    rule that will run.

    Args:
        record (Any): The record, typically a mapping.
        mapping (FieldMapping): Which fields play which role.

    Returns:
        str: The joined text, empty when the primary content field is.
    '''
    if not mapping.content:
        return ''

    primary = _field_text(record, mapping.content[0])
    if not primary:
        return ''

    parts = [primary] + [
        _field_text(record, name) for name in mapping.content[1:]
    ]

    return '\n\n'.join(part for part in parts if part)


# build a document from one structured record
def document_from_record(
    record: Any,
    mapping: FieldMapping,
    source_ref: str,
    position: int
) -> Optional[Document]:
    '''
    Assemble one record into a document.

    Args:
        record (Any): The record, typically a mapping.
        mapping (FieldMapping): Which fields play which role.
        source_ref (str): Path of the file the record came from.
        position (int): Index within the file, used when no identity field is \
            named.

    Returns:
        Optional[Document]: The document, or None when the record has nothing \
            worth searching: no primary content, or less of it than \
            `mapping.min_chars` asks for.
    '''
    text = content_for(record, mapping)
    if not text or len(text) < mapping.min_chars:
        return None

    identifier = ''
    if mapping.identity:
        value = value_at(record, mapping.identity)
        identifier = '' if value is None else str(value).strip()

    metadata = {}
    for name in mapping.metadata:
        value = value_at(record, name)
        if value is not None and str(value).strip():
            metadata[name] = value

    return Document(
        ref=f'{source_ref}{REF_SEPARATOR}{identifier or position}',
        text=text,
        metadata=metadata,
        timestamp=_field_text(record, mapping.timestamp),
        author=_field_text(record, mapping.author)
    )


def _field_text(record: Any, path: str) -> str:
    if not path:
        return ''
    value = value_at(record, path)

    return '' if value is None else str(value).strip()


def _as_count(value: Any) -> int:
    '''
    A recorded count, or zero. A hand-edited project file can hold anything at
    that key, and refusing to load the corpus over it costs more than ignoring
    it.
    '''
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]

    return [str(item) for item in value if str(item).strip()]
