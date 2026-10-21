'''Immutable extracted text shared by indexing, graph extraction, and evidence reads.'''

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from uuid import uuid4

from .documents import Document, FieldMapping
from .loaders import load_documents
from .provenance import TextSpan, quote_locations, text_spans
from .transcription import cached_transcriber, cache_path

EXTRACTION_VERSION = 1


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def source_fingerprint(path, mapping):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    digest.update(json.dumps(mapping.to_dict() if mapping else {}, sort_keys=True).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class TextSnapshot:
    ref: str
    fingerprint: str
    version: str
    documents: tuple[Document, ...]
    statistics: dict = field(default_factory=dict)
    transcriptions: dict = field(default_factory=dict)

    @cached_property
    def missing_pages(self):
        return sorted({span.page for document in self.documents for span in document.spans
                       if span.missing and span.page is not None})

    @cached_property
    def by_ref(self):
        return {document.ref: document for document in self.documents}

    def document(self, ref):
        return self.by_ref.get(ref)

    def provenance(self, document, quote, *, after=0, first_only=False):
        matches = quote_locations(document, quote, after=after, first_only=first_only)
        return {'source_ref': self.ref, 'document_ref': document.ref,
                'source_version': self.version, 'source_fingerprint': self.fingerprint,
                'extraction_version': EXTRACTION_VERSION, 'matches': matches,
                'offset_basis': 'extracted_record_text', 'offset_unit': 'unicode_characters',
                'match_status': 'found' if matches else 'not_found',
                'missing_pages': self.missing_pages}


def _payload(snapshot):
    return {'ref': snapshot.ref, 'fingerprint': snapshot.fingerprint,
            'extraction_version': EXTRACTION_VERSION,
            'documents': [asdict(document) for document in snapshot.documents],
            'statistics': snapshot.statistics, 'transcriptions': snapshot.transcriptions}


def _decode(data, version):
    if data.get('extraction_version') != EXTRACTION_VERSION or _digest(data) != version:
        raise ValueError('Extracted-text snapshot is incompatible or damaged.')
    documents = tuple(Document(**{**item, 'spans': tuple(TextSpan(**span)
                       for span in item['spans'])}) for item in data['documents'])
    return TextSnapshot(data['ref'], data['fingerprint'], version, documents,
                        data['statistics'], data['transcriptions'])


def transcriptions_current(snapshot, cache_dir):
    for name, signature in snapshot.transcriptions.items():
        if not re.fullmatch('[a-f0-9]{64}\\.md', name):
            return False
        try:
            if hashlib.sha256((Path(cache_dir) / name).read_bytes()).hexdigest() != signature:
                return False
        except OSError:
            return False
    return True


def read_snapshot(cache_dir, version):
    if not re.fullmatch('[a-f0-9]{64}', version):
        raise ValueError('Invalid extracted-text version.')
    path = Path(cache_dir) / 'text' / f'{version}.json'
    return _decode(json.loads(path.read_text(encoding='utf-8')), version)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_snapshot(path, mapping=None, *, ref=None, cache_dir=None, transcriber=None,
                  statistics=None):
    path = Path(path)
    mapping = mapping or FieldMapping()
    ref = ref or path.as_posix()
    fingerprint = source_fingerprint(path, mapping)
    key = _digest([ref, fingerprint, EXTRACTION_VERSION])
    pointer = Path(cache_dir) / 'text' / f'current-{key}.json' if cache_dir else None
    if pointer and pointer.is_file():
        try:
            snapshot = read_snapshot(cache_dir, json.loads(pointer.read_text(encoding='utf-8')))
            if (snapshot.ref == ref and snapshot.fingerprint == fingerprint
                    and not snapshot.missing_pages and transcriptions_current(snapshot, cache_dir)):
                if statistics is not None:
                    statistics.update(snapshot.statistics)
                return snapshot
        except (OSError, ValueError, KeyError, TypeError):
            pass

    counts, transcriptions = {}, {}
    reader = transcriber or (cached_transcriber(cache_dir) if cache_dir else None)

    def tracked_reader(image, page):
        text = reader(image, page)
        if cache_dir:
            cached = cache_path(cache_dir, image)
            try:
                transcriptions[cached.name] = hashlib.sha256(cached.read_bytes()).hexdigest()
            except OSError:
                pass
        return text

    documents = load_documents(path, mapping, tracked_reader if reader else None, statistics=counts)
    if source_fingerprint(path, mapping) != fingerprint:
        raise ValueError('Source changed during text extraction; retry with the current file.')
    normalized = []
    for document in documents:
        document_ref = document.ref
        if document_ref == path.as_posix() or document_ref.startswith(path.as_posix() + '#'):
            document_ref = ref + document_ref[len(path.as_posix()):]
        normalized.append(replace(document, ref=document_ref,
                                  metadata=json.loads(json.dumps(document.metadata, default=str)),
                                  spans=document.spans or text_spans(document.text)))
    if len({document.ref for document in normalized}) != len(normalized):
        raise ValueError('Duplicate record identities; choose a unique identity field in the source mapping.')
    snapshot = TextSnapshot(ref, fingerprint, '', tuple(normalized), counts, transcriptions)
    snapshot = replace(snapshot, version=_digest(_payload(snapshot)))
    if pointer:
        _write(pointer.parent / f'{snapshot.version}.json', _payload(snapshot))
        _write(pointer, snapshot.version)
    if statistics is not None:
        statistics.update(counts)
    return snapshot
