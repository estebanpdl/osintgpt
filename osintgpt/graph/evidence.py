'''Bounded source-context reads anchored to a stored assertion and its exact text version.'''

import sqlite3

from osintgpt.ingestion import Corpus, Document
from osintgpt.ingestion.provenance import locations, quote_locations, text_spans
from osintgpt.ingestion.snapshots import read_snapshot
from osintgpt.ingestion.source_view import read_source

from .reconciliation import registered
from .retrieval import DEFAULT_RESPONSE_BYTES, MAX_RESPONSE_BYTES, _base, _unavailable, claim, payload_size, text_argument
from .store import GraphStore
from .traversal import bounded_integer
from .verify import verify_claims


def assertion_context(project, assertion_id, *, quote_index=0, occurrence=0, context_chars=800,
                      offset=0, limit_chars=4000, max_bytes=DEFAULT_RESPONSE_BYTES, allow_disabled=False):
    assertion_id = text_argument(assertion_id, 'assertion_id', 100)
    bounded_integer(quote_index, 'quote_index', 0, 199)
    bounded_integer(occurrence, 'occurrence', 0, 10000)
    bounded_integer(context_chars, 'context_chars', 0, 4000)
    bounded_integer(offset, 'offset', 0, 1_000_000)
    bounded_integer(limit_chars, 'limit_chars', 1, 8000)
    bounded_integer(max_bytes, 'max_bytes', 4096, MAX_RESPONSE_BYTES)
    payload = _base(project, allow_disabled)
    if not payload['built']:
        return payload
    try:
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            edge = graph.assertion(assertion_id)
        if edge is None:
            payload.update(status='assertion_not_found', note='This assertion is no longer in the published graph. Query again.')
            return payload
        quotes = edge.evidence_quotes or (edge.evidence,)
        if quote_index >= len(quotes):
            raise ValueError('quote_index is outside this assertion\'s supporting quotations.')
        verified = verify_claims(project, [edge], first_only=True, require_registered=True).results[0]
        full = claim(edge, verified)
        payload.update({key: full[key] for key in (
            'assertion_id', 'source', 'relation', 'target', 'source_id', 'target_id', 'ref', 'citation',
            'source_ref', 'source_version', 'relation_polarity', 'assertion', 'attributed_to', 'event_time',
            'statement_time', 'verification')})
        payload['identity_decisions'] = edge.provenance.get('identity_decisions', {})
        document = _document(project, edge)
        matches = _matches(document, edge, quote_index, quotes[quote_index])
        if occurrence >= len(matches):
            raise ValueError('The selected quotation occurrence is unavailable in this text version.')
        match = matches[occurrence]
        start, end = max(0, match['start'] - context_chars), min(len(document.text), match['end'] + context_chars)
        if offset > end - start:
            raise ValueError('offset is outside this evidence context.')
        left, right = start + offset, min(end, start + offset + limit_chars)
        payload.update(status='ok', quote_index=quote_index, quote_count=len(quotes), occurrence=occurrence,
                       occurrences=len(matches), quote_start=match['start'], quote_end=match['end'],
                       context_start=start, context_end=end, offset=offset,
                       text_basis='saved_snapshot' if edge.provenance.get('source_version') else 'current_source',
                       response_bytes_limit=max_bytes)
        byte_limited = False
        while True:
            payload.update(text=document.text[left:right], start=left, end=right,
                           locations=locations(document.spans, left, right),
                           quote_complete=left <= match['start'] and right >= match['end'],
                           truncated=right < end or left > start,
                           truncation_reasons=(['context_window'] if right < end or left > start else [])
                           + (['response_bytes'] if byte_limited else []))
            if right < end:
                payload['next_offset'] = right - start
            else:
                payload.pop('next_offset', None)
            if payload_size(payload) <= max_bytes and (right > left or left == end):
                return payload
            if right <= left:
                return {'status': 'response_too_large', 'assertion_id': assertion_id, 'text': '', 'truncated': True,
                        'truncation_reasons': ['response_bytes'], 'note': 'Assertion metadata exceeds the response budget.'}
            right = left + (right - left) // 2
            byte_limited = True
    except (OSError, ValueError, sqlite3.Error) as error:
        payload = _unavailable(payload, error)
        payload['status'] = 'evidence_unavailable'
        return payload


def _document(project, edge):
    source_ref = edge.provenance.get('source_ref', edge.ref)
    version = edge.provenance.get('source_version', '')
    if version:
        snapshot = read_snapshot(project.paths.extracts, version)
        if snapshot.ref != source_ref:
            raise ValueError('Saved text belongs to a different source.')
        document = snapshot.document(edge.ref)
        if document is None:
            raise ValueError('Record does not exist in the saved text version.')
        return document
    corpus = Corpus.load(project.paths.sources)
    if not registered(corpus, project.paths.root, source_ref):
        raise ValueError('Source is no longer registered and the assertion has no saved text version.')
    text, _ = read_source(project, project.paths.root / source_ref, edge.ref)
    return Document(edge.ref, text, spans=text_spans(text))


def _matches(document, edge, index, quote):
    supporting = edge.provenance.get('supporting', [])
    saved = (supporting[index].get('matches', []) if index < len(supporting)
             else edge.provenance.get('matches', []) if index == 0 else [])
    if saved:
        return [located for match in saved for located in quote_locations(
            document, quote, after=match['start'], before=match['end'], first_only=True)]
    return quote_locations(document, quote)
