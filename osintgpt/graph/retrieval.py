'''Provider-free graph retrieval with explicit identity, evidence, status, and response bounds.'''

import json
import sqlite3
from dataclasses import asdict, fields

from .state import saved_graph_status
from .status_messages import ABSENCE_NOTE, EMPTY_NOTES, build_messages, query_outcome
from .store import GraphStore
from .traversal import DIRECTIONS, MAX_DEPTH, MAX_EDGES, MAX_NODES, MAX_RESULTS, bounded_integer, endpoint, walk
from .verify import verify_claims

DEFAULT_RESPONSE_BYTES = 24000
MAX_RESPONSE_BYTES = 65536
FALLBACK = ['semantic_search', 'exact_search']


def text_argument(value, name, maximum=300, allow_empty=False):
    if not isinstance(value, str) or len(value) > maximum or not allow_empty and not value.strip():
        raise ValueError(f'{name} must be text from {0 if allow_empty else 1} to {maximum} characters.')
    return value.strip()


def retrieval_status(project, *, allow_disabled=False, saved=None):
    if not project.settings.graph_enabled and not allow_disabled:
        return {'status': 'disabled', 'enabled': False, 'available': False,
                'note': 'Relationship graph retrieval is disabled for this project. Use the other retrieval tools.'}
    try:
        saved = saved if saved is not None else saved_graph_status(project)
    except (OSError, ValueError, sqlite3.Error) as error:
        return {'status': 'unavailable', 'enabled': project.settings.graph_enabled, 'available': False,
                'problem': str(error)[:500], 'note': 'The graph cannot be read. Use semantic or exact search.'}
    incomplete = saved['status'] in ('building', 'interrupted', 'failed', 'complete_with_gaps')
    result = {'status': 'ready' if saved['available'] else 'unbuilt', 'enabled': project.settings.graph_enabled,
              'available': saved['available'], 'build_status': saved['status'],
              'build_started_at': saved.get('started_at', ''), 'build_finished_at': saved.get('finished_at', ''),
              'stale': saved.get('stale', False) or saved['status'] == 'stale', 'incomplete': incomplete}
    result['legacy'] = saved.get('legacy', {})
    messages = build_messages(saved)
    result.update(build_active=messages['active'], build_message=messages['activity'], coverage_notes=messages['coverage'])
    if any(result['legacy'].values()):
        result['coverage_notes'].append('Migrated legacy records need re-extraction for current identity and evidence guarantees. '
                                        'Unknown qualifications must remain unknown.')
    result['note'] = ' '.join([messages['activity'], *result['coverage_notes']])
    if not saved['available']:
        result['note'] += ' Use the other retrieval tools.'
    return result


def _base(project, allow_disabled=False):
    state = retrieval_status(project, allow_disabled=allow_disabled)
    return {'status': state['status'], 'enabled': state['enabled'], 'built': state['available'],
            'graph': state, 'note': state.get('note', ''), 'fallback_tools': FALLBACK,
            'truncated': False, 'truncation_reasons': []}


def candidate(entity):
    resolution = entity.provenance.get('resolution', {})
    origins = [entity.provenance, *entity.provenance.get('identity_sources', [])]
    sources = list({(item.get('document_ref', ''), item.get('source_version', '')): item for item in origins
                    if item.get('document_ref')}.values())
    return {'id': entity.key, 'name': entity.name[:300], 'type': entity.type[:100],
            'resolution': {'action': resolution.get('action', 'unknown'), 'method': resolution.get('method', 'unknown'),
                           'reason': str(resolution.get('reason', ''))[:600],
                           'reviewable': bool(resolution), 'mention_count': len(entity.provenance.get('mentions', []))},
            'aliases': [value[:300] for value in entity.aliases[:12]], 'description': entity.description[:600],
            'identifiers': [{key: str(item.get(key, ''))[:300] for key in ('issuer', 'value')}
                            for item in entity.identifiers[:8]],
            'sources': [{'ref': item['document_ref'][:2000], 'source_version': item.get('source_version', '')}
                        for item in sources[:8]],
            'details_truncated': len(entity.aliases) > 12 or len(entity.identifiers) > 8 or len(sources) > 8}


def payload_size(payload):
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8'))


def fit_response(payload, collection, max_bytes, *, atomic=False):
    rows = payload[collection]
    initial = list(rows)
    payload['response_bytes_limit'] = max_bytes
    while payload_size(payload) > max_bytes and rows:
        rows.pop()
        payload['truncated'] = True
        if 'response_bytes' not in payload['truncation_reasons']:
            payload['truncation_reasons'].append('response_bytes')
        if atomic:
            rows.clear()
            payload.update(connected=None, path_complete=False)
        payload['omitted_results'] = len(initial) - len(rows)
    if len(rows) < len(initial):
        payload['note'] += ' Results exceeded the response budget; narrow the query or fetch an assertion by ID.'
    if payload_size(payload) > max_bytes:
        return {'status': 'response_too_large', 'truncated': True, 'truncation_reasons': ['response_bytes'],
                collection: [], 'note': 'No complete evidence fits this response budget. Narrow the query.',
                'response_bytes_limit': max_bytes, 'fallback_tools': FALLBACK}
    return payload


def source_selection(refs):
    if refs is None:
        return None
    if not isinstance(refs, (list, tuple)):
        raise ValueError('refs must be a list of source references; omit it for all sources.')
    return list(dict.fromkeys(text_argument(ref, 'ref', 2000) for ref in refs))


def discover_entities(project, query, *, limit=20, max_bytes=DEFAULT_RESPONSE_BYTES, allow_disabled=False, refs=None):
    bounded_integer(max_bytes, 'max_bytes', 4096, MAX_RESPONSE_BYTES)
    payload = _discover_entities(project, query, limit=limit, allow_disabled=allow_disabled, refs=refs)
    return fit_response(payload, 'candidates', max_bytes) if 'candidates' in payload else payload


def _discover_entities(project, query, *, limit=20, allow_disabled=False, refs=None):
    query = text_argument(query, 'query')
    refs = source_selection(refs)
    bounded_integer(limit, 'limit', 1, MAX_RESULTS)
    payload = _base(project, allow_disabled)
    if not payload['built']:
        return payload
    try:
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            found, mode = graph.entity_candidates(query, limit + 1)
            candidates = [{**candidate(entity), 'relationships': graph.relationship_counts(entity.key, refs)}
                          for entity in found[:limit]]
    except (OSError, ValueError, sqlite3.Error) as error:
        return _unavailable(payload, error)
    payload.update(status='ambiguous' if len(found) > 1 else 'ok' if found else 'entity_not_found',
                   query=query, match=mode, candidates=candidates,
                   source_scope={'mode': 'all' if refs is None else 'selected', 'refs': refs},
                   truncated=len(found) > limit, truncation_reasons=['candidate_limit'] if len(found) > limit else [])
    if len(found) != 1:
        payload['note'] += ' ' + query_outcome(payload)
    return payload


def _claim_metadata(edge, verification=None, step=None):
    row = {'source_id': endpoint(edge, 'source'), 'target_id': endpoint(edge, 'target')}
    row['source_version'] = edge.provenance.get('source_version', '')
    row['source_ref'] = edge.provenance.get('source_ref', edge.ref)
    row['citation'] = f'{edge.ref} [{edge.assertion_id}]'
    row['verification'] = ({'snapshot': verification.snapshot_status, 'current': verification.status,
                            'source_state': verification.source_state, 'current_version': verification.current_version,
                            'problem': verification.problem[:500], 'snapshot_problem': verification.snapshot_problem[:500]}
                           if verification else {'snapshot': 'not_checked', 'current': 'not_checked'})
    row['read_context'] = {'tool': 'graph_evidence', 'assertion_id': edge.assertion_id, 'quote_index': 0}
    if step:
        row['traversal'] = {'from_id': step.from_id, 'to_id': step.to_id,
                            'against_direction': step.against_direction, 'depth': step.depth}
    return row


def claim(edge, verification=None, step=None):
    return {**asdict(edge), **_claim_metadata(edge, verification, step),
            'evidence_quotes': list(edge.evidence_quotes or (edge.evidence,)),
            'locations': edge.provenance.get('supporting', []) or edge.provenance.get('matches', [])}


def claim_summary(edge, verification=None, step=None):
    '''Share claim metadata without serializing quotations or nested provenance for list views.'''
    summary = {field.name: getattr(edge, field.name) for field in fields(edge)
               if field.name not in ('evidence', 'evidence_quotes', 'provenance')}
    summary.update(_claim_metadata(edge, verification, step),
                   quotation_count=len(edge.evidence_quotes or (edge.evidence,)),
                   quotation_preview=(edge.evidence_quotes or (edge.evidence,))[0][:240])
    return summary


def _unavailable(payload, error):
    payload.update(status='unavailable', note='Graph retrieval failed; use semantic or exact search.',
                   problem=str(error)[:500])
    return payload


def query_graph(project, entity, target=None, *, direction='connectivity', depth=None, limit=20,
                refs=None, max_bytes=DEFAULT_RESPONSE_BYTES, max_nodes=MAX_NODES, max_edges=MAX_EDGES,
                allow_disabled=False):
    bounded_integer(max_bytes, 'max_bytes', 4096, MAX_RESPONSE_BYTES)
    payload = _query_graph(project, entity, target, direction=direction, depth=depth, limit=limit,
                           refs=refs, max_nodes=max_nodes, max_edges=max_edges, allow_disabled=allow_disabled)
    if 'limits' in payload:
        payload['limits']['response_bytes'] = max_bytes
    for collection in ('candidates', 'path', 'claims'):
        if collection in payload:
            return fit_response(payload, collection, max_bytes, atomic=collection == 'path')
    return payload


def select_endpoints(graph, payload, entity, target, refs):
    """Resolve identities once for bounded retrieval and progressive operator exploration."""
    selected = {}
    for label, value in (('entity', entity), ('target', target)):
        if value is None:
            continue
        found, mode = graph.entity_candidates(value, 21)
        if len(found) != 1:
            payload.update(status='ambiguous' if found else 'entity_not_found', endpoint=label,
                           candidates=[candidate(item) for item in found[:20]], connected=False if not found else None,
                           truncated=len(found) > 20, truncation_reasons=['candidate_limit'] if len(found) > 20 else [])
            payload['note'] += ' ' + query_outcome(payload)
            if not found:
                payload['note'] += ' An unmatched identity does not establish that no relationship exists.'
            return None
        selected[label] = {'id': found[0].key, 'name': found[0].name[:300],
                           'relationships': graph.relationship_counts(found[0].key, refs)}
    return selected


def _query_graph(project, entity, target=None, *, direction='connectivity', depth=None, limit=20,
                 refs=None, max_nodes=MAX_NODES, max_edges=MAX_EDGES, allow_disabled=False, row_factory=claim):
    '''Shared identity resolution, traversal, and verification; callers choose delivery format/bounds.'''
    entity = text_argument(entity, 'entity')
    if target is not None:
        target = text_argument(target, 'target')
    depth = depth if depth is not None else MAX_DEPTH if target else 1
    for value, name, high in ((depth, 'depth', MAX_DEPTH), (limit, 'limit', MAX_RESULTS),
                               (max_nodes, 'max_nodes', MAX_NODES), (max_edges, 'max_edges', MAX_EDGES)):
        bounded_integer(value, name, 1, high)
    if direction not in DIRECTIONS:
        raise ValueError('Direction must be incoming, outgoing, or connectivity.')
    refs = source_selection(refs)
    payload = _base(project, allow_disabled)
    if not payload['built']:
        return payload
    payload['limits'] = {'direction': direction, 'depth': depth, 'results': limit,
                         'nodes': max_nodes, 'edges': max_edges}
    payload['source_scope'] = {'mode': 'all' if refs is None else 'selected', 'refs': refs}
    try:
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            selected = select_endpoints(graph, payload, entity, target, refs)
            if selected is None:
                return payload
            result = walk(graph, selected['entity']['id'], goal=selected.get('target', {}).get('id'),
                          direction=direction, depth=depth, limit=limit, refs=refs,
                          max_nodes=max_nodes, max_edges=max_edges)
            steps = result.path.steps if result.path is not None else [] if target else result.hits
            verified = verify_claims(project, [step.edge for step in steps], first_only=True, require_registered=True)
    except (OSError, ValueError, sqlite3.Error) as error:
        return _unavailable(payload, error)
    payload.update(status='ok', selected=selected, truncated=result.truncated,
                   truncation_reasons=result.reasons, nodes_visited=result.nodes_visited, edges_examined=result.edges_examined)
    collection = 'path' if target else 'claims'
    payload[collection] = [row_factory(step.edge, evidence, step) for step, evidence in zip(steps, verified.results)]
    if target:
        found = result.path is not None
        payload.update(connected=found, path_found=found, path_complete=found,
                       path_assertion_ids=[step.edge.assertion_id for step in steps])
        if not found:
            payload.update(status='no_connection', note=payload['note'] + ' ' + ABSENCE_NOTE)
    elif not steps:
        counts = selected['entity']['relationships']
        reason = ('empty_source_selection' if refs == [] else 'isolated_identity' if not counts['all_sources']
                  else 'source_filter' if not counts['selected_sources'] else 'direction_or_limits')
        payload.update(status='no_claims', empty_reason=reason, note=payload['note'] + ' ' + EMPTY_NOTES[reason])
    if target and result.path is not None and not steps:
        payload['note'] += ' Both endpoints resolve to the same identity: a zero-step connection.'
    payload['note'] += ' Quotations verify textual support, not truth. Keep each assertion\'s direction and qualifications.'
    return payload
