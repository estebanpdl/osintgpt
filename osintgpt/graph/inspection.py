'''Read-only graph coverage and assertion inspection for operator interfaces.'''

import sqlite3
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from osintgpt.ingestion.snapshots import read_snapshot, source_fingerprint, transcriptions_current

from .reconciliation import inventory
from .retrieval import _base, _discover_entities, _query_graph, _unavailable, claim, claim_summary
from .state import saved_graph_status
from .store import GraphStore
from .traversal import bounded_integer, endpoint
from .verify import verify_claims


def overview(project):
    status = saved_graph_status(project)
    path = project.paths.root / 'graph.sqlite'
    with GraphStore(path, read_only=True) if path.is_file() else GraphStore(':memory:') as graph:
        corpus, targets, missing, removed, images = inventory(project, graph)
        completed = graph.state.entries()
        mappings = corpus.mappings(project.paths.root)
        rows = []
        for ref, path in targets.items():
            previous = completed.get(ref)
            row = {'source': ref, 'state': 'new', 'published': bool(previous), 'gaps': False}
            if previous:
                row.update(version=previous['source_version'], gaps=bool(previous['quality_gaps']))
                try:
                    unchanged = source_fingerprint(path, mappings.get(path.resolve())) == previous['fingerprint']
                    snapshot = read_snapshot(project.paths.extracts, previous['source_version']) if previous['source_version'] else None
                    row['state'] = ('current' if unchanged and (snapshot is None or
                                    transcriptions_current(snapshot, project.paths.extracts)) else 'changed')
                    row['missing_pages'] = snapshot.missing_pages if snapshot else []
                except (OSError, ValueError) as error:
                    row.update(state='unavailable', problem=str(error))
            rows.append(row)
        rows += [{'source': ref, 'state': label, 'published': ref in completed,
                  'gaps': bool(completed.get(ref, {}).get('quality_gaps'))}
                 for label, refs in [('unavailable', missing), ('unregistered', removed)] for ref in refs]
    return {'status': status, 'sources': rows, 'images_skipped': images,
            'coverage': {'registered': len(targets) + len(missing),
                         'current': sum(row['state'] == 'current' for row in rows),
                         'with_gaps': sum(row.get('gaps', False) for row in rows)}}


def inspect_assertion(project, assertion_id, *, limit=30, verification_cache=None):
    bounded_integer(limit, 'limit', 1, 60)
    result = _base(project, allow_disabled=True)
    if not result['built']:
        return result
    try:
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            selected = graph.assertion(assertion_id)
            if selected is None:
                return {**result, 'status': 'assertion_not_found', 'note': 'This assertion was replaced. Refresh the query.'}
            related = graph.between(endpoint(selected, 'source'), endpoint(selected, 'target'), limit=limit + 1)
        others = [edge for edge in related if edge.assertion_id != assertion_id]
        edges = [selected, *others[:limit - 1]]
        verified = verify_claims(project, edges, first_only=True, require_registered=True, cache=verification_cache)
        result.update(status='ok', claims=[claim(edge, evidence) for edge, evidence in zip(edges, verified.results)],
                      truncated=len(others) >= limit,
                      truncation_reasons=['result_limit'] if len(others) >= limit else [])
        return result
    except (OSError, ValueError, sqlite3.Error) as error:
        return _unavailable(result, error)


def filter_claims(claims, *, text='', relations=(), verification='All', quote_reader=None):
    needle = text.casefold().strip()
    return [row for row in claims if (not relations or row['relation'] in relations)
            and (verification == 'All' or row['verification']['current'] == verification)
            and (not needle or needle in ' '.join([*(str(row.get(key, '')) for key in
                 ('source', 'target', 'relation', 'ref', 'assertion_id')),
                 str(quote_reader(row) if quote_reader else row.get('evidence_quotes', []))]).casefold())]


@dataclass
class OperatorQuery:
    '''Local query snapshot: paged summaries with full evidence serialized only on request.

    Retain the already-read edges privately so pages, quote searches, and exports cannot
    silently mix in replacement assertions. Traversal work limits are still enforced.
    '''
    result: dict
    _edges: dict = field(repr=False)
    _summaries: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._summaries = {row['assertion_id']: row for row in self.rows}

    def append_summary(self, row):
        self.rows.append(row)
        self._summaries[row['assertion_id']] = row

    @property
    def rows(self):
        return self.result.get('path', self.result.get('claims', []))

    def page(self, *, offset=0, page_size=30):
        for value, name, low in ((offset, 'offset', 0), (page_size, 'page_size', 1)):
            if type(value) is not int or value < low:
                raise ValueError(f'{name} must be an integer of at least {low}.')
        rows = self.rows[offset:offset + page_size]
        end = offset + len(rows)
        return {'rows': deepcopy(rows), 'offset': offset, 'total_returned': len(self.rows),
                'next_offset': end if end < len(self.rows) else None}

    def details(self, assertion_id):
        summary = self._summaries.get(assertion_id)
        if summary is None:
            raise ValueError('Assertion is not part of this query result.')
        row = claim(self._edges[assertion_id])
        row['verification'] = deepcopy(summary['verification'])
        if 'traversal' in summary:
            row['traversal'] = deepcopy(summary['traversal'])
        # claim() already copies the dataclass; locations alone reference its source.
        row['locations'] = deepcopy(row['locations'])
        return row

    def full_result(self):
        return self.selected_result([row['assertion_id'] for row in self.rows])

    def selected_result(self, assertion_ids):
        '''Full query metadata with only requested assertions expanded into evidence.'''
        result = deepcopy({key: value for key, value in self.result.items()
                           if key not in ('delivery', 'claims', 'path')})
        collection = 'path' if 'path' in self.result else 'claims'
        if collection in self.result:
            result[collection] = [self.details(key) for key in assertion_ids]
        return result

    def matching(self, **filters):
        def quotes(row):
            edge = self._edges[row['assertion_id']]
            return list(edge.evidence_quotes or (edge.evidence,))
        return filter_claims(self.rows, quote_reader=quotes, **filters)


def inspect_graph(project, entity, target=None, **options):
    '''Operator delivery over the same graph engine, without model response-byte pruning.'''
    edges = {}

    def summarize(edge, verification, step):
        edges[edge.assertion_id] = edge
        return claim_summary(edge, verification, step)

    result = operator_query_result(_query_graph(project, entity, target, allow_disabled=True,
                                               row_factory=summarize, **options))
    result['delivery'] = 'operator_summaries_v1'
    return OperatorQuery(result, edges)


def inspect_entities(project, query, *, limit=30, refs=None):
    return _discover_entities(project, query, limit=limit, refs=refs, allow_disabled=True)


def operator_query_result(result):
    '''Describe operator scope without changing the bounded model-tool response contract.'''
    reasons = [reason for reason in result.get('truncation_reasons', []) if reason != 'depth_boundary']
    collection = 'path' if 'path' in result else 'claims'
    rows = result.get(collection, [])
    known = collection in result and result.get('status') in ('ok', 'no_claims', 'no_connection')
    complete = known and not any(reason != 'response_bytes' for reason in reasons)
    start = result.get('selected', {}).get('entity', {}).get('id')
    direct = sum(start in (row['source_id'], row['target_id']) for row in rows) if start else None
    return {**result, 'truncated': bool(reasons) or bool(result.get('truncated') and not result.get('truncation_reasons')),
            'truncation_reasons': reasons,
            'inspection': {'kind': collection, 'discovery_complete': complete,
                           'assertions_found': len(rows) + result.get('omitted_results', 0) if known else None,
                           'assertions_returned': len(rows), 'direct_assertions_returned': direct,
                           'depth_boundary_reached': 'depth_boundary' in result.get('truncation_reasons', [])}}


def query_messages(result):
    captions, warnings = [], []
    summary, limits = result['inspection'], result.get('limits', {})
    if limits:
        scope = f"up to {limits['depth']} relationship steps ({limits['direction']})"
        if summary['kind'] == 'path':
            captions.append(f'Connection search scope: {scope}.')
        elif summary['discovery_complete']:
            captions.append(f'Exploration complete within the selected scope: {scope}.')
        else:
            captions.append(f'Exploration scope: {scope}; complete results for this scope are not available.')
    reasons = {'response_bytes': 'The response size limit omitted assertions from the displayed result.',
               'result_limit': 'The maximum assertion count stopped exploration before the full scope was examined.',
               'node_limit': 'The visited-identity limit stopped exploration before the full scope was examined.',
               'edge_limit': 'The examined-assertion limit stopped exploration before the full scope was examined.'}
    progress = result.get('progress', {})
    if progress:
        state = progress['state']
        if state in ('running', 'paused'):
            captions.append(f'Exploration {state}: {result.get("nodes_visited", 0)} identities visited; '
                            f'{result.get("edges_examined", 0)} adjacency entries examined. '
                            'Results found so far are available; the total is unknown.')
        elif state in ('invalidated', 'failed'):
            warnings.append(result.get('problem', 'Exploration could not continue. Run a new query.'))
    if result.get('truncated') and not progress:
        messages = [reasons.get(reason, f'Results are incomplete: {reason}.')
                    for reason in result['truncation_reasons']]
        warnings.append(' '.join(messages) or 'The query returned incomplete results.')
    found = summary['assertions_found']
    if found is not None:
        label = 'Path' if summary['kind'] == 'path' else 'Neighborhood'
        extent = '' if summary['discovery_complete'] else ' so far; total unknown'
        captions.append(f"{label}: {found} assertions found{extent}; {summary['assertions_returned']} returned.")
        if summary['kind'] == 'claims' and summary['direct_assertions_returned'] is not None:
            direct = summary['direct_assertions_returned']
            captions.append(f"Returned neighborhood: {direct} assertions involve the selected identity directly; "
                       f"{summary['assertions_returned'] - direct} connect other identities reached during traversal.")

    return captions, warnings


def relationship_rows(claims):
    return [{'Source': row['source'], 'Relationship': row['relation'],
             'Target': row['target'], 'Relation polarity': row['relation_polarity'], 'Status': row['assertion'],
             'Document': row['ref'], 'Current evidence': row['verification']['current'],
             'Saved evidence': row['verification']['snapshot'], 'Event date': row['event_time'],
             ('Quotation preview' if 'quotation_preview' in row else 'Quotation'):
                 row.get('quotation_preview', '\n'.join(row.get('evidence_quotes', []))),
             'Assertion ID': row['assertion_id']} for row in claims]


def rejected_proposals(project, *, limit=100):
    bounded_integer(limit, 'limit', 1, 100)
    with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
        return [asdict(item) for item in graph.rejections(limit=limit)]
