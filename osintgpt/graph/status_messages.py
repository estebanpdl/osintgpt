'''Separate checked build activity, published coverage, and individual query outcomes.'''

ABSENCE_NOTE = 'No connection was found within this query and its limits. That is not the same as there being none.'
EMPTY_NOTES = {
    'empty_source_selection': 'Select at least one source or choose All sources.',
    'isolated_identity': 'This identity has no accepted relationships in the published graph.',
    'source_filter': 'This identity has relationships, but none are in the selected sources.',
    'direction_or_limits': 'No assertions found in this direction and traversal limits; try connectivity.',
}


def count_label(counts, key):
    value = counts.get(key)
    return str(value) if type(value) is int and value >= 0 else 'unknown'


def build_messages(saved, *, job=None):
    '''Use status from saved_graph_status/GraphState.read, which checks the build lock.

    Optional process-local job activity covers startup/finalization before/after
    the persisted building state; a finished job does not imply current activity.
    '''
    status = saved.get('status', 'unknown')
    active = status == 'building' or bool(job and job.get('active'))
    if job and job.get('active') and job.get('status') == 'starting':
        activity = 'A graph build is starting.'
    elif active and status != 'building':
        activity = 'A graph build is finishing.'
    else:
        activity = {
            'building': 'A graph build is running.',
            'interrupted': 'The last graph build was interrupted.',
            'failed': 'The last graph build failed.',
            'complete_with_gaps': 'The last graph build finished with gaps.',
            'complete': 'The last graph build finished.',
            'stale': 'The last graph build finished.',
            'never_built': 'No graph build has completed yet.',
        }.get(status, 'The graph build outcome is unknown.')
    coverage = []
    if not saved.get('available'):
        coverage.append('No published graph is available yet.')
    elif active or status in ('interrupted', 'failed', 'complete_with_gaps'):
        coverage.append('Published claims remain available, but coverage may be incomplete.')
    if saved.get('available') and (saved.get('stale') or status == 'stale'):
        coverage.append('The published graph is stale: current sources or pending extraction settings differ '
                        'from its saved versions. Updating it requires an explicit build.')
    failed = count_label(saved.get('counts', {}), 'documents_failed')
    if failed not in ('unknown', '0'):
        coverage.append(f"{failed} source document(s) failed in {'this' if status == 'building' else 'the last'} build.")
    return {'active': active, 'activity': activity, 'coverage': coverage}


def query_outcome(result):
    '''An individual query outcome, independent of the last build's coverage.'''
    status = result.get('status')
    if status == 'entity_not_found':
        endpoint = {'entity': 'starting ', 'target': 'destination '}.get(result.get('endpoint'), '')
        return f'No matching {endpoint}identity was found in the published graph.'
    if status == 'no_claims':
        return EMPTY_NOTES.get(result.get('empty_reason'), 'No assertions were returned for this query.')
    if status == 'ambiguous':
        return 'Several identities match. Select an ID using source context.'
    if status == 'no_connection':
        return ABSENCE_NOTE
    if status == 'unbuilt':
        return 'No published graph is available to search.'
    if status == 'disabled':
        return 'Relationship graph retrieval is disabled for this project.'
    if status == 'unavailable':
        return 'The graph could not be read. Check graph status for details.'
    if status == 'response_too_large':
        return 'No complete result fits this response budget. Narrow the query.'
    if status == 'ok' and result.get('path_complete') and not result.get('path_assertion_ids'):
        return 'Both endpoints resolve to the same identity: a zero-step connection.'
    return ''
