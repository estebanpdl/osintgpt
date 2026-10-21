'''Export one operator query selection using its existing evidence and revision.'''

from .export_snapshot import query_export_context, selection_package
from .verification_cache import VerificationChanged


def prepare_selection(query, *, scope='all_matching', filters=None, assertion_ids=None, display=None, request=None):
    '''Caller owns the query lock; reject changes before and after serialization input.'''
    if not query.check_current():
        raise VerificationChanged(query.result['problem'])
    filters = dict(filters or {})
    filters.update(sources=query.result['source_scope']['refs'],
                   direction=query.result['limits']['direction'])
    matching = query.matching(text=filters.get('text', ''), relations=filters.get('relations', ()),
                              verification=filters.get('verification', 'All'))
    matching_ids = [row['assertion_id'] for row in matching]
    if scope not in ('all_matching', 'table_page', 'diagram'):
        raise ValueError('Unknown operator export scope.')
    selected_ids = matching_ids if scope == 'all_matching' else list(assertion_ids or [])
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(matching_ids):
        raise ValueError('Displayed assertions must be unique matches from this query.')
    query._refresh()
    result = query.selected_result(selected_ids)
    context = query_export_context(query.project, result, request=request)
    package = selection_package(query.project, result, context=context,
                                filters=filters)
    if 'path' in query.result and selected_ids != [row['assertion_id'] for row in query.rows]:
        package['metadata']['path_complete'] = False
    complete = query.state == 'complete'
    package['metadata'].update(
        scope=scope, verification_basis='query_snapshot',
        selection={'scope': scope, 'assertion_ids': selected_ids, 'exported_assertions': len(selected_ids),
                   'matching_discovered': len(matching_ids), 'matching_total': len(matching_ids) if complete else None,
                   'query_complete': complete, 'all_matching_exported': selected_ids == matching_ids,
                   'complete_matching_scope': complete and selected_ids == matching_ids,
                   'display': dict(display or {}) if scope != 'all_matching' else {}})
    if not query.check_current():
        raise VerificationChanged(query.result['problem'])
    return package
