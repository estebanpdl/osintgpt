'''Independent table pages and explicit diagram subsets over discovered assertions.'''

import math

from osintgpt.graph.inspection import relationship_rows

from . import graph_chart, graph_evidence, graph_exports


def render(st, project, state, query, context, prefix):
    result, claims = query.result, query.rows
    if state.get(prefix + 'diagram-query') != query.token:
        state[prefix + 'diagram-chosen'] = []
        state[prefix + 'diagram-query'] = query.token
    if result.get('path_complete'):
        st.caption(f'Complete supporting path: {len(claims)} steps. Table pages and diagram subsets do not change that chain.')
    relations = st.multiselect('Relationship filter', sorted({row['relation'] for row in claims}), key=prefix + 'relations')
    verification = st.selectbox('Current quotation verification', ['All', 'found', 'not_found', 'unreadable'], key=prefix + 'verification')
    st.caption(f'Evidence checks: {len(claims)} discovered assertions checked against current source text. '
               + ('Checks cover the completed query scope.' if query.state == 'complete' else
                  'Exploration is unfinished; assertions not yet discovered have not been checked and their total is unknown.')
               + ' Unreadable means the source could not be checked; it does not mean the quotation was absent.')
    search = st.text_input('Search relationships', key=prefix + 'search')
    filtered = query.matching(text=search, relations=relations, verification=verification)
    page_size = st.selectbox('Rows per page', [10, 30, 60, 100], index=1, key=prefix + 'page-size')
    page_key = prefix + 'page'
    filter_key = (query.token, tuple(relations), verification, search, page_size)
    if state.get(prefix + 'page-filter') != filter_key:
        state[page_key] = 1
        state[prefix + 'page-filter'] = filter_key
    pages = max(1, math.ceil(len(filtered) / page_size))
    state[page_key] = min(max(1, state.get(page_key, 1)), pages)
    left, middle, right = st.columns([1, 2, 1], vertical_alignment='bottom')
    if left.button('Previous page', disabled=state[page_key] <= 1, key=prefix + 'previous', width='stretch'):
        state[page_key] -= 1
    if right.button('Next page', disabled=state[page_key] >= pages, key=prefix + 'next', width='stretch'):
        state[page_key] += 1
    page = middle.number_input('Table page', min_value=1, max_value=pages, step=1, key=page_key)
    offset = (page - 1) * page_size
    visible = filtered[offset:offset + page_size]
    st.caption(f'Table page {page} of {pages}: assertions {offset + 1 if visible else 0}–{offset + len(visible)} '
               f'of {len(filtered)} filter matches among {len(claims)} discovered assertions. '
               + ('Exploration is unfinished; these are results found so far, not a final total.'
                  if query.state != 'complete' else 'Exploration is complete for the selected scope.'))
    st.caption('Order: connection steps from start to destination.' if 'path' in result else
               'Order: traversal discovery, nearer steps first, then saved database order within each visited identity. '
               'Row numbers are positions in these results, not importance, confidence, or date rankings.')
    st.caption('Quotation previews are shortened for the table. Select an assertion for complete evidence. '
               'Quotation searches and prepared exports use the full saved text. No response-size limit removes assertions here.')
    def render_exports(displayed):
        graph_exports.render(st, project, state, result=result, context=context, filtered=filtered, inspection=query,
                             visible=visible, displayed=displayed,
                             display={'table_page': page, 'rows_per_page': page_size, 'diagram_selection': mode},
                             filters={'relations': relations, 'verification': verification, 'text': search,
                                      'sources': result['source_scope']['refs'], 'direction': result['limits']['direction']})
    if not state.get(prefix + 'page-diagram-default'):
        state[prefix + 'diagram-mode'] = 'Current table page'
        state[prefix + 'page-diagram-default'] = True
    mode = st.selectbox('Diagram selection', ['Current table page', 'First matching assertions', 'Choose assertions', 'Hidden'],
                        key=prefix + 'diagram-mode',
                        help='Current table page follows Previous/Next page. Other selections keep the diagram independent of table navigation.')
    st.caption('The diagram follows the current table page.' if mode == 'Current table page' else
               'Table navigation does not change this diagram selection. Choose Current table page to make them follow each other.')
    if not filtered:
        render_exports([])
        st.caption('Diagram: 0 matching discovered assertions. Table page: 0 assertions.')
        return
    ids, page_ids = [row['assertion_id'] for row in filtered], [row['assertion_id'] for row in visible]
    inspect_key, choice_key = prefix + 'inspect', prefix + 'page-choice'
    if state.get(inspect_key) not in ids:
        state[inspect_key] = ids[0]

    def select_assertion(value):
        state[inspect_key] = value
        state[choice_key] = value if value in page_ids else None

    if mode == 'First matching assertions':
        count_key = prefix + 'diagram-count'
        state[count_key] = min(len(filtered), max(1, state.get(count_key, 30)))
        if st.button('Show more assertions in diagram', disabled=state[count_key] >= len(filtered), key=prefix + 'diagram-more'):
            state[count_key] = min(len(filtered), state[count_key] + 30)
        count = st.number_input('Assertions in diagram', min_value=1, max_value=len(filtered), step=1, key=count_key,
                               help='Changes only the diagram. Table pages retain all discovered assertions.')
        displayed = filtered[:count]
    elif mode == 'Current table page':
        displayed = visible
    elif mode == 'Choose assertions':
        # Select on the current page so a huge query does not become a huge dropdown.
        known = {row['assertion_id'] for row in claims}
        keep = [key for key in state.get(prefix + 'diagram-chosen', []) if key in known]
        selected = st.multiselect('Assertions from this table page', page_ids,
                                 default=[key for key in keep if key in page_ids],
                                 key=prefix + f'diagram-picks:{query.token}:{page}:{filter_key[1:]}',
                                 format_func=lambda value: next(f"{row['source']} — {row['relation']} → {row['target']} · {value[-10:]}"
                                                               for row in visible if row['assertion_id'] == value))
        chosen = set(key for key in keep if key not in page_ids) | set(selected)
        state[prefix + 'diagram-chosen'] = [row['assertion_id'] for row in claims if row['assertion_id'] in chosen]
        displayed = [row for row in filtered if row['assertion_id'] in chosen]
        if hidden := len(chosen - set(ids)):
            st.caption(f'{hidden} chosen assertions are outside the current display filters; their selection is retained.')
    else:
        displayed = []
    st.caption(f'Diagram: {len(displayed)} of {len(filtered)} matching discovered assertions. '
               f'Table page: {len(visible)} assertions. Diagram selection does not limit exploration or table access.')
    render_exports(displayed)
    if displayed:
        graph_chart.render(st, displayed, prefix + 'chart', all_claims=claims,
                           starting_id=result['selected']['entity']['id'], scope=query.token,
                           selected_id=state[inspect_key], on_select=select_assertion)
    elif mode != 'Hidden':
        st.info('Select assertions to display a diagram.')

    def select_table():
        rows = state[prefix + 'table'].get('selection', {}).get('rows', [])
        if rows and 0 <= rows[0] < len(page_ids):
            select_assertion(page_ids[rows[0]])

    table_rows = relationship_rows(visible)
    if 'path' in result:
        table_rows = [{'Path step': row['traversal']['depth'], **item} for row, item in zip(visible, table_rows)]
    st.dataframe(table_rows, hide_index=True, width='stretch', key=prefix + 'table',
                 on_select=select_table, selection_mode='single-row')
    state[choice_key] = state[inspect_key] if state[inspect_key] in page_ids else None

    def select_page():
        if state[choice_key] is not None:
            state[inspect_key] = state[choice_key]

    by_id = {row['assertion_id']: row for row in visible}
    st.selectbox('Inspect assertion', page_ids, index=None, key=choice_key, on_change=select_page,
                 format_func=lambda value: f"{by_id[value]['source']} — {by_id[value]['relation']} → "
                                           f"{by_id[value]['target']} · {by_id[value]['ref']} · {value[-10:]}")
    if state[inspect_key] not in page_ids:
        st.caption('The selected assertion is outside this table page; its evidence remains open below.')
    graph_evidence.render(st, project, state[inspect_key], state, inspection=query)
