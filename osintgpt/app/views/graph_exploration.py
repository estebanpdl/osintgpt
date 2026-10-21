'''Progressive operator exploration, driven by bounded Streamlit fragment ticks.'''

from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.export_snapshot import query_export_context
from osintgpt.graph.inspection import query_messages
from osintgpt.graph.status_messages import query_outcome
from osintgpt.graph.verification_cache import VerificationChanged

from . import graph_results


def render(st, project, state, sources, revision, picker):
    prefix = f'graph-explore:{project.id}:'
    query = state.get(prefix + 'inspection')
    running = isinstance(query, OperatorExploration) and query.state == 'running'

    @st.fragment(run_every=.2 if running else None)
    def panel():
        _render(st, project, state, sources, revision, picker, prefix)

    panel()


def _render(st, project, state, sources, revision, picker, prefix):
    identity_revision = (revision, 'operator-progressive-v1')
    if state.get(prefix + 'identity-revision') != identity_revision:
        for key in list(state):
            if key.startswith((prefix + 'start:', prefix + 'end:')):
                del state[key]
        state[prefix + 'identity-revision'] = identity_revision
    if not state.get(prefix + 'progressive-controls'):
        state[prefix + 'depth'] = 1
        state[prefix + 'progressive-controls'] = True
    scope = st.radio('Source scope', ['All sources', 'Selected sources'], horizontal=True, key=prefix + 'scope')
    refs = (st.multiselect('Source filter', sources, key=prefix + 'sources',
                          help='All selected sources are applied during traversal; there is no 20-source limit.')
            if scope == 'Selected sources' else None)
    st.caption('All published graph sources are included.' if refs is None else f'{len(refs)} sources selected.')
    if refs == []:
        st.info('Select at least one source, or choose All sources.')
    mode = st.radio('Explore', ['Neighborhood', 'Connection'], horizontal=True, key=prefix + 'mode')
    start = picker(st, project, state, 'Starting entity', prefix + 'start', refs)
    end = picker(st, project, state, 'Destination entity', prefix + 'end', refs) if mode == 'Connection' else None
    direction = st.selectbox('Traversal direction', ['connectivity', 'outgoing', 'incoming'], key=prefix + 'direction',
        help='Connectivity follows either direction. Incoming/outgoing follow stored arrows; they do not interpret relationship labels.')
    query = state.get(prefix + 'inspection')
    base = (start, end, mode, direction, None if refs is None else tuple(refs))
    same = isinstance(query, OperatorExploration) and state.get(prefix + 'base') == base
    disabled = start is None or refs == [] or mode == 'Connection' and end is None
    direct = st.button('Direct connections', disabled=disabled or mode != 'Neighborhood', key=prefix + 'direct')
    if direct:
        state[prefix + 'depth'] = 1
    depth = st.number_input('Traversal depth', min_value=1, step=1, key=prefix + 'depth',
        help='Start with direct connections (1). Change this value, then run the query. Increasing depth reuses existing progress when the identity, sources, direction, and graph are unchanged.')
    request = (*base, depth)
    run = st.button('Find connection' if mode == 'Connection' else 'Explore neighborhood', disabled=disabled, key=prefix + 'run')
    if run or direct:
        if (run and same and query.cursor is not None and depth > query.cursor.depth
                and state.get(prefix + 'revision') == revision and query.check_current()):
            query.expand(depth)
            state[prefix + 'request'] = request
            state[prefix + 'export-context']['request']['depth'] = depth
            st.rerun()
        if isinstance(query, OperatorExploration):
            query.close()
        query = OperatorExploration(project, start, end, direction=direction, depth=depth, refs=refs)
        state[prefix + 'inspection'] = query
        state[prefix + 'base'], state[prefix + 'request'], state[prefix + 'revision'] = base, request, revision
        state[prefix + 'export-context'] = query_export_context(project, query.result, request={
            'entity': start, 'target': end, 'direction': direction, 'depth': depth, 'refs': refs})
        st.rerun()
    if not isinstance(query, OperatorExploration):
        return
    if state.get(prefix + 'revision') != revision or state.get(prefix + 'request') != request:
        if query.state == 'running':
            query.pause()
            st.rerun()
        st.info('The graph or query settings changed. Run the query again to see matching results.')
        return
    if not query.check_current():
        st.warning(query.result.get('problem', 'The graph changed. Run a new exploration.'))
        return
    if query.state == 'running' and st.button('Stop exploration', key=prefix + 'stop'):
        query.pause()
        st.rerun()
    if query.state == 'paused' and st.button('Continue exploration', key=prefix + 'continue'):
        query.resume()
        st.rerun()
    if query.state == 'running':
        st.caption('Exploring in work batches. Stop exploration pauses after the current batch; Continue resumes without losing discoveries.')
        query.advance()
        if query.state != 'running':
            st.rerun()  # Remove the timer when work completes, fails, or is invalidated.
    result = query.result
    state[prefix + 'result'] = result
    if query.state in ('invalidated', 'failed'):
        st.warning(result.get('problem', 'Exploration could not continue.'))
        return
    if outcome := query_outcome(result):
        st.info(outcome)
    captions, warnings = query_messages(result)
    for message in captions:
        st.caption(message)
    for message in warnings:
        st.warning(message)
    if query.cursor is None:
        return
    if query.cursor.goal and query.state != 'complete':
        st.info('The connection search is unfinished. No partial chain is presented as a complete path.')
    try:
        graph_results.render(st, project, state, query, state[prefix + 'export-context'], prefix)
        if not query.check_current():
            st.rerun()
    except VerificationChanged:
        st.rerun()  # Clear stale rows/downloads and display the query's invalidation reason.
