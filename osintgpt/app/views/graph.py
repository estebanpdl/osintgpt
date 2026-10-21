'''Graph build, exploration, and evidence workflow over provider-free inspection services.'''

import sqlite3

from osintgpt.graph import verify_evidence
from osintgpt.graph.inspection import inspect_entities, overview, rejected_proposals
from osintgpt.graph.status_messages import query_outcome

from . import graph_controls, graph_exports, graph_identities
from .graph_status import render as render_status


def entity_picker(st, project, state, label, key, refs=None):
    query = st.text_input(label, key=key + ':query', help='Search names or aliases in their original language, or paste an entity ID.')
    previous = state.get(key + ':searched')
    scope_changed = isinstance(previous, tuple) and previous[0] == query and previous[1] != refs
    if st.button('Find identities', key=key + ':find', disabled=not query.strip()) or scope_changed:
        state[key + ':found'] = inspect_entities(project, query, refs=refs)
        state[key + ':searched'] = (query, refs)
    if state.get(key + ':searched') != (query, refs):
        return None
    found = state.get(key + ':found', {})
    candidates = found.get('candidates', [])
    if found.get('truncated'):
        st.info('Candidate results were limited. Narrow the name or use a persistent ID.')
    if found and not candidates:
        st.info(query_outcome(found) or 'No matching identity.')
    if not candidates:
        return None
    indexed = {row['id']: row for row in candidates}
    choice = st.selectbox('Select identity', list(indexed), index=None, key=key + ':choice',
                          format_func=lambda value: f"{indexed[value]['name']} ({indexed[value]['type'] or 'unspecified type'}) · {value}")
    if choice not in indexed:
        return None
    row = indexed[choice]
    counts = row.get('relationships', {})
    st.caption(f"Direct assertions involving this identity: {counts.get('all_sources', 0)} across all sources; "
               f"{counts.get('selected_sources', 0)} in the active source scope.")
    st.caption(f"Distinct connected identities: {counts.get('distinct_neighbors_all_sources', 0)} across all sources; "
               f"{counts.get('distinct_neighbors_selected_sources', 0)} in the active source scope.")
    st.caption('Direct counts include both arrow directions, repeated sourced assertions, and all qualifications. '
               'They do not apply the traversal direction or result filters and do not establish truth.')
    if not counts.get('all_sources'):
        st.info('Isolated identity: no accepted relationships are published for this identity.')
    st.caption(row['description'] or 'No distinguishing description recorded.')
    if row['aliases']:
        st.caption('Aliases: ' + ', '.join(row['aliases']))
    st.caption('Sources: ' + ', '.join(source['ref'] for source in row['sources']))
    if row['identifiers']:
        st.json(row['identifiers'])
    return choice


def explore(st, project, state, sources, revision):
    from .graph_exploration import render as render_exploration
    render_exploration(st, project, state, sources, revision, entity_picker)


def render(st, runtime, home, state):
    project = runtime.project
    st.subheader(f'Graph — {project.name}')
    st.caption('Explore sourced assertions. A located quotation supports what a document says; it does not establish that it is true.')
    if st.button('Refresh graph status', key=f'graph-refresh:{project.id}'):
        query = state.pop(f'graph-explore:{project.id}:inspection', None)
        if hasattr(query, 'close'):
            query.close()
        state.pop(f'graph-explore:{project.id}:result', None)
        state.pop(f'graph-build:{project.id}:preview', None)
        st.rerun()
    try:
        data = overview(project)
    except (OSError, ValueError, sqlite3.Error) as error:
        st.error(f'Cannot open graph: {error}')
        st.caption(f'Check supported upgrades with "osintgpt graph migrate --project {project.slug} --dry-run". '
                   'Run without --dry-run to create a backup and upgrade; migration makes no model calls.')
        return
    status, coverage = data['status'], data['coverage']
    columns = st.columns(4)
    for column, label, value in zip(columns, ['Entities', 'Assertions', 'Current sources', 'Sources with extraction gaps'],
                                     [status['entities'], status['edges'], f"{coverage['current']}/{coverage['registered']}", coverage['with_gaps']]):
        column.metric(label, value)
    st.caption('Current source coverage includes completed sources with zero relationships. Extraction gaps are shown separately.')
    render_status(st, project, status=status)
    with st.expander('Corpus coverage and last-run details'):
        st.dataframe(data['sources'], hide_index=True, width='stretch')
        st.json({name: status.get(name) for name in ('started_at', 'finished_at', 'counts', 'usage', 'problem', 'previous_runs')})
    with st.expander('Build and recovery', expanded=not status['available'] or status['status'] in ('building', 'interrupted', 'failed')):
        graph_controls.render(st, runtime, home, state, status)
        graph_controls.progress(st, project)
    with st.expander('Generation diagnostics'):
        from osintgpt.graph.diagnostics import generation_diagnostics
        diagnostic_offset = st.number_input('Response page offset', min_value=0, value=0, step=50, key=f'graph-diagnostic-offset:{project.id}')
        raw_responses = st.checkbox('Include original local response content', value=False, key=f'graph-diagnostic-raw:{project.id}')
        if st.button('Inspect generation responses', key=f'graph-diagnostic-read:{project.id}'):
            st.json(generation_diagnostics(project, offset=diagnostic_offset, raw=raw_responses))
            st.caption('Local inspection makes no model calls. Raw responses can contain source material and opaque provider continuation.')
    if status['available']:
        graph_identities.render(st, project, state)
        graph_identities.revalidation(st, project, state)
        graph_exports.render(st, project, state)
        with st.expander('Evidence checks and rejected proposals'):
            if st.button('Verify all graph evidence', key=f'graph-verify:{project.id}'):
                checked = verify_evidence(project)
                st.info(checked.summary)
                st.dataframe([{'Assertion': row.edge.assertion_id, 'Source': row.edge.ref,
                               'Saved': row.snapshot_status, 'Current': row.status, 'State': row.source_state,
                               'Problem': row.problem or row.snapshot_problem} for row in checked.results[:100]], hide_index=True)
                st.caption('The summary covers all assertions; the table shows the first 100.')
            if st.button('Inspect rejected proposals', key=f'graph-rejected:{project.id}'):
                st.json(rejected_proposals(project))
                st.caption('At most 100 audit entries. History includes successful quotation corrections and may outlive a rebuild.')
        explore(st, project, state, [row['source'] for row in data['sources']],
                (status.get('finished_at'), status['edges'], status['entities'], status['status'], status.get('identity_revision', 0)))
