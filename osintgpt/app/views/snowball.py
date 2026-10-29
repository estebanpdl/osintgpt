'''Operator controls and results for semantic similarity chains.'''

from osintgpt.agentic.snowball import DEFAULT_DEPTH, DEFAULT_THRESHOLD, DEFAULT_DEDUPE, snowball
from osintgpt.agentic.snowball_output import (
    METHOD_NOTE, PERIODS, grouped_rows, snowball_csv, snowball_data,
    snowball_rows, snowball_xlsx, usage_text,
)
from osintgpt.app.session import list_projects, select_project


def render(st, runtime, home, state):
    st.title('Snowball')
    st.caption('Start with a query, then follow the closest unseen passage at each hop.')
    st.info(METHOD_NOTE)
    project = runtime.project
    prefix = f'snowball:{project.id}:'
    entries = list_projects(home)
    slugs = [entry.slug for entry in entries]
    if project.slug in slugs:
        selected = st.selectbox('Project', slugs, index=slugs.index(project.slug), key=prefix + 'project')
        if selected != project.slug:
            select_project(state, selected)
            st.rerun()
    st.caption(f'Embedding model: {runtime.embedding_model or "project default"}')
    with st.form(prefix + 'form'):
        query = st.text_area('Initial query', key=prefix + 'query')
        columns = st.columns(2)
        depth = columns[0].number_input(
            'Depth', min_value=1, value=DEFAULT_DEPTH, step=1, key=prefix + 'depth',
            help='Maximum accepted hits, including the first. No additional hop cap.',
        )
        threshold = columns[1].number_input(
            'Similarity threshold', min_value=-1.0, max_value=1.0,
            value=DEFAULT_THRESHOLD, step=0.05, key=prefix + 'threshold',
            help='Accept scores at or above this value. Scores depend on the embedding model.',
        )
        dedupe = st.selectbox(
            'Deduplicate by', ['text', 'chunk'], index=['text', 'chunk'].index(DEFAULT_DEDUPE),
            key=prefix + 'dedupe', help='Text skips copies after Unicode, case and whitespace normalization.',
        )
        date_field = st.text_input(
            'Date field (optional)', key=prefix + 'date-field',
            help='Leave blank for the indexed timestamp, or enter a stored metadata field name.',
        )
        date_format = st.text_input(
            'Date format (optional)', key=prefix + 'date-format',
            help='ISO dates by default. For example, %d/%m/%Y explicitly means day/month/year.',
        )
        drift = st.checkbox('Show similarity to query', value=True, key=prefix + 'drift')
        refs_text = st.text_area(
            'Source refs (optional)', key=prefix + 'refs',
            help='One indexed file ref per line. Leave blank to search the project.',
        )
        st.caption('Each hop re-embeds its seed; similarity to query adds document embeddings. Provider charges may apply.')
        submitted = st.form_submit_button('Run Snowball', type='primary')
    if submitted:
        state.pop(prefix + 'result', None)
        state.pop(prefix + 'downloads', None)
        try:
            with st.spinner('Following the similarity path…'):
                state[prefix + 'result'] = snowball(
                    project, query, home=home, config=runtime.config,
                    depth=int(depth), threshold=float(threshold), dedupe=dedupe,
                    score_against_initial=drift, date_field=date_field,
                    date_format=date_format.strip() or None,
                    refs=[ref.strip() for ref in refs_text.splitlines() if ref.strip()] or None,
                )
        except Exception as error:
            st.error(str(error))
    walk = state.get(prefix + 'result')
    if walk is None:
        return
    st.subheader('Last completed chain')
    st.text(f'Initial query: {walk.query}')
    st.caption(f'{len(walk)} hops; {walk.stopped}. Model: {walk.model}; depth {walk.depth}; threshold {walk.threshold}; dedupe {walk.dedupe}.')
    st.caption(usage_text(walk))
    group_by = st.selectbox('Group by period', PERIODS, key=prefix + 'group-by')
    if group_by != 'none':
        st.caption('Grouping changes the display only. Hop numbers and export rows retain traversal order; dates use their stated timezone.')
    rows = snowball_rows(walk, group_by)
    if not rows:
        st.info('No hit met the search conditions. Try another query or threshold, or check the project index and embedding model.')
    for period, grouped in grouped_rows(rows, group_by):
        if group_by != 'none':
            st.subheader(period)
        for row in grouped:
            similarity = row['similarity_to_query']
            label = f'Hop {row["hop"]} · score {row["score"]:.4f}'
            with st.expander(label, expanded=True):
                st.caption('Similarity to query: ' + (f'{similarity:.4f}' if similarity is not None else 'off'))
                st.text(f'Source: {row["citation"]} (chunk {row["sequence"]})')
                st.text(f'Date: {row["date"] or "unknown"} ({row["date_status"]})')
                st.caption('Seed')
                st.text(row['query'])
                st.caption('Hit')
                st.text(row['text'])
                st.text('Identical-text sources encountered: ' + ', '.join(row['identical_text_refs']))
    st.caption(snowball_data(walk)['identical_text_refs_scope'])
    _downloads(st, state, prefix, walk, group_by)


def _downloads(st, state, prefix, walk, group_by):
    key = prefix + 'downloads'
    ready = state.get(key)
    if ready is not None and ready['group_by'] != group_by:
        state.pop(key, None)
        ready = None
    if st.button('Prepare downloads', key=prefix + 'prepare'):
        ready = {'group_by': group_by, 'csv': snowball_csv(walk, group_by)}
        try:
            ready['xlsx'] = snowball_xlsx(walk, group_by)
        except ValueError as error:
            ready['xlsx_error'] = str(error)
        state[key] = ready
    if ready is None:
        return
    st.download_button('Download CSV', ready['csv'], file_name='snowball.csv',
                       mime='text/csv', key=prefix + 'csv')
    if 'xlsx' in ready:
        st.download_button('Download XLSX', ready['xlsx'], file_name='snowball.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           key=prefix + 'xlsx')
    else:
        st.warning(ready['xlsx_error'])
