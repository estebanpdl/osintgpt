'''Explicit, provider-free identity review and reconciliation controls.'''

from osintgpt.graph.identity_review import reconcile_identities, review_identities
import sqlite3


def render(st, project, state):
    prefix = f'graph-identity:{project.id}:'
    with st.expander('Identity mentions and resolution decisions'):
        query = st.text_input('Identity name or ID', key=prefix + 'query')
        if st.button('Review identities', key=prefix + 'read'):
            try:
                state[prefix + 'data'] = review_identities(project, query)
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(str(error))
        data = state.get(prefix + 'data')
        if not data:
            return
        st.caption(data['note'])
        st.json(data)
        if data['truncated']:
            st.info('More candidates exist. Refine the query or use CLI --offset to page them.')
        source = st.text_input('Identity ID to combine', key=prefix + 'source')
        target = st.text_input('Identity ID to retain', key=prefix + 'target')
        reason = st.text_input('Evidence-based reason', key=prefix + 'reason')
        undo = st.text_input('Decision ID to undo (optional)', key=prefix + 'undo')
        split_source = st.text_input('Source reference to separate (optional)', key=prefix + 'split')
        request = (source, target, reason, undo, split_source)
        if st.button('Preview identity decision', key=prefix + 'preview'):
            try:
                state[prefix + 'preview-data'] = reconcile_identities(project, source, target, reason=reason, undo=undo,
                                                                     split_source=split_source)
                state[prefix + 'request'] = request
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(str(error))
        preview = state.get(prefix + 'preview-data')
        if preview and state.get(prefix + 'request') == request:
            st.json(preview)
            st.caption('Applying creates a SQLite backup. It changes identity mappings without model calls or deleting evidence.')
            if st.button('Apply identity decision', key=prefix + 'apply'):
                try:
                    result = reconcile_identities(project, source, target, reason=reason, undo=undo,
                                                  apply=True, expected_revision=preview['revision'], split_source=split_source)
                    st.success('Identity decision saved. Backup: ' + result['backup'])
                    state.pop(prefix + 'preview-data', None)
                    state.pop(f'graph-explore:{project.id}:result', None)
                    state.pop(f'graph-explore:{project.id}:identity-revision', None)
                    state[prefix + 'data'] = review_identities(project, query)
                except (ValueError, OSError, sqlite3.Error) as error:
                    st.error(str(error))


def revalidation(st, project, state):
    from osintgpt.graph.revalidation import revalidate_proposals
    key = f'graph-revalidation:{project.id}'
    with st.expander('Recheck saved rejected proposals'):
        st.caption('Local evidence checks only. No new quotations or model calls. Old audit records missing '
                   'source ranges or endpoint identities remain rejected with an explanation.')
        offset = st.number_input('Saved proposal page offset', min_value=0, value=0, step=100, key=key + ':offset')
        if st.button('Preview local revalidation', key=key + ':preview'):
            try:
                state[key] = revalidate_proposals(project, offset=offset)
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(str(error))
        preview = state.get(key)
        if preview and preview['offset'] == offset:
            st.json(preview)
            if preview['truncated']:
                st.caption(f"More saved proposals: continue with page offset {preview['next_offset']}.")
            if st.button('Apply supported corrections', key=key + ':apply',
                         disabled=not (preview['accepted_entities'] or preview['accepted_claims'])):
                try:
                    result = revalidate_proposals(project, apply=True, offset=offset, expected_revision=preview['revision'])
                    st.success('Supported corrections saved. Backup: ' + str(result['backup']))
                    if result['truncated']:
                        st.caption(f"Continue with page offset {result['next_offset']} after this application.")
                    state.pop(key, None)
                    state.pop(f'graph-explore:{project.id}:result', None)
                except (ValueError, OSError, sqlite3.Error) as error:
                    st.error(str(error))
