'''Select saved answers and authored canon pages for provider-free findings downloads.'''

from osintgpt.findings import canon_pages, findings_package, findings_markdown, findings_csv


def render(st, project, conversation, state):
    prefix = f'findings-export:{project.id}:{conversation.id if conversation else "none"}:'
    with st.expander('Export answers and findings'):
        st.caption('Choose saved answers or existing canon Markdown pages. Original text, citations, and saved tool '
                   'evidence are retained; no new synthesis or current-source verification runs.')
        available = list(range(len(conversation.turns))) if conversation else []
        turns = st.multiselect('Answers to export', available, key=prefix + 'answers',
                               format_func=lambda index: f'{index + 1}. {conversation.turns[index].question}')
        pages = st.multiselect('Canon findings to export', canon_pages(project), key=prefix + 'pages')
        if not available:
            st.caption('Open a saved conversation to select answers.')
        signature = (tuple(turns), tuple(pages))
        if st.button('Prepare findings downloads', key=prefix + 'prepare', disabled=not (turns or pages)):
            try:
                package = findings_package(project, conversation=conversation, turns=turns, pages=pages)
                state[prefix + 'ready'] = {'signature': signature, 'package': package,
                    'md': findings_markdown(package), 'csv': findings_csv(package)}
            except (OSError, ValueError) as error:
                st.error(str(error))
        prepared = state.get(prefix + 'ready')
        if not prepared:
            return
        if prepared['signature'] != signature:
            st.info('The selection changed. Prepare the downloads again.')
            return
        st.caption(f"Prepared {len(prepared['package']['items'])} items. This fixed snapshot retains answer-time evidence; "
                   'prepare again after editing a canon page. Older answers may contain citations without saved evidence.')
        for suffix, mime, label in [('md', 'text/markdown', 'Markdown'), ('csv', 'text/csv', 'CSV')]:
            st.download_button(f'Download findings {label}', prepared[suffix].encode('utf8'),
                               file_name=f'{project.slug}-findings.{suffix}', mime=mime,
                               key=prefix + suffix, on_click='ignore')
