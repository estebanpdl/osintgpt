'''Explicit preparation of graph downloads from full or displayed query snapshots.'''

import json
import sqlite3

from osintgpt.graph.export import serialize_package
from osintgpt.graph.export_snapshot import full_graph_package
from osintgpt.graph.verification_cache import VerificationChanged


def render(st, project, state, *, result=None, context=None, filtered=None, filters=None, inspection=None,
           visible=None, displayed=None, display=None):
    selected = result is not None
    prefix = f'graph-export:{project.id}:{"selection" if selected else "full"}:'
    with st.expander('Export query results' if selected else 'Export full graph'):
        st.caption('Exports preserve citations, source versions, qualifications, verification, and build/query context. '
                   'Preparing an export makes no model calls.')
        if selected:
            scopes = {'All matching results': ('all_matching', filtered),
                      'Current table page': ('table_page', visible), 'Displayed diagram': ('diagram', displayed)}
            if state.get(prefix + 'scope') not in scopes:
                state[prefix + 'scope'] = 'All matching results'
            choice = st.radio('Export scope', list(scopes), key=prefix + 'scope')
            scope, rows = scopes[choice]
            ids = [row['assertion_id'] for row in rows or []]
            display_context = ({} if scope == 'all_matching' else
                               {key: value for key, value in (display or {}).items()
                                if (key == 'diagram_selection') == (scope == 'diagram')})
            st.caption(f'{choice}: {len(ids)} assertions. All matching results applies the current filters across '
                       'the discovered query scope, before pagination. Displayed exports contain only the chosen page or diagram. '
                       'Unfinished exploration is exported as incomplete; its final matching total is unknown.')
            package = None
            fingerprint = json.dumps({'context': context, 'scope': choice, 'filters': filters,
                # The token and current-state checks bind evidence. Opening related
                # evidence can add watched sources without changing this selection.
                'query': {key: value for key, value in result.items() if key not in ('path', 'claims', 'verification_state')},
                'token': getattr(inspection, 'token', None), 'display': display_context,
                'ids': ids}, ensure_ascii=False, sort_keys=True)
        else:
            package, fingerprint = None, None
            st.caption('Checks current source text during preparation. Includes all published assertions, even if the build is incomplete.')
        if st.button('Prepare graph downloads', key=prefix + 'prepare'):
            try:
                with st.spinner('Preparing export…'):
                    package = (inspection.prepare_export(scope=scope, filters=filters, assertion_ids=ids,
                                                         display=display_context, request=context.get('request'))
                               if selected else full_graph_package(project))
                    prepared = {'package': package, 'fingerprint': fingerprint,
                        'json': serialize_package(package, '.json'), 'cypherl': serialize_package(package, '.cypherl')}
                    if selected and not inspection.check_current():
                        raise VerificationChanged(inspection.result['problem'])
                    state[prefix + 'ready'] = prepared
            except VerificationChanged:
                raise  # The exploration panel clears stale results and shows the reason.
            except (OSError, ValueError, sqlite3.Error) as error:
                st.error(str(error))
                return
        prepared = state.get(prefix + 'ready')
        if not prepared:
            return
        if selected and not inspection.check_current():
            raise VerificationChanged(inspection.result['problem'])
        if selected and prepared['fingerprint'] != fingerprint:
            st.info('The selection changed. Prepare the downloads again.')
            return
        metadata = prepared['package']['metadata']
        st.caption(f"Prepared {len(prepared['package']['edges'])} assertions; scope: {metadata['scope']}. "
                   'This download is a fixed snapshot. Prepare it again to refresh.')
        if metadata['graph'].get('stale') or metadata['graph'].get('incomplete') or metadata.get('truncated'):
            st.warning('This export carries stale, incomplete, or limited-result status. Review its metadata before drawing conclusions.')
        for extension, mime in [('json', 'application/json'), ('cypherl', 'text/plain')]:
            st.download_button('Download graph ' + extension.upper(), prepared[extension].encode('utf8'),
                               file_name=f'{project.slug}-{metadata["scope"]}.{extension}', mime=mime,
                               key=prefix + extension, on_click='ignore')
