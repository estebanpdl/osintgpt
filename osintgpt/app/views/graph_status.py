'''Saved relationship-graph status alongside its retrieval setting.'''

import sqlite3

from osintgpt.graph import saved_graph_status
from osintgpt.graph.jobs import JOBS
from osintgpt.graph.status_messages import build_messages, count_label
from .graph_models import saved_models
from osintgpt.graph.reconciliation import refresh_label


def render(st, project, status=None, *, compact=False):
    try:
        status = status if status is not None else saved_graph_status(project)
    except (OSError, ValueError, sqlite3.Error) as error:
        st.warning(f'Could not read graph status: {error}')
        return
    label = status['status'].replace('_', ' ')
    st.caption(
        f'Relationship graph: {label}. '
        f'{status["entities"]} entities, {status["edges"]} relationships.'
    )
    messages = build_messages(status, job=JOBS.get(project))
    st.caption(messages['activity'])
    for message in messages['coverage']:
        st.warning(message)
    if not project.settings.graph_enabled:
        st.caption('Graph retrieval is disabled; any published graph remains available for inspection.')
    if compact:
        legacy = status.get('legacy', {})
        if any(legacy.values()):
            st.warning(f"Migrated legacy data: {legacy.get('entities', 0)} entities and {legacy.get('edges', 0)} relationships "
                       'need re-extraction. See Graph for details.')
        st.caption('Build progress, source refresh details, and published extraction models are available in Graph.')
        return
    counts = status['counts']
    saved_models(st, status)
    if counts:
        st.caption(
            f'{"Build so far" if status["status"] == "building" else "Last build"}: '
            f'{count_label(counts, "documents_completed")} documents completed; '
            f'{count_label(counts, "documents_failed")} failed; '
            f'{count_label(counts, "records_loaded")} records read; '
            f'{count_label(counts, "windows_completed")}/{count_label(counts, "windows_total")} windows; '
            f'{count_label(counts, "requests_attempted")} generation calls '
            f'({count_label(counts, "requests_failed")} failed, '
            f'{count_label(counts, "invalid_responses")} invalid responses).'
        )
        st.caption(f"{count_label(counts, 'windows_reused')} saved windows reused; "
                   f"{count_label(counts, 'responses_reused')} saved responses reused; "
                   f"{count_label(counts, 'retries')} retries scheduled; "
                   'generation-call and usage counts cover only this run.')
        st.caption(f"Identity lookup: {count_label(counts, 'entity_lookups')} local tool lookups; "
                   f"{count_label(counts, 'entity_lookups_reused')} saved lookups reused; "
                   f"{count_label(counts, 'identity_unresolved')} unresolved mentions. "
                   'Model follow-up turns are included in generation calls; local lookups are not.')
        if counts.get('interactive_lookup_unavailable'):
            st.caption('This build used non-interactive identity context for some records because controlled tool support was unavailable.')
        if counts.get('field_warnings') or counts.get('repaired_entities'):
            st.caption(f"{count_label(counts, 'field_warnings')} entity field/identity warnings; "
                       f"{count_label(counts, 'repaired_entities')} entity quotations corrected. "
                       'Warnings retain supported entities and are separate from final rejections.')
        usage = status['usage']
        if counts.get('rejected_claims') or counts.get('rejected_entities') or counts.get('repaired_claims'):
            st.caption(f"Evidence checks: {count_label(counts, 'rejected_claims')} relationships and "
                       f"{count_label(counts, 'rejected_entities')} entities rejected; "
                       f"{count_label(counts, 'repaired_claims')} quotations corrected. "
                       'Inspect details with "osintgpt graph rejected --json".')
        if usage.get('output_policy'):
            from osintgpt.graph.output_policy import OutputPolicy
            st.caption('Last build output: ' + OutputPolicy(**usage['output_policy']).label())
        if counts.get('request_controls_unenforced'):
            st.caption('The custom provider does not expose controlled HTTP attempts; '
                       'retry pacing and saving before cost accounting cannot be guaranteed.')
        if usage.get('total_tokens') is not None:
            st.caption(f'Recorded usage: {usage["total_tokens"]:,} tokens.')
        if usage.get('estimated_cost_usd') is not None:
            st.caption(f'Recorded estimated cost: ${usage["estimated_cost_usd"]:.6f}.')
        if not usage.get('complete', True):
            st.caption('Recorded token usage or cost is incomplete.')
        st.caption(f"Sources: {count_label(counts, 'documents_skipped')} unchanged; "
                   f"{count_label(counts, 'documents_changed')} changed; "
                   f"{count_label(counts, 'documents_removed')} unregistered and removed from the graph.")
    checkpoints = status.get('checkpoints', {})
    if checkpoints.get('documents'):
        st.caption(f"Saved work: {checkpoints['documents']} unfinished sources; "
                   f"{checkpoints['windows_saved']} windows saved, "
                   f"{checkpoints.get('responses_saved', 0)} responses awaiting validation, "
                   f"{checkpoints['windows_pending']} remaining in started records. "
                   'Rerun "osintgpt graph build" to resume compatible work. '
                   'Changed sources or extraction settings require fresh extraction.')
        st.caption('Use "osintgpt graph build --dry-run" to estimate all remaining records before resuming.')
    event = status.get('last_event', {})
    if event and status['status'] == 'building':
        st.caption(f"Latest activity: {event['kind'].replace('_', ' ')}; "
                   f"{event.get('data', {}).get('ref', '')}")
    details = status.get('details', [])
    if counts.get('checkpoints_discarded'):
        changed = sum(bool(item.get('checkpoint_discarded')) and item.get('checkpoint_cause') == 'source_or_settings'
                      for item in details)
        if changed:
            st.caption(f'Previous extraction checkpoints for {changed} sources could not be reused because their '
                       'sources or settings changed. See each source’s processing outcome below.')
        others = counts['checkpoints_discarded'] - changed
        if others:
            st.caption(f'Previous extraction checkpoints for {others} sources were replaced for other reasons, '
                       'including explicit rebuilds. See source outcomes and diagnostics below.')
    if details:
        with st.expander('Source refresh details'):
            for item in details:
                st.text(f"{item['ref']}: {item['action']} — {refresh_label(item['reason'])}")
                if item.get('previous_published') and item['action'] != 'published':
                    st.caption('Previous published claims remain available until this source finishes successfully.')
                if item.get('problem'):
                    st.caption(item['problem'])
            st.caption('Unusable checkpoints do not mean source documents were skipped or deleted. '
                       'Only successfully completed replacements become published claims.')
            with st.expander('Source refresh technical diagnostics'):
                st.json(details)
    if status.get('previous_runs'):
        st.caption('Usage above is for the latest run. "graph status --json" includes up to ten previous runs.')
    if status['problem']:
        st.warning(status['problem'])
    legacy = status.get('legacy', {})
    if any(legacy.values()):
        st.warning(f"Migrated legacy data: {legacy.get('entities', 0)} entities and {legacy.get('edges', 0)} relationships "
                   'need re-extraction for the current identity and evidence guarantees. '
                   'Rebuild registered sources; migration preserves unknown qualifications and cannot recover old source snapshots.')
    if status['status'] in ('never_built', 'stale', 'interrupted', 'failed', 'complete_with_gaps'):
        st.caption(
            'Build explicitly in the Graph view or with "osintgpt graph build" in the CLI. '
            'Enabling retrieval does not build or update the graph.'
        )
