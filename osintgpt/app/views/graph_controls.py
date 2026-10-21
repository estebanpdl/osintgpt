'''Graph build controls and progress rendering over the shared background runner.'''

from dataclasses import asdict
import sqlite3

from osintgpt.graph import preflight_graph, saved_graph_status
from osintgpt.graph.jobs import JOBS
from osintgpt.graph.provider import LazyGraphGenerator
from osintgpt.graph.model_metadata import selection_label, selection_metadata
from osintgpt.graph.run_settings import GraphRunSettings
from osintgpt.graph.output_policy import policy_for
from osintgpt.graph.status_messages import build_messages, count_label
from osintgpt.llm import build_generation_provider
from osintgpt.llm.usage import UsageRecorder
from osintgpt.projects import load_user_defaults


def preferences(st, project):
    try:
        saved = GraphRunSettings.load(project)
    except (OSError, ValueError) as error:
        st.error(f'Cannot read saved graph preferences: {error}')
        return None
    prefix = f'graph-settings:{project.id}:'
    values = asdict(saved)
    with st.expander('Build settings'):
        st.caption('Saved for this project’s app builds. CLI builds continue to use their command options. '
                   'Running builds keep their starting settings. Changing source windows may discard '
                   'incompatible saved windows; pacing changes do not.')
        st.caption('Output allowance is managed by the selected provider. Older saved output-token caps are ignored.')
        fields = [('context_budget', 'Source window allowance (UTF-8 bytes)', 64, 'Maximum serialized source text per window; the existing character cap also applies. Identity context and continuation are automatic. This does not cap the whole conversation.'),
                  ('rpm', 'Generation requests per minute', 0, 'Zero adds no operator ceiling. Provider limits still apply.'),
                  ('tpm', 'Generation budget per minute', 0, 'Pacing estimate: input bytes plus known model output allowance, otherwise twice input bytes. Does not cap output; zero adds no pacing ceiling.'),
                  ('retry_attempts', 'Maximum attempts per request', 1, 'Includes the initial HTTP attempt.')]
        for name, label, minimum, help_text in fields:
            values[name] = st.number_input(label, min_value=minimum, value=values[name], step=1,
                                           key=prefix + name, help=help_text)
        values['retry_seconds'] = st.number_input('Retry time budget (seconds)', min_value=0.001,
            value=float(saved.retry_seconds), key=prefix + 'retry_seconds')
        values['repair_quotes'] = st.checkbox('Allow entity and relationship quotation corrections', saved.repair_quotes,
            key=prefix + 'repair', help='May add a model request when a supporting quotation cannot be located.')
        values['cost_ceiling_usd'] = st.number_input('Run cost ceiling (USD)', min_value=0.001,
            value=saved.cost_ceiling_usd, step=0.01, key=prefix + 'cost',
            help='Optional. The lower of this and the project ceiling applies. Stops after a response crosses '
                 'the ceiling, or when its usage or price cannot be counted. Saved responses can be resumed.')
        try:
            chosen = GraphRunSettings(**values)
        except ValueError as error:
            st.error(str(error))
            return None
        if st.button('Save build settings', key=prefix + 'save'):
            chosen.save(project)
            st.success('Graph build settings saved.')
    return chosen


def generator_factory(runtime, home, options):
    effective = runtime.project.effective_settings(load_user_defaults(home))
    config = runtime.config
    ceilings = [value for value in (effective.cost_ceiling_usd, options.cost_ceiling_usd) if value is not None]
    def create():
        recorder = UsageRecorder(cost_ceiling_usd=min(ceilings) if ceilings else None)
        return LazyGraphGenerator(build_generation_provider, effective, config, recorder)
    return create


def render(st, runtime, home, state, status):
    project = runtime.project
    options = preferences(st, project)
    if options is None:
        return
    factory = generator_factory(runtime, home, options)
    try:
        generator = factory()
        identity = generator.graph_options()
        selection = selection_metadata(generator, identity)
    except ValueError as error:
        st.error(f'Graph provider settings need attention: {error}')
        return
    st.caption('Next build model: ' + selection_label(selection) + '. Change this in Settings → Graph extraction.')
    st.caption('Next build output: ' + policy_for(generator).label())
    if selection['mode'] == 'inherit':
        st.caption('Using Chat/RAG settings; changing them affects the next graph build.')
    job = JOBS.get(project)
    active = bool(job and job['active']) or status['status'] == 'building'
    action = st.selectbox('Build action', ['Build / update changed sources', 'Resume saved work', 'Rebuild all sources'],
                          key=f'graph-action:{project.id}', disabled=active)
    rebuild = action == 'Rebuild all sources'
    if rebuild:
        st.warning('Rebuild extracts registered sources again and discards saved extraction windows. Provider charges may apply.')
    prefix = f'graph-build:{project.id}:'
    preview_settings = (asdict(options), action, selection)
    if state.get(prefix + 'preview') and state.get(prefix + 'preview-settings') != preview_settings:
        state.pop(prefix + 'preview', None)
        state.pop(prefix + 'preview-settings', None)
        st.info('Build settings or model changed. Preview again for an updated estimate.')
    if st.button('Preview build workload', key=prefix + 'preflight', disabled=active):
        with st.spinner('Reading sources and saved work…'):
            try:
                state[prefix + 'preview'] = preflight_graph(project, factory(), options=options.extraction(),
                                                            request_limits=options.limits(), rebuild=rebuild)
                state[prefix + 'preview-settings'] = preview_settings
            except Exception as error:
                st.error(str(error))
    preview = state.get(prefix + 'preview')
    if preview:
        counts = preview['counts']
        st.info(f"Preview: {counts['documents_to_extract']} sources to extract; {counts['documents_unchanged']} unchanged; "
                f"{counts['windows_pending']} pending windows. Follow-up requests and final cost depend on the content.")
        st.caption(f"{counts['documents_model_changed']} completed sources have a changed graph model; "
                   f"{counts['checkpoints_incompatible']} incompatible saved source extractions "
                   f"({counts['windows_incompatible']} windows, {counts['responses_incompatible']} responses).")
        estimate = preview['budget_cost_estimate_usd']
        st.caption(preview['output_policy_label'])
        lookup = preview.get('identity_lookup', {})
        st.caption('Entity lookup: ' + lookup.get('mode', 'unknown').replace('_', ' ') + '. ' + lookup.get('note', ''))
        st.caption('Future cost is unknown: useful follow-up work has no fixed call ceiling.' if estimate is None else f'Budget-based cost estimate: ${estimate:.6f}.')
        with st.expander('Preview details and assumptions'):
            st.dataframe(preview['details'], hide_index=True)
            for note in preview['assumptions']:
                st.caption(note)
    label = ('Rebuild graph' if rebuild else 'Resume build' if action == 'Resume saved work'
             else 'Update changed sources' if status['available'] else 'Build graph')
    st.caption('Only starting a build can make model calls. Builds continue while you navigate or refresh this app. '
               'If the app process stops, saved work remains available for an explicit resume.')
    if st.button(label, type='primary', key=prefix + 'start', disabled=active or not project.settings.graph_enabled):
        run_action = 'rebuild' if rebuild else 'resume' if action == 'Resume saved work' else 'update' if status['available'] else 'initial'
        JOBS.start(project, factory, options, action=run_action)
        st.rerun()
    if not project.settings.graph_enabled:
        st.info('Enable Relationship graph under Settings to build. Saved graphs can still be inspected here.')
    if status['status'] == 'building' and not (job and job['active']):
        st.info('A build is running outside this app process. Its saved progress is shown below; cancel it where it was started.')


def progress(st, project):
    @st.fragment(run_every='2s')
    def show():
        job = JOBS.get(project)
        try:
            status = saved_graph_status(project, check_freshness=False)
        except (OSError, ValueError, sqlite3.Error) as error:
            st.warning(f'Progress is temporarily unavailable: {error}')
            return
        seen_key = f'graph-running:{project.id}'
        was_active = st.session_state.get(seen_key, False)
        active = bool(job and job['active']) or status['status'] == 'building'
        st.session_state[seen_key] = active
        if was_active and not active:
            st.rerun()
        st.caption(build_messages(status, job=job)['activity'])
        if active:
            running_model = (job or {}).get('graph_model') or status.get('run_model', {})
            st.caption('Running build model: ' + (selection_label(running_model) if running_model else 'unknown (not recorded yet)')
                       + '. Changes in Settings apply to a later build.')
        counts = status['counts']
        completed, total = (count_label(counts, key) for key in ('windows_completed', 'windows_total'))
        if 'unknown' in (completed, total):
            st.caption('Window progress has not been recorded.')
        else:
            st.progress(min(1.0, int(completed) / max(1, int(total))),
                        text=f'{completed}/{total} windows in records read so far')
        event = (job or {}).get('last_event') or status.get('last_event', {})
        if event:
            st.caption(f"Latest activity: {event['kind'].replace('_', ' ')} — {event.get('data', {}).get('ref', '')}")
        if job and job['active']:
            if st.button('Cancel build', key=f'graph-cancel:{project.id}', disabled=job['cancel_requested']):
                JOBS.cancel(project)
                st.rerun(scope='fragment')
            if job['cancel_requested']:
                st.info('Cancellation requested. The active request may finish before the next safe stopping point.')
        elif job:
            getattr(st, 'warning' if job.get('problem') else 'success')(job.get('problem') or job.get('summary', job['status']))
        usage = status.get('usage', {})
        st.caption(f"{count_label(counts, 'requests_attempted')} requests; {count_label(counts, 'retries')} retries; "
                   f"{count_label(counts, 'windows_reused')} saved windows reused. Recorded tokens: {usage.get('total_tokens', 'unknown')}.")
        st.caption(f"{count_label(counts, 'entity_lookups')} local entity lookups; "
                   f"{count_label(counts, 'entity_lookups_reused')} saved lookup results reused. "
                   'Model turns that request or use these results are included in generation requests.')
        st.caption(f"{count_label(counts, 'responses_reused')} exact responses reused; "
                   f"{count_label(counts, 'responses_recovered')} legacy responses recovered; "
                   f"{count_label(counts, 'context_restarts')} context restarts. "
                   f"Generation requests include {count_label(counts, 'completion_requests')} finalization and "
                   f"{count_label(counts, 'structured_correction_requests')} structured-correction requests.")
        if usage.get('estimated_cost_usd') is not None:
            st.caption(f"Recorded estimated cost: ${usage['estimated_cost_usd']:.6f}.")
        from osintgpt.graph.accounting import usage_messages
        for message in usage_messages(usage):
            st.caption(message)
    show()
