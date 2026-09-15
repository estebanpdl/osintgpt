'''Live indexing status and recovery controls for the Material view.'''

from osintgpt import index_project
from osintgpt.index_status import saved_index_status, status_rows, model_changes
from osintgpt.ingestion.transcription import transcriber_for_project
from osintgpt.llm.usage import Usage, UsageRecorder, CostLimitReached


def model_notice(st, status, provider, model):
    changes = model_changes(status, provider, model)
    if changes:
        previous = ', '.join(f'{p or "unrecorded provider"} / {m or "unrecorded model"}' for p, m in changes)
        st.warning(
            f'Selected embeddings: {provider} / {model or "model required"}. '
            f'Existing index/checkpoints include {previous}. Indexing with the '
            'selected configuration rebuilds affected files; incompatible '
            'checkpoints cannot be reused. To resume the previous run, restore '
            'its provider and model in Settings, with the same mapping and destination.'
        )
    return bool(changes)


def show_saved(st, status):
    for problem in status['problems']:
        st.warning(problem)
    if not (status['published'] or status['pending']):
        if not status['problems']:
            st.caption('No completed index or saved checkpoints yet.')
        return
    st.markdown('#### Saved indexing status')
    columns = st.columns(3)
    columns[0].metric('Last completed chunks', 'Unknown' if status['problems'] else f'{sum(r["chunks"] for r in status["published"]):,}')
    columns[1].metric('Saved checkpoint chunks', 'Unknown' if status['problems'] else f'{sum(r["completed"] for r in status["pending"]):,}')
    columns[2].metric('Pending embedding chunks', 'Unknown' if status['problems'] else f'{sum(r["total"] - r["completed"] for r in status["pending"]):,}')
    st.dataframe(status_rows(status), hide_index=True, use_container_width=True)
    st.caption(
        'Completed counts describe the last published versions. Checkpoints '
        'may replace those versions and become searchable only after the whole '
        'file is published. Missing record counts mean an older run did not '
        'record them. Index now verifies source, mapping, model and destination '
        'before reusing saved work.'
    )
    if status['pending']:
        st.info('Saved work is available. Use Index now with the same configuration to resume. A ready-to-publish file needs no more embeddings when its source and configuration still match.')


def usage_text(recorder, attempts):
    prefix = f'{attempts} embedding request attempt(s); {recorder.calls} successful provider response(s)'
    if not recorder.billable_calls:
        return prefix + '; no billable responses recorded.'
    prefix += f'; {recorder.total_tokens:,} reported tokens'
    if recorder.cost_complete:
        return prefix + f'; estimated cost ${recorder.estimated_billable_cost:.6f}.'
    return prefix + (
        f'; cost estimate incomplete (${recorder.estimated_billable_cost:.6f} known, '
        f'{recorder.unpriced_calls} unpriced, {recorder.uncounted_billable_calls} without token usage).'
    )


class LiveProgress:
    def __init__(self, st, state, key, ceiling=None):
        self.st, self.state, self.key = st, state, key
        self.recorder = UsageRecorder(cost_ceiling_usd=ceiling)
        self.data = {'phase': 'Planning', 'ref': '', 'completed': 0, 'total': 0, 'attempts': 0, 'history': []}
        state[key] = self.data
        self.phase, self.bar, self.details, self.usage = st.empty(), st.progress(0.0), st.empty(), st.empty()

    def __call__(self, event):
        kind, facts = event.kind, event.data
        data = self.data
        if kind == 'usage':
            try:
                self.recorder.record(Usage(**facts))
            finally:
                data['usage'] = usage_text(self.recorder, data['attempts'])
                self.usage.caption(data['usage'])
            return
        if kind == 'planning':
            data['phase'] = f'Planning with {facts["provider"]} / {facts["model"]}'
        elif kind == 'plan':
            data['plan'] = f'{facts["work"]} files to process; {facts["unchanged"]} unchanged; {facts["blocked"]} need attention.'
        elif kind == 'reading':
            data.update(ref=facts['ref'], phase=f'Reading file {facts["position"]}/{facts["total"]}', total=0, completed=0, statistics={})
        elif kind in ('embedding', 'checkpoint'):
            data.update(facts)
            data['phase'] = 'Embedding' if kind == 'embedding' else 'Checkpoint saved'
            if kind == 'embedding':
                data['reused'] = facts['completed']
        elif kind == 'publishing':
            data['phase'] = 'Publishing saved embeddings'
        elif kind == 'published':
            data.update(ref=facts['ref'], completed=facts['chunks'], total=facts['chunks'], phase='File published', completed_records=facts['statistics'].get('chunked_documents', 0))
        elif kind == 'request':
            data['attempts'] += 1
            data['phase'] = f'Embedding request: attempt {facts["attempt"]}/{facts["maximum"]}, {facts["inputs"]} inputs'
        elif kind == 'waiting':
            data['phase'] = f'Waiting for quota capacity/cooldown: about {facts["seconds"]:.1f}s until the next request'
        elif kind == 'retry':
            data['phase'] = f'Retry {facts["attempt"]}/{facts["maximum"]}: wait at least {facts["seconds"]:.2f}s after {facts["status"] or "transport failure"}'
        elif kind == 'request_stopped':
            data['phase'] = facts['reason']
        elif kind in ('failed', 'skipped', 'stopped'):
            data.update(ref=facts['ref'], phase=facts['problem'])
        if kind in ('retry', 'request_stopped', 'published', 'failed', 'skipped', 'stopped'):
            data['history'] = (data['history'] + [f'{data["ref"]}: {data["phase"]}'])[-20:]
        self.phase.text(f'{data["phase"]} — {data["ref"]}')
        self.bar.progress(min(data['completed'] / max(data['total'], 1), 1.0), text='Current file: saved embedding progress')
        if data['total'] or data.get('statistics'):
            stats = data.get('statistics', {})
            text = f'{data["completed"]:,}/{data["total"]:,} chunks saved; {data["total"] - data["completed"]:,} pending; {data.get("reused", 0):,} reused from checkpoints.'
            if 'chunked_documents' in stats:
                text += f' {data.get("completed_records", 0):,}/{stats["chunked_documents"]:,} records/documents fully embedded.'
            if 'records' in stats:
                text += f' Kept {stats["kept"]:,}/{stats["records"]:,} records; excluded {stats["without_content"]:,} with no content and {stats["too_short"]:,} below the minimum.'
            self.details.caption(text)
        else:
            self.details.caption(data.get('plan', 'Counting chunks after extraction; totals are not yet known.'))
        self.usage.caption(usage_text(self.recorder, data['attempts']))


def render(st, runtime, state):
    st.markdown('### Indexing')
    status = saved_index_status(runtime.project)
    st.caption(f'Will index with {runtime.embedding_provider or "configured provider"} / {runtime.embedding_model or "model required"}.')
    model_notice(st, status, runtime.embedding_provider, runtime.embedding_model)
    force = st.checkbox('Rebuild all files and discard saved checkpoints', key=f'index-rebuild:{runtime.project.id}', help='Re-embeds unchanged files too. Leave off to update or resume.')
    if force:
        st.warning('This run will recompute all registered files and discard their saved embedding checkpoints. Provider charges may apply.')
    ceiling = st.number_input('Stop at estimated run cost (USD)', min_value=0.0, value=None, step=0.01, key='index-cost-ceiling', help='Optional. Stops after a response crosses the ceiling, or if its usage/price is unavailable. The successful response is saved first.')
    st.caption('Use the app’s Stop control to interrupt a run, then Index now to resume. Changing pacing limits keeps compatible checkpoints.')
    key = f'index-run:{runtime.project.id}'
    if st.button('Index now', type='primary'):
        progress = LiveProgress(st, state, key, ceiling)
        try:
            report = index_project(
                runtime.project, runtime.embedder, force=force, config=runtime.config,
                on_event=progress,
                transcriber=transcriber_for_project(runtime.project, lambda: runtime.generator)
            )
            summary = (f'{len(report.indexed)} files published, {report.chunks:,} chunks; '
                       f'{report.unchanged} unchanged; {len(report.failed)} failed; '
                       f'{len(report.skipped)} skipped; {report.removed} removed.')
            progress.data.update(summary=summary, outcome='warning' if report.failed or report.skipped else 'success',
                                 problems=[f'{r.ref}: {r.problem}' for r in report.failed], notices=report.notices)
        except (CostLimitReached, KeyboardInterrupt) as error:
            progress.data.update(summary=str(error) or 'Indexing interrupted.', outcome='warning')
        except Exception as error:
            progress.data.update(summary=str(error), outcome='error')
        finally:
            progress.data['usage'] = usage_text(progress.recorder, progress.data['attempts'])
            # Streamlit's Stop/Rerun exceptions also take this path.
            progress.data.setdefault('summary', 'Indexing interrupted. Check saved progress below.')
            progress.data.setdefault('outcome', 'warning')
        status = saved_index_status(runtime.project)
    last = state.get(key)
    if last:
        st.caption('Last indexing run')
        getattr(st, last.get('outcome', 'info'))(last.get('summary', 'Run interrupted.'))
        for problem in last.get('problems', []):
            st.error(problem)
        for notice in last.get('notices', []):
            st.warning(notice)
        st.caption(last.get('usage', 'No usage recorded.'))
        if last.get('outcome') != 'success':
            st.info('Resolve the reported issue, turn off Rebuild all, then use Index now to retry. Compatible checkpoints are reused automatically. For credential or quota errors, review Settings and your provider dashboard.')
        if last.get('history'):
            with st.expander('Recent indexing activity'):
                for item in last['history']:
                    st.text(item)
    show_saved(st, status)
