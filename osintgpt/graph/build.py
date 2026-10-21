'''Explicit graph construction with persistent status and per-request accounting.'''

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from osintgpt.ingestion.snapshots import load_snapshot
from osintgpt.llm.base import GenerationProvider
from osintgpt.llm.generation_pacing import GenerationRequests, GenerationRequestError
from osintgpt.projects import Project

from .accounting import BuildCounts, recorded_usage
from .checkpoints import Checkpoints
from .extraction import Extraction, extract_document
from .events import observe_graph, emit_graph, check_cancelled
from .recipes import extraction_recipe, fingerprint, compatible_recipe
from .model_metadata import selection_metadata
from .reconciliation import inventory, refresh_reason, validate_current
from .state import build_lock, freshness, transaction
from .store import GraphStore, graph_for
from .windowing import ExtractionOptions
from .output_policy import prepare_policy, policy_data

log = logging.getLogger('osintgpt.graph')
Progress = Callable[[str, int, int], None]

NOT_ENABLED = (
    '{slug}: the graph is off for this project. Building uses generation calls '
    'per document window and must be requested explicitly. '
    'Enable it with graph_enabled, then build it.'
)
NOT_BUILT = (
    '{slug}: the graph has not been built yet, so there is nothing to update. '
    'Build it once explicitly first.'
)


@dataclass
class GraphReport:
    '''Results and observed work for one build invocation.'''
    extracted: List[Extraction] = field(default_factory=list)
    failed: List[Extraction] = field(default_factory=list)
    skipped: int = 0
    removed: int = 0
    refused: str = ''
    status: str = 'never_built'
    accounting: BuildCounts = field(default_factory=BuildCounts)
    usage: dict = field(default_factory=dict)
    details: list = field(default_factory=list)

    @property
    def entities(self) -> int:
        return sum(len(e.entities) for e in self.extracted)

    @property
    def edges(self) -> int:
        return sum(len(e.edges) for e in self.extracted)

    @property
    def calls(self) -> int:
        return self.accounting.requests_attempted

    @property
    def summary(self) -> str:
        if self.refused:
            return self.refused
        counts = self.accounting
        parts = [
            f'{len(self.extracted)} documents, {self.entities} entities, {self.edges} edges',
            f'{counts.records_loaded} records read',
            f'{counts.windows_completed}/{counts.windows_total} windows',
            f'{counts.windows_reused} saved windows reused',
            f'{counts.requests_attempted} generation calls '
            f'({counts.requests_succeeded} responses, {counts.requests_failed} failed)',
        ]
        if self.skipped:
            parts.append(f'{self.skipped} previously processed')
        if self.removed:
            parts.append(f'{self.removed} edges removed')
        if counts.documents_removed:
            parts.append(f'{counts.documents_removed} sources removed')
        if self.failed:
            parts.append(f'{len(self.failed)} documents failed')
        if counts.invalid_responses:
            parts.append(f'{counts.invalid_responses} invalid responses')
        if counts.entity_lookups or counts.entity_lookups_reused:
            parts.append(f'{counts.entity_lookups} local entity lookups, {counts.entity_lookups_reused} saved lookups reused')
        if counts.responses_reused or counts.responses_recovered or counts.context_restarts:
            parts.append(f'{counts.responses_reused} exact responses reused, {counts.responses_recovered} legacy responses recovered; '
                         f'{counts.context_restarts} context restarts')
        if counts.completion_requests or counts.structured_correction_requests:
            parts.append(f'{counts.completion_requests} finalization and {counts.structured_correction_requests} structured-correction requests (included above)')
        if counts.field_warnings or counts.identity_unresolved or counts.repaired_entities:
            parts.append(f'{counts.field_warnings} field/identity warnings, {counts.identity_unresolved} unresolved identities; '
                         f'{counts.repaired_entities} entity quotations corrected')
        if counts.rejected_claims or counts.rejected_entities or counts.repaired_claims:
            parts.append(f'{counts.rejected_claims} claims and {counts.rejected_entities} entities rejected; '
                         f'{counts.repaired_claims} quotations corrected')
        return ', '.join(parts)


def build_graph(
    project: Project,
    generator: GenerationProvider,
    incremental: bool = False,
    rebuild: bool = False,
    store: Optional[GraphStore] = None,
    on_progress: Optional[Progress] = None,
    options: Optional[ExtractionOptions] = None,
    *, request_limits=None, on_event=None, should_cancel=None
) -> GraphReport:
    '''Refresh sources and resume compatible work; rebuild discards saved windows.'''
    if incremental and rebuild:
        raise ValueError('incremental and rebuild cannot be used together')
    if not project.settings.graph_enabled:
        return GraphReport(refused=NOT_ENABLED.format(slug=project.slug))

    with build_lock(store.path if store is not None else project.paths.root / 'graph.sqlite'):
        graph = store or graph_for(project)
        try:
            if incremental and not graph.is_built:
                return GraphReport(refused=NOT_BUILT.format(slug=project.slug))
            def observe(event):
                graph.state.event(event)
                if on_event:
                    on_event(event)

            with observe_graph(observe, should_cancel):
                return _run(project, generator, graph, rebuild, on_progress, options,
                            GenerationRequests(request_limits))
        finally:
            if store is None:
                graph.close()


def _run(project, generator, graph, rebuild, on_progress, options, requests):
    report = GraphReport(status='building')
    entries = [*graph.state.entries().values(), *Checkpoints(graph).jobs().values()]
    policy = prepare_policy(generator, entries, resolve=True)
    options = policy.options(options or ExtractionOptions())
    recorder = getattr(generator, 'recorder', None)
    offset = len(recorder) if recorder is not None else 0
    recipe_data = extraction_recipe(generator, options)
    recipe = fingerprint(recipe_data)
    graph_model = selection_metadata(generator, recipe_data['provider'])

    def save_progress():
        report.usage = recorded_usage(generator, report.accounting, offset)
        report.usage['extraction_budget'] = asdict(options or ExtractionOptions())
        report.usage['request_limits'] = {'rates': asdict(requests.limits.rates), 'retry': asdict(requests.limits.retry)}
        report.usage['input_estimator'] = 'utf8_bytes_conservative'
        report.usage['output_policy'] = policy_data(generator)
        report.usage['pacing_estimator'] = 'input_bytes_plus_model_output' if policy.max_tokens else 'twice_input_bytes'
        report.usage['graph_model'] = graph_model
        graph.state.progress(report.accounting, report.usage, report.details)
        emit_graph('progress', counts=report.accounting.to_dict(), usage=report.usage)

    graph.state.begin(report.accounting)
    try:
        save_progress()
        emit_graph('started', graph_model=graph_model, output_policy=policy_data(generator), output_policy_label=policy.label())
        check_cancelled()
        corpus, targets, missing, removed, images = inventory(project, graph)
        report.accounting.images_skipped = images
        report.accounting.documents_total = len(targets) + len(missing)
        journal = Checkpoints(graph)
        for ref in removed:
            check_cancelled()
            with transaction(graph.connection):
                report.removed += graph.forget([ref])
                journal.discard(ref)
            report.accounting.documents_removed += 1
            report.details.append({'ref': ref, 'action': 'removed', 'reason': 'Source is no longer registered'})
            emit_graph('source_removed', ref=ref)
        for ref in missing:
            _failed(report, ref, 'Registered source is unavailable; published claims and checkpoints were preserved.')
        completed, pending = graph.state.entries(), journal.jobs()
        mappings = corpus.mappings(project.paths.root)
        source_details = {ref: {'ref': ref, 'action': 'pending', 'reason': 'Waiting to process this source.',
                                'previous_published': ref in completed} for ref in targets}
        report.details.extend(source_details.values())
        save_progress()

        for position, (ref, path) in enumerate(targets.items(), 1):
            check_cancelled()
            detail = source_details[ref]
            detail['action'] = 'extracting'
            emit_graph('source', ref=ref, position=position, total=len(targets))
            if on_progress:
                try:
                    on_progress(ref, position, len(targets))
                except Exception:
                    log.exception('Graph progress observer failed')
            try:
                mapping = mappings.get(path.resolve())
                snapshot = load_snapshot(path, mapping, ref=ref, cache_dir=project.paths.extracts)
            except Exception as error:
                log.warning('%s could not be read: %s', ref, error)
                detail['action'] = 'failed'
                _failed(report, ref, str(error), detail)
                save_progress()
                continue

            previous, job = completed.get(ref), pending.get(ref)
            reason = refresh_reason(previous, snapshot, recipe, recipe_data)
            compatible = job and job['source_version'] == snapshot.version and compatible_recipe(job['recipe_data'], recipe_data)
            if not rebuild and not reason and not job:
                report.skipped += 1
                report.accounting.documents_skipped += 1
                detail.update(action='reused', reason='Source and extraction settings unchanged')
                save_progress()
                emit_graph('source_reused', ref=ref)
                continue
            if previous and reason:
                report.accounting.documents_changed += 1
            if rebuild or not compatible:
                if job:
                    report.accounting.checkpoints_discarded += 1
                    reason = ('Explicit rebuild' if rebuild else
                              refresh_reason(job, snapshot, recipe, recipe_data)) + '; saved windows discarded'
                journal.prepare(ref, snapshot.version, recipe, recipe_data)
            elif compatible:
                reason = 'Continuing saved extraction windows'
            detail.update({'ref': ref, 'action': 'extracting', 'reason': 'Explicit rebuild' if rebuild else reason,
                      'previous_published': bool(previous),
                      'checkpoint_discarded': bool(job and (rebuild or not compatible)),
                      'checkpoint_cause': 'explicit_rebuild' if rebuild else 'source_or_settings'})
            save_progress()
            try:
                entities, edges, quality_gaps = _extract_source(
                    generator, journal, snapshot, options, report, save_progress, requests)
                check_cancelled()
                validate_current(project, path, snapshot)
                from .publication import validate_identities
                entities, edges, identity_warnings = validate_identities(graph, snapshot, entities, edges)
                detail['identity_revalidation'] = identity_warnings
                report.accounting.identity_unresolved += len(identity_warnings)
                if snapshot.missing_pages:
                    raise ValueError(f'Missing extracted text on pages {snapshot.missing_pages}; '
                                     'index the PDF to transcribe these pages, then rerun graph build.')
                report.removed += graph.replace_source(snapshot, entities, edges, quality_gaps, recipe, recipe_data)
            except GenerationRequestError as error:
                detail['action'] = 'failed'
                _failed(report, ref, str(error), detail)
                raise
            except Exception as error:
                detail['action'] = 'failed'
                _failed(report, ref, str(error), detail)
            else:
                detail['action'] = 'published'
                report.accounting.documents_completed += 1
                report.extracted.append(Extraction(ref=ref, entities=entities, edges=edges))
                emit_graph('source_published', ref=ref)
            save_progress()

        if report.failed or graph.state.has_quality_gaps():
            report.status = (
                'complete_with_gaps' if report.extracted or report.skipped or graph.is_built
                or any('Missing extracted text' in item.problem for item in report.failed) else 'failed'
            )
        else:
            report.status = 'complete'
        problem = '; '.join(f'{item.ref}: {item.problem}' for item in report.failed)
        if graph.state.has_quality_gaps():
            problem += ('; ' if problem else '') + 'Some candidates were rejected; inspect graph rejected --json.'
        graph.state.finish(report.status, problem)
        if report.status == 'complete':
            try:
                current = freshness(project, graph.state.documents(), graph.state.versions())
                problem = ''
            except (OSError, ValueError) as error:
                current, problem = False, str(error)
            if not current:
                report.status = 'stale'
                graph.state.finish(report.status, problem)
        emit_graph('finished', status=report.status, counts=report.accounting.to_dict())
        return report
    except BaseException as error:
        report.status = 'failed' if isinstance(error, Exception) else 'interrupted'
        for detail in report.details:
            if detail.get('action') == 'extracting':
                detail.update(action=report.status, problem=str(error) or type(error).__name__)
        save_progress()
        graph.state.finish(report.status, str(error) or type(error).__name__)
        emit_graph('finished', status=report.status, counts=report.accounting.to_dict())
        raise


def _failed(report, ref, problem, detail=None):
    emit_graph('source_failed', ref=ref, problem=problem)
    report.failed.append(Extraction(ref=ref, problem=problem))
    report.accounting.documents_failed += 1
    if detail is None:
        report.details.append({'ref': ref, 'action': 'failed', 'reason': problem})
    else:
        detail['problem'] = problem


def _extract_source(generator, journal, snapshot, options, report, save_progress, requests):
    report.accounting.records_loaded += len(snapshot.documents)
    entities, edges, quality_gaps = [], [], False
    known = journal.seed(snapshot.ref)
    for document in snapshot.documents:
        check_cancelled()
        emit_graph('record', ref=document.ref)
        before = report.accounting

        def extraction_progress(counts):
            report.accounting = before.including(counts)
            save_progress()

        readable = any(not span.missing for span in document.spans)
        result = extract_document(generator, document.ref, document.text if readable else '',
                                  on_progress=extraction_progress, snapshot=snapshot, document=document,
                                  known_entities=known + entities, options=options, checkpoints=journal, requests=requests)
        quality_gaps |= any(item.status == 'rejected' for item in result.rejected)
        if not result.ok:
            raise ValueError(result.problem)
        entities.extend(result.entities)
        edges.extend(result.edges)
    return entities, edges, quality_gaps
