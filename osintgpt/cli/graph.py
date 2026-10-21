'''Graph build, traversal, verification, and export commands.'''

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.text import Text

from osintgpt.credentials import resolve_credentials
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.graph import (
    GraphStore,
    build_graph,
    export_graph,
    graph_for,
    saved_graph_status,
    verify_evidence
)
from osintgpt.llm.usage import CostLimitReached
from osintgpt.llm.generation_pacing import GenerationLimits, GenerationRequestError
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.rate_settings import RateLimits
from osintgpt.graph.preflight import preflight_graph
from osintgpt.graph.model_metadata import model_label, selection_label, selection_metadata
from osintgpt.graph.migration import migrate_graph
from osintgpt.graph.windowing import ExtractionOptions
from osintgpt.projects import load_user_defaults

from .graph_generation import _LazyGenerator
from .graph_inspection import register as register_inspection
from .costs import (
    recorder_for,
    render_usage,
    usage_data
)
from .output import console, emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

graph_app = typer.Typer(help='Build, inspect, or export a knowledge graph.')



def _project(context, explicit: Optional[str], json_output: bool):
    state = state_from(context)
    try:
        return state, resolve_project(state.home, explicit)
    except ProjectSelectionError as error:
        fail(str(error), json_output)


@graph_app.command('migrate')
def migrate_graph_command(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Inspect the supported upgrade and counts without changing files.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    '''Upgrade a supported graph schema with a backup and no model calls.'''
    _, project = _project(context, project_slug, json_output)
    try:
        result = migrate_graph(project, dry_run=dry_run)
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)
    emit_record(result, json_output, title='Graph migration')


@graph_app.command('build')
def build_graph_command(
    context: typer.Context,
    incremental: bool = typer.Option(
        False, '--incremental', help='Refresh new or changed sources and resume saved work; requires an initial build.'
    ),
    rebuild: bool = typer.Option(
        False, '--rebuild', help='Discard compatible saved work and extract registered sources again.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.'),
    context_budget: int = typer.Option(32768, '--source-window-bytes', '--context-budget', min=64,
        help='Maximum serialized source text per window in UTF-8 bytes; continuation and identity context are automatic.'),
    output_tokens: Optional[int] = typer.Option(None, '--output-tokens', hidden=True),
    max_record_calls: Optional[int] = typer.Option(None, '--max-record-calls', hidden=True),
    repair_quotes: bool = typer.Option(True, '--repair-quotes/--no-repair-quotes', help='Allow source-grounded corrections for entity and relationship quotations.'),
    dry_run: bool = typer.Option(False, '--dry-run', help='Estimate work and saved progress without provider calls or changing the graph.'),
    rpm: int = typer.Option(0, '--rpm', min=0, help='Generation requests per minute; 0 adds no operator ceiling.'),
    tpm: int = typer.Option(0, '--tpm', min=0, help='Generation budget per minute: conservative input units plus output reservation; 0 adds no ceiling.'),
    retry_attempts: int = typer.Option(6, '--retry-attempts', min=1, help='Maximum HTTP attempts per request, including the initial attempt.'),
    retry_seconds: float = typer.Option(300, '--retry-seconds', min=0.001, help='Retry time budget, including waits; active generation has a separate 600-second read timeout.'),
    provider: Optional[str] = typer.Option(None, '--provider', help='Graph provider for this invocation only, including --dry-run. Supply --model unless the provider has a default; never borrows the chat model.'),
    model: Optional[str] = typer.Option(None, '--model', help='Graph extraction and quotation-correction model for this invocation only. Alone, keeps the selected graph provider. Without overrides, uses graph settings or Chat/RAG inheritance.')
) -> None:
    '''Build or refresh sourced relationships, resuming compatible saved windows.'''
    if output_tokens is not None:
        fail('--output-tokens was removed from graph builds; output allowance is now provider-managed. Remove this option.', json_output)
    if max_record_calls is not None:
        fail('--max-record-calls is retired; useful extraction continues without a record-call ceiling. Use an explicit cost ceiling or cancellation to stop.', json_output)
    if incremental and rebuild:
        fail('--incremental and --rebuild cannot be used together', json_output)

    state, project = _project(context, project_slug, json_output)
    try:
        defaults = load_user_defaults(state.home)
        effective = project.effective_settings(defaults)
        config = project.settings_for(resolve_credentials(state.home), defaults)
    except (OSError, ValueError) as error:
        fail(str(error), json_output)

    from osintgpt.llm import build_generation_provider

    recorder = recorder_for(effective)
    generator = _LazyGenerator(
        build_generation_provider, effective, config, recorder, provider=provider, model=model
    )
    progress = None if json_output else (
        lambda ref, position, total: console.print(
            f'{position}/{total} {ref}'
        )
    )
    if not json_output and not dry_run:
        console.print(
            'Graph build processes source windows with identity lookup and quotation corrections as needed. '
            'Follow-up requests have no fixed turn or record-call ceiling.'
        )

    try:
        if not json_output and not dry_run and project.settings.graph_enabled:
            console.print(Text('Graph extraction: ' + selection_label(selection_metadata(generator))))
        options = ExtractionOptions(context_budget=context_budget,
                                    repair_quotes=repair_quotes)
        limits = GenerationLimits(RateLimits(rpm=rpm, tpm=tpm), RetryPolicy(retry_attempts, retry_seconds),
                                  str(state.home / 'generation-rates.sqlite'))
        if dry_run:
            data = preflight_graph(project, generator, incremental=incremental, rebuild=rebuild,
                                   options=options, request_limits=limits)
            def render_preflight(target):
                counts = data['counts']
                target.print(Text('Graph extraction: ' + selection_label(data['graph_model'])))
                target.print(Text('Output: ' + data['output_policy_label']))
                if not data['graph_enabled']:
                    target.print('Enable graph_enabled before building this graph.')
                target.print(f"{counts['documents_to_extract']} sources to extract; "
                             f"{counts['windows_pending']} pending windows across {counts['records']} records; "
                             f"{counts['responses_saved']} saved responses.")
                target.print("Follow-up requests and future cost depend on the content; no fixed request maximum.")
                cost = data['budget_cost_estimate_usd']
                target.print('Budget-based cost estimate: unknown.' if cost is None else f'Budget-based cost estimate: ${cost:.6f}.')
                target.print(f"{counts['documents_unchanged']} completed sources reused; "
                             f"{counts['documents_model_changed']} completed sources have a changed graph model.")
                target.print(f"{counts['checkpoints_incompatible']} incompatible saved source extractions "
                             f"({counts['windows_incompatible']} windows, {counts['responses_incompatible']} responses).")
                for detail in data['details']:
                    if detail.get('reason'):
                        target.print(Text(f"{detail['ref']}: {detail['action']} — {detail['reason']}"))
                    checkpoint = detail.get('checkpoint', {})
                    if checkpoint.get('present') and not checkpoint['compatible']:
                        target.print(Text(f"  Saved work cannot be reused: {model_label(checkpoint.get('previous_model'))} "
                                          f"-> {model_label(checkpoint.get('next_model'))}; "
                                          f"source changed: {checkpoint.get('source_changed', False)}."))
                if counts['documents_unavailable']:
                    target.print(f"{counts['documents_unavailable']} sources could not be read; use --json for details.")
                for note in data['assumptions']:
                    target.print(note)
            emit(data, json_output, render_preflight)
            return
        report = build_graph(
            project, generator, incremental=incremental, rebuild=rebuild,
            on_progress=progress, options=options, request_limits=limits,
            on_event=None if json_output else _build_event
        )
    except CostLimitReached as error:
        status = saved_graph_status(project)
        fail(str(error), json_output, {'usage': status['usage'], 'graph': status})
    except GenerationRequestError as error:
        status = saved_graph_status(project)
        fail(str(error), json_output, {'usage': status['usage'], 'graph': status})
    except (
        ImportError, MissingEnvironmentVariableError, OSError,
        sqlite3.Error, ValueError
    ) as error:
        fail(str(error), json_output, {'usage': usage_data(recorder)})

    if report.refused:
        fail(
            report.refused, json_output,
            {'usage': usage_data(recorder)}
        )

    data = {
        'project': project.slug,
        'summary': report.summary,
        'status': report.status,
        'calls': report.calls,
        'accounting': report.accounting.to_dict(),
        'usage': report.usage,
        'documents': len(report.extracted),
        'entities': report.entities,
        'edges': report.edges,
        'skipped': report.skipped,
        'removed': report.removed,
        'details': report.details,
        'failed': [asdict(result) for result in report.failed]
    }

    def render(target) -> None:
        target.print(f'{report.status}: {report.summary}', style='bold')
        for detail in report.details:
            if detail['action'] != 'reused':
                target.print(Text(f"{detail['ref']}: {detail['reason']}"))
        for result in report.failed:
            target.print(f'{result.ref}: {result.problem}', style='bold red')
        render_usage(target, recorder)
        from osintgpt.graph.accounting import usage_messages
        for message in usage_messages(report.usage):
            target.print(message)

    emit(data, json_output, render)
    if report.failed:
        raise typer.Exit(code=1)


def _build_event(event):
    if event.kind == 'started':
        console.print(Text('Output: ' + event.data['output_policy_label']))
    elif event.kind == 'retry':
        console.print(Text(f"Retry {event.data['attempt']}/{event.data['maximum']} "
                           f"after at least {event.data['seconds']:.1f}s ({event.data['ref']})."))
    elif event.kind == 'window_saved':
        console.print(Text(f"  Window {event.data['window']}/{event.data['total']} saved."))
    elif event.kind == 'response_reused':
        console.print(Text(f"  Reusing saved {event.data['purpose']} response for window {event.data['window']}."))
    elif event.kind == 'response_recovered':
        console.print(Text(f"  Recovering legacy {event.data['purpose']} work for window {event.data['window']}; local validation follows."))
    elif event.kind in {'finding_identities', 'retrieving_evidence', 'reorganizing_context', 'completing_extraction', 'correcting_structure', 'partitioning_source'}:
        console.print(Text('  ' + event.kind.replace('_', ' ').capitalize() + '.'))


@graph_app.command('rejected')
def rejected_graph_command(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
    ref: Optional[str] = typer.Option(None, '--ref', help='Limit to a source file or record.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    '''Inspect rejected candidates, field warnings, and corrections without model calls.

    Relation polarity (relation_polarity) retains positive, negative, or unknown, not sentiment. Date fields
    retain source wording; unavailable dates stay empty. Warnings retain supported
    entities and are distinct from final rejections.
    '''
    _, project = _project(context, project_slug, json_output)
    rows = []
    if (project.paths.root / 'graph.sqlite').is_file():
        try:
            with graph_for(project) as graph:
                rows = [asdict(item) for item in graph.rejections([ref] if ref else None)]
        except (OSError, sqlite3.Error, ValueError) as error:
            fail(str(error), json_output)
    def render(target):
        target.print(f'{len(rows)} extraction audit records')
        for row in rows:
            target.print(Text(f"{row['ref']} [{row['status']}]: {row['reason']}"))
    emit({'project': project.slug, 'results': rows}, json_output, render)


@graph_app.command('status')
def graph_status_command(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    '''Inspect saved build status and usage without opening a model provider.'''
    _, project = _project(context, project_slug, json_output)
    try:
        status = saved_graph_status(project)
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)
    if not json_output:
        status['details'] = f"{len(status.get('details', []))} source decisions; use --json for details."
        status['previous_runs'] = f"{len(status.get('previous_runs', []))} previous runs; use --json for usage."
        models = status['extraction_models']
        names = sorted({model_label(row['identity']) for row in models['sources'] if row['known']})
        status['extraction_models'] = (f"{'; '.join(names) or 'none recorded'}; "
            f"{models['unknown_sources']} sources with unknown model; "
            f"{models['unattributed_entities']} unattributed entities. Use --json for source identities.")
        status['run_model'] = selection_label(status['run_model']) if status['run_model'] else 'unknown (not recorded)'
    emit_record(
        {'project': project.slug, 'enabled': project.settings.graph_enabled, **status},
        json_output, title='Graph status'
    )


@graph_app.command('verify')
def verify_graph_command(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.'),
    strict: bool = typer.Option(
        False, '--strict', help='Exit non-zero when evidence is missing or unreadable in the current source.'
    )
) -> None:
    '''Check stored quotations against saved snapshots and current source text.'''
    _, project = _project(context, project_slug, json_output)

    try:
        report = verify_evidence(project)
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)

    failures = [
        {**asdict(result.edge), 'status': result.status,
         'problem': result.problem}
        for result in report.failures
    ]
    data = {
        'project': project.slug,
        'summary': report.summary,
        'total': report.total,
        'found': report.found,
        'not_found': report.not_found,
        'unreadable': report.unreadable,
        'failures': failures,
        'results': [asdict(result) for result in report.results],
        'changed_sources': sum(result.source_state in ('changed', 'transcription_changed')
                               for result in report.results),
        'snapshot_found': sum(result.snapshot_status == 'found' for result in report.results)
    }

    def render(target) -> None:
        target.print(f'Evidence: {report.summary}', style='bold')
        if data['changed_sources']:
            target.print(f"{data['changed_sources']} claims refer to a changed source; "
                         'use --json to compare saved and current text verification.')
        if not failures:
            return
        table = Table(title='Unverified evidence')
        table.add_column('Document')
        table.add_column('Status')
        table.add_column('Claim')
        table.add_column('Problem')
        for failure in failures:
            table.add_row(
                failure['ref'], failure['status'],
                f'{failure["source"]} — {failure["relation"]} → '
                f'{failure["target"]}', failure['problem']
            )
        target.print(table)

    emit(data, json_output, render)
    if strict and report.failures:
        raise typer.Exit(code=1)


@graph_app.command('export')
def export_graph_command(
    context: typer.Context,
    path: Path = typer.Argument(
        ..., help='Destination ending in .cypherl or .json.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    '''Export claims, evidence, verification, and build context to .json or .cypherl.'''
    _, project = _project(context, project_slug, json_output)

    try:
        if not (project.paths.root / 'graph.sqlite').is_file():
            fail(f'{project.slug}: graph has not been built', json_output)
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            if not graph.is_built:
                fail(
                    f'{project.slug}: graph has not been built', json_output
                )
            written = export_graph(graph, path, project=project)
            data = {
                'project': project.slug,
                'path': str(written),
                'format': written.suffix.lower().lstrip('.'),
                'entities': graph.entity_count,
                'edges': graph.edge_count
            }
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)

    emit_record(data, json_output, title='Graph exported')


register_inspection(graph_app, _project)
