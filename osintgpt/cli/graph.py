'''Graph build, traversal, verification, and export commands.'''

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.text import Text

from osintgpt.config import Settings
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.graph import (
    build_graph,
    export_graph,
    graph_for,
    neighborhood,
    neighbors,
    path_between,
    verify_evidence
)
from osintgpt.llm.usage import CostLimitReached
from osintgpt.projects import load_user_defaults

from .costs import (
    add_usage,
    fail_for_cost,
    recorder_for,
    render_usage,
    usage_data
)
from .output import console, emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

graph_app = typer.Typer(help='Build, inspect, or export a knowledge graph.')


class _LazyGenerator:
    def __init__(self, factory, effective, config, recorder) -> None:
        self.factory = factory
        self.effective = effective
        self.config = config
        self.recorder = recorder
        self._built = None

    @property
    def provider(self):
        if self._built is None:
            self._built = self.factory(
                self.effective.generation_provider,
                self.config,
                model=self.effective.generation_model or None,
                recorder=self.recorder
            )

        return self._built

    @property
    def model(self) -> str:
        return self.effective.generation_model or self.provider.model

    def generate(self, system: str, user: str) -> str:
        return self.provider.generate(system, user)


def _project(context, explicit: Optional[str], json_output: bool):
    state = state_from(context)
    try:
        return state, resolve_project(state.home, explicit)
    except ProjectSelectionError as error:
        fail(str(error), json_output)


def _edge_row(edge, depth: Optional[int] = None):
    row = asdict(edge)
    if depth is not None:
        row['depth'] = depth

    return row


@graph_app.command('build')
def build_graph_command(
    context: typer.Context,
    incremental: bool = typer.Option(
        False, '--incremental', help='Read only documents not yet in the graph.'
    ),
    rebuild: bool = typer.Option(
        False, '--rebuild', help='Re-read documents and replace their claims.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    if incremental and rebuild:
        fail('--incremental and --rebuild cannot be used together', json_output)

    state, project = _project(context, project_slug, json_output)
    try:
        defaults = load_user_defaults(state.home)
        effective = project.effective_settings(defaults)
        config = project.settings_for(Settings.from_env(), defaults)
    except (OSError, ValueError) as error:
        fail(str(error), json_output)

    from osintgpt.llm import build_generation_provider

    recorder = recorder_for(effective)
    generator = _LazyGenerator(
        build_generation_provider, effective, config, recorder
    )
    progress = None if json_output else (
        lambda ref, position, total: console.print(
            f'{position}/{total} {ref}'
        )
    )
    if not json_output:
        console.print('Graph build may make one generation call per document.')

    try:
        report = build_graph(
            project, generator, incremental=incremental, rebuild=rebuild,
            on_progress=progress
        )
    except CostLimitReached as error:
        fail_for_cost(error, recorder, json_output)
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
        'calls': report.calls,
        'documents': len(report.extracted),
        'entities': report.entities,
        'edges': report.edges,
        'skipped': report.skipped,
        'removed': report.removed,
        'failed': [asdict(result) for result in report.failed]
    }
    add_usage(data, recorder)

    def render(target) -> None:
        target.print(report.summary, style='bold')
        for result in report.failed:
            target.print(f'{result.ref}: {result.problem}', style='bold red')
        render_usage(target, recorder)

    emit(data, json_output, render)
    if report.failed:
        raise typer.Exit(code=1)


@graph_app.command('neighbors')
def neighbors_command(
    context: typer.Context,
    entity: str = typer.Argument(..., help='Entity name to inspect.'),
    depth: int = typer.Option(1, '--depth', min=1, help='Traversal depth.'),
    limit: int = typer.Option(30, '--limit', min=1, help='Maximum edges.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _project(context, project_slug, json_output)
    try:
        with graph_for(project) as graph:
            if not graph.is_built:
                fail(f'{project.slug}: graph has not been built', json_output)
            hits = (
                neighbors(graph, entity, limit=limit)
                if depth == 1
                else neighborhood(graph, entity, depth=depth, limit=limit)
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)

    rows = [_edge_row(hit.edge, hit.depth) for hit in hits]
    data = {
        'project': project.slug, 'entity': entity,
        'depth': depth, 'results': rows
    }

    def render(target) -> None:
        if not rows:
            target.print(f'No relationships found for {entity}.')
            return
        table = Table(title=f'Relationships near {entity}')
        for label in ('Depth', 'Relationship', 'Document', 'Evidence'):
            table.add_column(label)
        for row in rows:
            table.add_row(
                str(row['depth']),
                Text(
                    f'{row["source"]} —{row["relation"]}→ {row["target"]}'
                ),
                Text(row['ref']), Text(row['evidence'])
            )
        target.print(table)

    emit(data, json_output, render)


@graph_app.command('path')
def path_command(
    context: typer.Context,
    entity_a: str = typer.Argument(..., help='Starting entity.'),
    entity_b: str = typer.Argument(..., help='Destination entity.'),
    max_depth: int = typer.Option(
        4, '--max-depth', min=1, help='Longest path to consider.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _project(context, project_slug, json_output)
    try:
        with graph_for(project) as graph:
            if not graph.is_built:
                fail(f'{project.slug}: graph has not been built', json_output)
            found = path_between(
                graph, entity_a, entity_b, max_depth=max_depth
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        fail(str(error), json_output)

    rows = [_edge_row(edge, step) for step, edge in enumerate(
        found.edges if found else [], 1
    )]
    data = {
        'project': project.slug, 'source': entity_a, 'target': entity_b,
        'max_depth': max_depth, 'length': found.length if found else 0,
        'documents': found.refs if found else [], 'edges': rows
    }

    def render(target) -> None:
        if found is None:
            target.print(
                f'No path found from {entity_a} to {entity_b} '
                f'within depth {max_depth}.'
            )
            return
        if not rows:
            target.print(f'{entity_a} and {entity_b} are the same entity.')
            return
        table = Table(title=f'{entity_a} → {entity_b}')
        for label in ('Step', 'Relationship', 'Document', 'Evidence'):
            table.add_column(label)
        for row in rows:
            table.add_row(
                str(row['depth']),
                Text(
                    f'{row["source"]} —{row["relation"]}→ {row["target"]}'
                ),
                Text(row['ref']), Text(row['evidence'])
            )
        target.print(table)

    emit(data, json_output, render)


@graph_app.command('verify')
def verify_graph_command(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.'),
    strict: bool = typer.Option(
        False, '--strict', help='Exit non-zero when any evidence is unverified.'
    )
) -> None:
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
        'failures': failures
    }

    def render(target) -> None:
        target.print(f'Evidence: {report.summary}', style='bold')
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
    _, project = _project(context, project_slug, json_output)

    try:
        with graph_for(project) as graph:
            if not graph.is_built:
                fail(
                    f'{project.slug}: graph has not been built', json_output
                )
            written = export_graph(graph, path)
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
