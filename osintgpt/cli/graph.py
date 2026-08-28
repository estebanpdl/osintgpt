'''Graph inspection and export commands for the selected project.'''

import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from osintgpt.graph import export_graph, graph_for, verify_evidence

from .output import emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

graph_app = typer.Typer(help='Inspect or export a project knowledge graph.')


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
    state = state_from(context)
    try:
        project = resolve_project(state.home, project_slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

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
    state = state_from(context)
    try:
        project = resolve_project(state.home, project_slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

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
