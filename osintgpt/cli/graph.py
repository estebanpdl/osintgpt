'''Graph export commands for the selected project.'''

import sqlite3
from pathlib import Path
from typing import Optional

import typer

from osintgpt.graph import export_graph, graph_for

from .output import emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

graph_app = typer.Typer(help='Export a project knowledge graph.')


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
