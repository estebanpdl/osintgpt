'''Project creation, discovery, selection, inspection, and deletion.'''

import shutil
from typing import Optional

import typer

from osintgpt.projects import Project, Registry

from .output import emit_record, emit_rows, fail
from .selection import (
    ProjectSelectionError,
    clear_selection,
    resolve_project,
    state_from,
    write_selection
)

project_app = typer.Typer(
    help='Create, list, inspect, select, and delete projects.'
)


def _project_data(project: Project) -> dict:
    return {
        'id': project.id,
        'slug': project.slug,
        'name': project.name,
        'path': str(project.paths.root),
        'created_at': project.created_at,
        'settings': project.settings.to_dict()
    }


@project_app.command('create')
def create_project(
    context: typer.Context,
    name: str = typer.Argument(..., help='Display name for the project.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    try:
        project = Project.create(name, home=state.home)
    except (FileExistsError, OSError, ValueError) as error:
        fail(str(error), json_output)

    Registry.load(state.home).register(project)
    emit_record(_project_data(project), json_output, title='Project created')


@project_app.command('list')
def list_projects(
    context: typer.Context,
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    rows = [
        {
            'id': entry.id,
            'slug': entry.slug,
            'name': entry.name,
            'path': entry.path,
            'embedding_model': entry.embedding_model
        }
        for entry in Registry.load(state.home)
    ]
    emit_rows(
        rows, json_output,
        (('slug', 'Slug'), ('name', 'Name'), ('path', 'Path')),
        title='Projects'
    )


@project_app.command('show')
def show_project(
    context: typer.Context,
    slug: Optional[str] = typer.Argument(None, help='Project slug or id.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    try:
        project = resolve_project(state.home, slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

    emit_record(_project_data(project), json_output, title=project.name)


@project_app.command('use')
def use_project(
    context: typer.Context,
    slug: str = typer.Argument(..., help='Project slug or id.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    try:
        project = resolve_project(state.home, slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

    write_selection(state.home, project.slug)
    emit_record(
        {'selected': project.slug, 'path': str(project.paths.root)},
        json_output,
        title='Selected project'
    )


@project_app.command('delete')
def delete_project(
    context: typer.Context,
    slug: str = typer.Argument(..., help='Project slug or id.'),
    yes: bool = typer.Option(
        False, '--yes', help='Confirm permanent deletion.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    if not yes:
        fail('project deletion requires --yes and never prompts', json_output)

    try:
        project = resolve_project(state.home, slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

    try:
        shutil.rmtree(project.paths.root)
    except OSError as error:
        fail(f'could not delete {project.slug}: {error}', json_output)

    Registry.load(state.home).unregister(project.slug)
    clear_selection(state.home, project.slug)
    emit_record(
        {'deleted': project.slug, 'path': str(project.paths.root)},
        json_output,
        title='Project deleted'
    )
