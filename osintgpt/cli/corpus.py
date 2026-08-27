'''Top-level corpus registration, inspection, removal, and indexing commands.'''

import logging
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.table import Table

from osintgpt import Settings, index_project
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.ingestion import (
    Corpus,
    FieldMapping,
    IndexState,
    describe_fields
)
from osintgpt.ingestion.loaders import needs_mapping
from osintgpt.llm import build_embedding_provider
from osintgpt.projects import load_user_defaults

from .output import console, emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

LIST_KEYS = {'content', 'metadata'}
SINGLE_KEYS = {'timestamp', 'author', 'identity', 'records'}


def _mapping(pairs: Optional[List[str]]) -> FieldMapping:
    roles: Dict[str, object] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition('=')
        key = key.strip()
        if not separator or not value.strip():
            raise ValueError(f'--map {pair!r} should look like key=value')

        if key in LIST_KEYS:
            values = tuple(
                item.strip() for item in value.split(',') if item.strip()
            )
            roles[key] = tuple(roles.get(key, ())) + values
        elif key in SINGLE_KEYS:
            roles[key] = value.strip()
        else:
            valid = ', '.join(sorted(LIST_KEYS | SINGLE_KEYS))
            raise ValueError(f'unknown mapping key {key!r}; use one of: {valid}')

    return FieldMapping(**roles)


def _open_project(
    context: typer.Context,
    explicit: Optional[str],
    json_output: bool
):
    state = state_from(context)
    try:
        return state, resolve_project(state.home, explicit)
    except ProjectSelectionError as error:
        fail(str(error), json_output)


def _source_key(project, path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project.paths.root.resolve())
    except ValueError:
        return resolved


def add_source(
    context: typer.Context,
    path: Path = typer.Argument(..., help='File or folder to register.'),
    maps: Optional[List[str]] = typer.Option(
        None, '--map', help='Field role as key=value; repeatable.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _open_project(context, project_slug, json_output)
    if not path.exists():
        fail(f'no such path: {path}', json_output)

    try:
        mapping = _mapping(maps)
    except ValueError as error:
        fail(str(error), json_output)

    if path.is_file() and needs_mapping(path) and not mapping.is_set:
        fields = describe_fields(path)
        command = (
            f'osintgpt add "{path}" --project {project.slug} '
            '--map content=<field>'
        )
        fail(
            f'{path.name} needs a content field mapping', json_output,
            {'fields': fields, 'try': command}
        )

    source = Corpus.load(project.paths.sources).register(
        _source_key(project, path), mapping
    )
    emit_record(
        {'project': project.slug, **source.to_dict()},
        json_output,
        title='Source registered'
    )


def list_sources(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _open_project(context, project_slug, json_output)
    corpus = Corpus.load(project.paths.sources)
    state = IndexState.load(project.paths.index_state)
    data = {
        'project': project.slug,
        'sources': [source.to_dict() for source in corpus],
        'covered_files': [
            path.as_posix() for path in corpus.files(project.paths.root)
        ],
        'indexed_documents': [
            {
                'ref': document.ref,
                'chunks': document.chunks,
                'indexed_at': document.indexed_at
            }
            for document in sorted(
                state.documents.values(), key=lambda item: item.ref
            )
        ]
    }

    def render(target) -> None:
        target.print(f'{project.name} sources', style='bold')
        if not data['sources']:
            target.print('No sources registered.')
        for source in data['sources']:
            target.print(source['path'], soft_wrap=True)
            if source.get('fields'):
                target.print(f'  fields: {source["fields"]}')
        target.print(f'{len(data["covered_files"])} files currently covered')

        if data['indexed_documents']:
            index_table = Table(title='Indexed documents')
            index_table.add_column('Ref')
            index_table.add_column('Chunks')
            index_table.add_column('Indexed at')
            for document in data['indexed_documents']:
                index_table.add_row(
                    document['ref'], str(document['chunks']),
                    document['indexed_at']
                )
            target.print(index_table)

    emit(data, json_output, render)


def remove_source(
    context: typer.Context,
    path: Path = typer.Argument(..., help='Registered file or folder.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _open_project(context, project_slug, json_output)
    key = _source_key(project, path)
    if not Corpus.load(project.paths.sources).unregister(key):
        fail(f'source {key.as_posix()!r} is not registered', json_output)

    emit_record(
        {'project': project.slug, 'removed': key.as_posix()},
        json_output,
        title='Source removed'
    )


def index_corpus(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    force: bool = typer.Option(
        False, '--force', help='Re-embed every document.'
    ),
    purge_other_models: bool = typer.Option(
        False, '--purge-other-models', help='Remove vectors from other models.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state, project = _open_project(context, project_slug, json_output)
    defaults = load_user_defaults(state.home)
    project_settings = project.effective_settings(defaults)
    config = project.settings_for(Settings.from_env(), defaults)

    try:
        embedder = build_embedding_provider(
            project_settings.embedding_provider, config
        )
    except (ImportError, MissingEnvironmentVariableError, ValueError) as error:
        fail(str(error), json_output)

    progress = None if json_output else (
        lambda ref, position, total: console.print(
            f'{position}/{total} {ref}'
        )
    )
    index_log = logging.getLogger('osintgpt.indexing')
    was_disabled = index_log.disabled
    if json_output:
        index_log.disabled = True
    try:
        try:
            report = index_project(
                project,
                embedder,
                force=force,
                purge_other_models=purge_other_models,
                on_progress=progress,
                config=config
            )
        except (
            ImportError, MissingEnvironmentVariableError, OSError, ValueError
        ) as error:
            fail(str(error), json_output)
    finally:
        index_log.disabled = was_disabled

    data = {
        'project': project.slug,
        'summary': report.summary,
        'embedding_model': report.embedding_model,
        'indexed': [vars(result) for result in report.indexed],
        'failed': [vars(result) for result in report.failed],
        'unchanged': report.unchanged,
        'removed': report.removed,
        'purged': report.purged
    }

    def render(target) -> None:
        target.print(report.summary)
        for failed_result in report.failed:
            target.print(
                f'{failed_result.ref}: {failed_result.problem}', style='bold red'
            )

    emit(data, json_output, render)
    if report.failed:
        raise typer.Exit(code=1)


def register_corpus_commands(app: typer.Typer) -> None:
    app.command('add')(add_source)
    app.command('sources')(list_sources)
    app.command('remove')(remove_source)
    app.command('index')(index_corpus)
