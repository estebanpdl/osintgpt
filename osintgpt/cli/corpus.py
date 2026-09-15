'''Top-level corpus registration, inspection, removal, and indexing commands.'''

import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table

from osintgpt import index_project
from osintgpt.credentials import resolve_credentials
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.ingestion import (
    Corpus,
    FieldMapping,
    IndexState,
    describe_fields,
    preview_mapping
)
from osintgpt.ingestion.loaders import needs_mapping
from osintgpt.ingestion.transcription import transcriber_for_project
from osintgpt.llm import build_embedding_provider, build_generation_provider
from osintgpt.llm.usage import CostLimitReached
from osintgpt.projects import load_user_defaults

from .costs import (
    add_usage,
    fail_for_cost,
    recorder_for,
    render_usage,
    usage_data
)
from .mapping import MAPPING_EXAMPLE, build_mapping
from .output import console, emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from


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
        None, '--map',
        help='Field role as key=value; repeatable. Keys: content, metadata '
             '(comma-separated lists), timestamp, author, identity, records, '
             'min_chars. The first content field is the primary one.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    dry_run: bool = typer.Option(
        False, '--dry-run',
        help='Report what the mapping would keep and discard; register '
             'nothing. Structured files only.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project = _open_project(context, project_slug, json_output)
    if not path.exists():
        fail(f'no such path: {path}', json_output)

    try:
        mapping = build_mapping(maps)
    except ValueError as error:
        fail(str(error), json_output)

    if path.is_file() and needs_mapping(path) and not mapping.is_set:
        fields = describe_fields(path)
        command = (
            f'osintgpt add "{path}" --project {project.slug} {MAPPING_EXAMPLE}'
        )
        fail(
            f'{path.name} needs a content field mapping', json_output,
            {'fields': fields, 'try': command}
        )

    if dry_run:
        if not (path.is_file() and needs_mapping(path)):
            fail(
                '--dry-run reports on a structured file; '
                f'{path.name} is not one', json_output
            )
        _report_mapping(path, mapping, json_output)

        return

    source = Corpus.load(project.paths.sources).register(
        _source_key(project, path), mapping
    )
    emit_record(
        {'project': project.slug, **source.to_dict()},
        json_output,
        title='Source registered'
    )


def _report_mapping(path: Path, mapping: FieldMapping, json_output: bool) -> None:
    preview = preview_mapping(path, mapping)
    data = {
        'file': path.name,
        'fields': mapping.to_dict(),
        'records': preview.records,
        'kept': preview.kept,
        'without_content': preview.without_content,
        'too_short': preview.too_short,
        'shortest': preview.shortest,
        'example': preview.example,
        'summary': preview.summary
    }

    def render(target) -> None:
        target.print(f'{path.name} — nothing registered', style='bold')
        target.print(preview.summary)
        if preview.example:
            target.print('shortest kept record:', style='dim')
            target.print(preview.example, soft_wrap=True)

    emit(data, json_output, render)


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
    config = project.settings_for(resolve_credentials(state.home), defaults)
    recorder = recorder_for(project_settings)

    try:
        embedder = build_embedding_provider(
            project_settings.embedding_provider, config, recorder=recorder
        )
    except (ImportError, MissingEnvironmentVariableError, ValueError) as error:
        fail(str(error), json_output, {'usage': usage_data(recorder)})

    # Built on first use, so indexing born-digital documents never asks for
    # a generation credential. The ingestion model falls back to the
    # generation one, which is what the setting's help promises.
    transcriber = transcriber_for_project(
        project,
        lambda: build_generation_provider(
            project_settings.generation_provider,
            config,
            model=(
                project_settings.ingestion_model
                or project_settings.generation_model
                or None
            ),
            recorder=recorder
        )
    )

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
                config=config,
                transcriber=transcriber
            )
        except CostLimitReached as error:
            fail_for_cost(error, recorder, json_output)
        except (
            ImportError, MissingEnvironmentVariableError, OSError, ValueError
        ) as error:
            fail(str(error), json_output, {'usage': usage_data(recorder)})
    finally:
        index_log.disabled = was_disabled

    data = {
        'project': project.slug,
        'summary': report.summary,
        'embedding_model': report.embedding_model,
        'indexed': [vars(result) for result in report.indexed],
        'failed': [vars(result) for result in report.failed],
        'skipped': [vars(result) for result in report.skipped],
        'notices': report.notices,
        'adopted': report.adopted,
        'unchanged': report.unchanged,
        'removed': report.removed,
        'purged': report.purged
    }
    add_usage(data, recorder)

    def render(target) -> None:
        target.print(report.summary)
        for failed_result in report.failed:
            target.print(
                f'{failed_result.ref}: {failed_result.problem}', style='bold red'
            )
        for notice in report.notices:
            target.print(notice, style='yellow')
        render_usage(target, recorder)

    emit(data, json_output, render)
    if report.failed:
        raise typer.Exit(code=1)


def register_corpus_commands(app: typer.Typer) -> None:
    app.command(
        'add', help='Register a file or folder as material to index.'
    )(add_source)
    app.command(
        'sources', help='Show what is registered and what has been indexed.'
    )(list_sources)
    app.command(
        'remove', help='Unregister a source. The files stay on disk.'
    )(remove_source)
    app.command(
        'index', help='Index new and changed documents.'
    )(index_corpus)
