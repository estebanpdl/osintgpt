'''Grounded answers and raw semantic hits from the selected project.'''

from contextlib import contextmanager
from typing import Dict, Optional

import typer

from osintgpt import Settings, answer_question, search_project
from osintgpt.answering import DEFAULT_PASSAGES
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.llm import (
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.projects import load_user_defaults
from osintgpt.vector_store import SearchResult, store_for

from .output import emit, fail
from .selection import ProjectSelectionError, resolve_project, state_from

PREVIEW_CHARS = 700


def _runtime(context: typer.Context, explicit: Optional[str], json_output: bool):
    state = state_from(context)
    try:
        project = resolve_project(state.home, explicit)
        defaults = load_user_defaults(state.home)
        effective = project.effective_settings(defaults)
        config = project.settings_for(Settings.from_env(), defaults)
    except (ProjectSelectionError, OSError, ValueError) as error:
        fail(str(error), json_output)

    return project, effective, config


def _embedder(effective, config, json_output: bool):
    try:
        return build_embedding_provider(
            effective.embedding_provider,
            config,
            model=effective.embedding_model or None
        )
    except (ImportError, MissingEnvironmentVariableError, ValueError) as error:
        fail(str(error), json_output)


class _LazyGenerator:
    '''Build the generator only if grounded retrieval found a passage.'''

    def __init__(self, effective, config) -> None:
        self.effective = effective
        self.config = config

    def generate(self, system: str, user: str) -> str:
        generator = build_generation_provider(
            self.effective.generation_provider,
            self.config,
            model=self.effective.generation_model or None
        )

        return generator.generate(system, user)


@contextmanager
def _store(project, config):
    engine = store_for(project, config)
    try:
        yield engine
    finally:
        close = getattr(engine, 'close', None)
        if close is not None:
            close()


def _result(result: SearchResult, rank: int) -> Dict[str, object]:
    return {
        'rank': rank,
        'score': result.score,
        'citation': result.chunk.citation,
        'ref': result.ref,
        'text': result.text,
        'path': result.chunk.path,
        'timestamp': result.chunk.timestamp,
        'author': result.chunk.author
    }


def ask(
    context: typer.Context,
    question: str = typer.Argument(..., help='Question to answer.'),
    passages: int = typer.Option(
        DEFAULT_PASSAGES, '--passages', min=1,
        help='Maximum retrieved passages offered to the model.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project, effective, config = _runtime(
        context, project_slug, json_output
    )
    embedder = _embedder(effective, config, json_output)
    generator = _LazyGenerator(effective, config)
    try:
        with _store(project, config) as engine:
            answer = answer_question(
                project, question, embedder, generator, passages=passages,
                store=engine
            )
    except Exception as error:  # noqa: BLE001 — provider and store boundary
        fail(str(error), json_output)

    data = {
        'answer': answer.text,
        'passages': [
            {
                'text': result.text,
                'score': result.score,
                'citation': result.chunk.citation
            }
            for result in answer.passages
        ]
    }

    def render(target) -> None:
        target.print(answer.text, soft_wrap=True)
        target.print('\nSources', style='bold')
        if not answer.sources:
            target.print('None')
        for source in answer.sources:
            target.print(f'• {source}', soft_wrap=True)

    emit(data, json_output, render)


def search(
    context: typer.Context,
    query: str = typer.Argument(..., help='Text to search for.'),
    top_k: int = typer.Option(
        10, '--top-k', min=1, help='Maximum number of hits.'
    ),
    full: bool = typer.Option(
        False, '--full', help='Print complete passages without trimming.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project, effective, config = _runtime(
        context, project_slug, json_output
    )
    embedder = _embedder(effective, config, json_output)
    try:
        with _store(project, config) as engine:
            results = search_project(
                project, query, embedder, top_k=top_k, store=engine
            )
    except Exception as error:  # noqa: BLE001 — provider and store boundary
        fail(str(error), json_output)

    rows = [_result(result, rank) for rank, result in enumerate(results, 1)]

    def render(target) -> None:
        if not rows:
            target.print('Nothing found.')
            return
        for row in rows:
            target.print(
                f'{row["rank"]}. {row["score"]:.3f}  {row["citation"]}',
                style='bold'
            )
            text = str(row['text'])
            target.print(text if full else text[:PREVIEW_CHARS], soft_wrap=True)
            if not full and len(text) > PREVIEW_CHARS:
                target.print(f'… [{len(text) - PREVIEW_CHARS} more chars]')

    emit({'results': rows}, json_output, render)


def register_retrieval_commands(app: typer.Typer) -> None:
    app.command('ask')(ask)
    app.command('search')(search)
