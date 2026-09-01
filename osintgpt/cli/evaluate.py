'''Measure project retrieval against questions with known answers.'''

from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.table import Table

from osintgpt.credentials import resolve_credentials
from osintgpt.evaluation import (
    DEFAULT_TOP_K,
    HYBRID,
    SEMANTIC,
    EvaluationReport,
    evaluate,
    load_questions
)
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.llm import build_embedding_provider
from osintgpt.llm.usage import CostLimitReached
from osintgpt.projects import load_user_defaults
from osintgpt.vector_store import store_for

from .costs import (
    add_usage,
    fail_for_cost,
    recorder_for,
    render_usage,
    usage_data
)
from .output import emit, fail
from .selection import ProjectSelectionError, resolve_project, state_from


class RetrievalMethod(str, Enum):
    semantic = SEMANTIC
    hybrid = HYBRID


def _runtime(
    context: typer.Context, explicit: Optional[str], json_output: bool
):
    state = state_from(context)
    try:
        project = resolve_project(state.home, explicit)
        defaults = load_user_defaults(state.home)
        effective = project.effective_settings(defaults)
        config = project.settings_for(
            resolve_credentials(state.home), defaults
        )
    except (ProjectSelectionError, OSError, ValueError) as error:
        fail(str(error), json_output)

    return project, effective, config


def _embedder(effective, config, recorder, json_output: bool):
    try:
        return build_embedding_provider(
            effective.embedding_provider,
            config,
            model=effective.embedding_model or None,
            recorder=recorder
        )
    except (ImportError, MissingEnvironmentVariableError, ValueError) as error:
        fail(str(error), json_output, {'usage': usage_data(recorder)})


@contextmanager
def _store(project, config):
    engine = store_for(project, config)
    try:
        yield engine
    finally:
        close = getattr(engine, 'close', None)
        if close is not None:
            close()


def _payload(
    report: EvaluationReport, project_slug: str, questions_path: Path
) -> Dict[str, object]:
    return {
        'project': project_slug,
        'questions': str(questions_path),
        'retrieval': report.retrieval,
        'embedding_model': report.embedding_model,
        'top_k': report.top_k,
        'scored': report.scored,
        'found': report.found,
        'hit_rate': report.hit_rate,
        'mean_reciprocal_rank': report.mean_reciprocal_rank,
        'mean_recall': report.mean_recall,
        'misses': [
            {
                'question': result.question.text,
                'expected': result.question.expected,
                'retrieved': result.retrieved
            }
            for result in report.misses
        ],
        'unscorable': report.unscorable
    }


def _render(report: EvaluationReport, target, recorder) -> None:
    summary = Table(show_header=False)
    summary.add_column('Field', style='bold')
    summary.add_column('Value')
    summary.add_row('Retrieval', report.retrieval)
    summary.add_row('Embedding model', report.embedding_model)
    summary.add_row('Top k', str(report.top_k))
    summary.add_row('Scored', str(report.scored))
    summary.add_row('Found', str(report.found))
    summary.add_row('Hit rate', f'{report.hit_rate:.0%}')
    summary.add_row('MRR', f'{report.mean_reciprocal_rank:.3f}')
    summary.add_row('Recall', f'{report.mean_recall:.0%}')
    target.print(summary)

    target.print('')
    target.print('Misses', style='bold')
    if not report.misses:
        target.print('None')
    for result in report.misses:
        target.print(result.question.text, soft_wrap=True)
        target.print(f'  Expected: {", ".join(result.question.expected)}')
        retrieved = ', '.join(result.retrieved) or 'nothing'
        target.print(f'  Retrieved: {retrieved}')

    target.print('')
    target.print('Unscorable', style='bold')
    if not report.unscorable:
        target.print('None')
    for reason in report.unscorable:
        target.print(f'• {reason}', soft_wrap=True)
    render_usage(target, recorder)


def evaluate_command(
    context: typer.Context,
    questions: Path = typer.Argument(
        ..., metavar='QUESTIONS', help='Question set TOML file.'
    ),
    top_k: int = typer.Option(
        DEFAULT_TOP_K, '--top-k', min=1,
        help='Maximum documents considered for each question.'
    ),
    retrieval: RetrievalMethod = typer.Option(
        RetrievalMethod.semantic, '--retrieval',
        help='Retrieval method to measure.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project, effective, config = _runtime(
        context, project_slug, json_output
    )
    recorder = recorder_for(effective)
    questions_path = questions.expanduser()
    if not questions_path.is_file():
        fail(f'question set does not exist: {questions_path}', json_output)

    try:
        question_set = load_questions(questions_path)
    except (OSError, ValueError) as error:
        fail(str(error), json_output)

    embedder = _embedder(effective, config, recorder, json_output)
    try:
        with _store(project, config) as engine:
            report = evaluate(
                project,
                question_set,
                embedder,
                top_k=top_k,
                known_refs=engine.refs(embedder.model),
                retrieval=retrieval.value,
                store=engine
            )
    except CostLimitReached as error:
        fail_for_cost(error, recorder, json_output)
    except Exception as error:  # noqa: BLE001 — provider and store boundary
        fail(str(error), json_output, {'usage': usage_data(recorder)})

    data = _payload(report, project.slug, questions_path)
    add_usage(data, recorder)
    emit(
        data, json_output,
        lambda target: _render(report, target, recorder)
    )


def register_evaluate_command(app: typer.Typer) -> None:
    app.command(
        'evaluate', help='Measure retrieval against known-answer questions.'
    )(evaluate_command)
