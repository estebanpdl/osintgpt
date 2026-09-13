'''Inspect and manage the threads a project has been asked in.'''

from typing import Optional

import typer
from rich.table import Table

from osintgpt.projects import (
    delete_conversation,
    list_conversations,
    open_conversation
)

from .output import console, emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

conversation_app = typer.Typer(
    help='Inspect and manage conversation threads.', no_args_is_help=True
)


def _project(context: typer.Context, slug: Optional[str], json_output: bool):
    state = state_from(context)
    try:
        return resolve_project(state.home, slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)


def _open_or_fail(project, conversation_id: str, json_output: bool):
    conversation = open_conversation(project, conversation_id)
    if conversation is None:
        fail(f'no conversation {conversation_id!r} in {project.slug}',
             json_output)

    return conversation


@conversation_app.command('list', help='List this project\'s conversations.')
def list_threads(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project = _project(context, project_slug, json_output)
    rows = [
        {'id': item.id, 'title': item.name, 'created_at': item.created_at}
        for item in list_conversations(project)
    ]

    def render(target) -> None:
        if not rows:
            target.print('No conversations yet.')

            return

        table = Table(title=f'Conversations — {project.name}')
        for label in ('Title', 'Id'):
            table.add_column(label)
        for row in rows:
            table.add_row(row['title'], row['id'])
        target.print(table)

    emit({'project': project.slug, 'conversations': rows}, json_output, render)


@conversation_app.command('show', help='Print a conversation\'s turns.')
def show(
    context: typer.Context,
    conversation_id: str = typer.Argument(..., help='Conversation id.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project = _project(context, project_slug, json_output)
    conversation = _open_or_fail(project, conversation_id, json_output)

    turns = [
        {
            'question': turn.question,
            'answer': turn.answer.get('answer', ''),
            'sources': turn.answer.get('sources', []),
            'asked_at': turn.asked_at
        }
        for turn in conversation.turns
    ]

    def render(target) -> None:
        target.print(f'{conversation.name}', style='bold')
        for index, turn in enumerate(turns, 1):
            target.print(f'\n[{index}] {turn["question"]}', style='cyan')
            target.print(turn['answer'], soft_wrap=True)

    emit(
        {'id': conversation.id, 'title': conversation.name, 'turns': turns},
        json_output,
        render
    )


@conversation_app.command('delete', help='Delete a conversation.')
def delete(
    context: typer.Context,
    conversation_id: str = typer.Argument(..., help='Conversation id.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    yes: bool = typer.Option(
        False, '--yes', help='Delete without asking.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    project = _project(context, project_slug, json_output)
    conversation = _open_or_fail(project, conversation_id, json_output)

    # Deleting a transcript is not recoverable; --yes is how a script says
    # it meant to.
    if not yes and not json_output:
        console.print(
            f'{conversation.name} — {len(conversation)} turn(s)'
        )
        if not typer.confirm('Delete this conversation?'):
            raise typer.Abort()

    if not delete_conversation(conversation):
        fail(f'could not delete {conversation_id}', json_output)

    emit_record(
        {'id': conversation.id, 'deleted': True},
        json_output,
        title='Conversation deleted'
    )
