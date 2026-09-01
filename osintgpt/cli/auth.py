'''Store provider credentials for this machine, without ever printing one.'''

import sys

import typer
from rich.table import Table

from osintgpt.credentials import (
    credential_names,
    credential_status,
    credentials_file,
    field_for,
    load_credentials,
    remove_credential,
    store_credential
)

from .output import emit, emit_record, fail
from .selection import state_from

auth_app = typer.Typer(help='Store and inspect provider credentials.')


def _field_or_fail(provider: str, json_output: bool) -> str:
    field = field_for(provider)
    if field is None:
        fail(
            f'unknown provider {provider!r}', json_output,
            {'valid_providers': sorted(credential_names())}
        )

    return field


@auth_app.command(
    'set', help='Store a provider credential, prompting for it.'
)
def set_credential(
    context: typer.Context,
    provider: str = typer.Argument(
        ..., help='Provider name, e.g. openai, gemini, anthropic.'
    ),
    stdin: bool = typer.Option(
        False, '--stdin', help='Read the credential from standard input.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    # The credential is never taken as an argument. A key on the command line
    # is a key in the shell history and in the process table, which is how a
    # secret outlives the moment it was needed.
    state = state_from(context)
    field = _field_or_fail(provider, json_output)

    if stdin:
        value = sys.stdin.read().strip()
    else:
        value = typer.prompt(
            f'{provider} credential', hide_input=True, err=True
        )

    try:
        path = store_credential(state.home, field, value)
    except (OSError, ValueError) as error:
        fail(str(error), json_output)

    emit_record(
        {'provider': provider, 'field': field, 'stored_in': str(path)},
        json_output,
        title='Credential stored'
    )


@auth_app.command(
    'list', help='Show which credentials are set and where they come from.'
)
def list_credentials(
    context: typer.Context,
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    # Presence and source, never a value.
    state = state_from(context)
    rows = [
        {
            'provider': row.provider,
            'variable': row.variable,
            'status': 'set' if row.is_set else 'not set',
            'source': row.source or '—',
            'shadowed': row.shadowed
        }
        for row in credential_status(state.home)
    ]

    def render(target) -> None:
        table = Table(title='Credentials')
        for label in ('Provider', 'Status', 'Source', 'Variable'):
            table.add_column(label)
        for row in rows:
            table.add_row(
                row['provider'], row['status'], row['source'], row['variable']
            )
        target.print(table)

        for row in rows:
            if row['shadowed']:
                # Both sources hold this one and the environment wins, so an
                # operator who just ran `auth set` is not using what they set.
                target.print(
                    f'{row["provider"]}: {row["variable"]} in the environment '
                    'is being used instead of the stored credential.',
                    style='yellow', soft_wrap=True
                )
        target.print(
            f'Stored in {credentials_file(state.home)}', soft_wrap=True
        )

    emit({'credentials': rows, 'file': str(credentials_file(state.home))},
         json_output, render)


@auth_app.command('remove', help='Forget a stored credential.')
def remove(
    context: typer.Context,
    provider: str = typer.Argument(..., help='Provider to forget.'),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    field = _field_or_fail(provider, json_output)

    try:
        removed = remove_credential(state.home, field)
    except OSError as error:
        fail(str(error), json_output)

    if not removed:
        fail(f'no stored credential for {provider}', json_output)

    emit_record(
        {'provider': provider, 'field': field, 'removed': True},
        json_output,
        title='Credential removed'
    )


@auth_app.command('path', help='Print where credentials are stored.')
def path(
    context: typer.Context,
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    # Where credentials are kept, so the file can be backed up or deleted
    # deliberately rather than found by accident.
    state = state_from(context)
    location = credentials_file(state.home)
    emit_record(
        {
            'path': str(location),
            'exists': location.is_file(),
            'stored': len(load_credentials(state.home))
        },
        json_output,
        title='Credentials file'
    )
