'''Read and update project configuration without exposing credentials.'''

from dataclasses import fields, replace
from types import UnionType
from typing import Optional, Union, get_args, get_origin

import typer

from osintgpt.config import ENV_VARS, secret_fields
from osintgpt.credentials import resolve_credentials
from osintgpt.projects import (
    ProjectSettings,
    Registry,
    load_user_defaults,
    save_user_defaults
)

from .output import emit, emit_record, fail
from .selection import ProjectSelectionError, resolve_project, state_from

config_app = typer.Typer(help='Read and update project settings.')
PROJECT_FIELDS = {field.name: field for field in fields(ProjectSettings)}
SECRET_FIELDS = secret_fields()


def _context(context: typer.Context, explicit: Optional[str], json_output: bool):
    state = state_from(context)
    try:
        project = resolve_project(state.home, explicit)
        defaults = load_user_defaults(state.home)
        environment = resolve_credentials(state.home)
    except (ProjectSelectionError, OSError, ValueError) as error:
        fail(str(error), json_output)

    return state, project, defaults, environment


def _valid_keys() -> list[str]:
    return sorted(PROJECT_FIELDS) + sorted(SECRET_FIELDS)


def _field_or_fail(key: str, json_output: bool):
    if key not in PROJECT_FIELDS and key not in SECRET_FIELDS:
        fail(
            f'unknown setting {key!r}', json_output,
            {'valid_keys': _valid_keys()}
        )

    return PROJECT_FIELDS.get(key)


def _parse_value(key: str, raw: str):
    annotation = PROJECT_FIELDS[key].type
    origin = get_origin(annotation)
    args = get_args(annotation)
    nullable = origin in (Union, UnionType) and type(None) in args
    value_type = next((item for item in args if item is not type(None)), annotation)

    if nullable and raw.strip().lower() in {'none', 'null', 'unset'}:
        return None
    if value_type is bool:
        values = {
            'true': True, 'yes': True, 'on': True, '1': True,
            'false': False, 'no': False, 'off': False, '0': False
        }
        try:
            return values[raw.strip().lower()]
        except KeyError:
            raise ValueError(
                f'{key} must be true or false, got {raw!r}'
            ) from None
    if value_type is float:
        try:
            return float(raw)
        except ValueError:
            raise ValueError(f'{key} must be a number, got {raw!r}') from None

    return raw


@config_app.command('get')
def get_config(
    context: typer.Context,
    key: Optional[str] = typer.Argument(None, help='Setting to read.'),
    user: bool = typer.Option(False, '--user', help='Read user defaults.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    _, project, defaults, environment = _context(
        context, project_slug, json_output
    )
    selected = defaults if user else project.settings
    scope = 'user' if user else 'project'

    if key is not None:
        _field_or_fail(key, json_output)
        value = (
            {'set': bool(getattr(environment, key))}
            if key in SECRET_FIELDS else getattr(selected, key)
        )
        emit_record(
            {'scope': scope, 'key': key, 'value': value}, json_output,
            title=f'{scope.title()} setting'
        )
        return

    data = {
        'scope': scope,
        'settings': {
            name: getattr(selected, name) for name in sorted(PROJECT_FIELDS)
        },
        'secrets': {
            name: {'set': bool(getattr(environment, name))}
            for name in sorted(SECRET_FIELDS)
        }
    }

    def render(target) -> None:
        target.print(f'{scope.title()} settings', style='bold')
        for name, value in data['settings'].items():
            target.print(f'{name}: {value}')
        target.print('Environment secrets', style='bold')
        for name, status in data['secrets'].items():
            target.print(f'{name}: {"set" if status["set"] else "not set"}')

    emit(data, json_output, render)


@config_app.command('set')
def set_config(
    context: typer.Context,
    key: str = typer.Argument(..., help='Setting to update.'),
    value: str = typer.Argument(..., help='New value.'),
    user: bool = typer.Option(False, '--user', help='Write user defaults.'),
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state, project, defaults, _ = _context(context, project_slug, json_output)
    _field_or_fail(key, json_output)
    if key in SECRET_FIELDS:
        # Naming the command rather than only the prohibition: an operator who
        # reaches for this is trying to store a key, and telling them where
        # that actually works is the whole of the answer.
        provider = key.rsplit('_api_key', 1)[0].rsplit('_dsn', 1)[0]
        fail(
            f'{key} is a secret and is never written to a project file. '
            f'Store it with "osintgpt auth set {provider}", or set '
            f'{ENV_VARS[key]} in the environment.',
            json_output
        )
    try:
        parsed = _parse_value(key, value)
        if user:
            save_user_defaults(
                state.home, replace(defaults, **{key: parsed})
            )
        else:
            updated = project.with_settings(**{key: parsed})
            updated.save()
            Registry.load(state.home).register(updated)
    except (OSError, TypeError, ValueError) as error:
        fail(str(error), json_output)

    emit_record(
        {
            'scope': 'user' if user else 'project',
            'key': key,
            'value': parsed
        },
        json_output,
        title='Setting updated'
    )
