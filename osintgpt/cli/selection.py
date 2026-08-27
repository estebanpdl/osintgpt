'''Selected-project persistence for the interactive CLI convenience.'''

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from osintgpt.projects import Project, Registry

SELECTION_FILE = 'selected-project'


class ProjectSelectionError(ValueError):
    '''Raised when a command cannot resolve an explicit or selected project.'''


@dataclass(frozen=True)
class CliState:
    '''Values shared by commands in one CLI invocation.'''

    home: Path


def state_from(context: typer.Context) -> CliState:
    return context.ensure_object(CliState)


def selection_file(home: Path) -> Path:
    return home / SELECTION_FILE


def read_selection(home: Path) -> Optional[str]:
    path = selection_file(home)
    if not path.is_file():
        return None

    selected = path.read_text(encoding='utf-8').strip()

    return selected or None


def write_selection(home: Path, slug: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    selection_file(home).write_text(f'{slug}\n', encoding='utf-8')


def clear_selection(home: Path, slug: Optional[str] = None) -> None:
    path = selection_file(home)
    if not path.is_file():
        return
    if slug is not None and read_selection(home) != slug:
        return

    path.unlink()


def resolve_project(home: Path, explicit: Optional[str] = None) -> Project:
    key = explicit or read_selection(home)
    if not key:
        raise ProjectSelectionError(
            'no project selected; run `osintgpt project use <slug>` or pass '
            '`--project <slug>`'
        )

    try:
        return Registry.load(home).open(key)
    except KeyError:
        raise ProjectSelectionError(
            f'project {key!r} was not found; run `osintgpt project list`'
        ) from None
