'''Command-line access to osintgpt projects and corpora.'''

from pathlib import Path
from typing import Optional

import typer

from osintgpt.projects import default_home

from .corpus import register_corpus_commands
from .projects import project_app
from .selection import CliState

app = typer.Typer(
    help='Create osintgpt projects and manage their corpora.',
    no_args_is_help=True
)


@app.callback()
def configure(
    context: typer.Context,
    home: Optional[Path] = typer.Option(
        None, '--home', help='osintgpt home directory.'
    )
) -> None:
    context.obj = CliState(home=home or default_home())


app.add_typer(project_app, name='project')
register_corpus_commands(app)

__all__ = ['app']
