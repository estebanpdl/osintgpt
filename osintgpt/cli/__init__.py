'''Command-line access to osintgpt projects, corpora, and retrieval.'''

from pathlib import Path
from typing import Optional

import typer

from osintgpt.projects import default_home

from .config import config_app
from .corpus import register_corpus_commands
from .doctor import doctor as doctor_command
from .graph import graph_app
from .projects import project_app
from .retrieval import register_retrieval_commands
from .selection import CliState

app = typer.Typer(
    help='Create, configure, inspect, and query osintgpt projects.',
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
app.add_typer(config_app, name='config')
app.add_typer(graph_app, name='graph')
register_corpus_commands(app)
register_retrieval_commands(app)
app.command(
    'doctor', help='Report how this project is configured and what is wrong.'
)(doctor_command)

__all__ = ['app']
