'''Command-line access to osintgpt projects, corpora, and retrieval.'''

from pathlib import Path
from typing import Optional

import typer

from osintgpt.projects import default_home

from .app_command import register_app_command
from .auth import auth_app
from .config import config_app
from .conversations import conversation_app
from .convert import register_convert_command
from .corpus import register_corpus_commands
from .doctor import doctor as doctor_command
from .evaluate import register_evaluate_command
from .graph import graph_app
from .projects import project_app
from .retrieval import register_retrieval_commands
from .selection import CliState
from .snowball import register_snowball_command

app = typer.Typer(
    help='Create, configure, inspect, and query osintgpt projects.',
    no_args_is_help=True
)


def _show_version(context: typer.Context, value: bool) -> None:
    if value and not context.resilient_parsing:
        from osintgpt import __version__
        typer.echo(f'osintgpt {__version__}')
        raise typer.Exit()


@app.callback()
def configure(
    context: typer.Context,
    home: Optional[Path] = typer.Option(
        None, '--home', help='osintgpt home directory.'
    ),
    version: bool = typer.Option(
        False, '--version', '-V', callback=_show_version, is_eager=True,
        help='Show the installed version and exit.'
    )
) -> None:
    context.obj = CliState(home=home or default_home())


app.add_typer(project_app, name='project')
app.add_typer(auth_app, name='auth')
app.add_typer(config_app, name='config')
app.add_typer(graph_app, name='graph')
app.add_typer(conversation_app, name='conversations')
register_corpus_commands(app)
register_convert_command(app)
register_retrieval_commands(app)
register_evaluate_command(app)
register_snowball_command(app)
app.command(
    'doctor', help='Report how this project is configured and what is wrong.'
)(doctor_command)

register_app_command(app)

__all__ = ['app']
