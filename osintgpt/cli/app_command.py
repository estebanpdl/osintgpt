# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: app_command.py
# Description: `osintgpt app` — the browser interface, from the same command
#   as everything else.
# =================================================================================

# type hints
from typing import List, Optional

import typer


def launch_app(
    port: Optional[int] = typer.Option(
        None, '--port', help='Port to serve on.'
    ),
    headless: bool = typer.Option(
        False, '--headless', help='Do not open a browser window.'
    )
) -> None:
    from osintgpt.app.launch import main as run_app

    arguments: List[str] = []
    if port:
        arguments.append(f'--server.port={port}')
    if headless:
        arguments.append('--server.headless=true')

    raise typer.Exit(run_app(arguments))


def register_app_command(app: typer.Typer) -> None:
    app.command(
        # The brackets are escaped: Rich reads [app] as markup and would
        # print "Needs osintgpt." — which says nothing at all.
        'app',
        help=r'Open the browser interface. Needs osintgpt\[app].'
    )(launch_app)
