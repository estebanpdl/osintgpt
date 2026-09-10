'''One JSON-or-Rich output boundary shared by every command.'''

import json
from typing import Any, Callable, Dict, List, NoReturn, Sequence, Tuple

import typer
from rich.console import Console
from rich.table import Table

console = Console(highlight=False)
error_console = Console(stderr=True, highlight=False)


def emit(
    data: Any,
    json_output: bool,
    render: Callable[[Console], None]
) -> None:
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False, default=str))
        return

    render(console)


def emit_record(
    data: Dict[str, Any], json_output: bool, title: str = ''
) -> None:
    def render(target: Console) -> None:
        table = Table(title=title or None, show_header=False)
        table.add_column('Field', style='bold')
        table.add_column('Value')
        for key, value in data.items():
            table.add_row(key.replace('_', ' '), display(value))
        target.print(table)

    emit(data, json_output, render)


def emit_rows(
    rows: List[Dict[str, Any]],
    json_output: bool,
    columns: Sequence[Tuple[str, str]],
    title: str = ''
) -> None:
    def render(target: Console) -> None:
        table = Table(title=title or None)
        for _, label in columns:
            table.add_column(label)
        for row in rows:
            table.add_row(*(display(row.get(key, '')) for key, _ in columns))
        target.print(table)

    emit(rows, json_output, render)


def fail(
    message: str,
    json_output: bool = False,
    details: Dict[str, Any] | None = None
) -> NoReturn:
    payload = {'error': message}
    payload.update(details or {})
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, default=str))
    else:
        error_console.print(message, style='bold red', soft_wrap=True)
        for key, value in (details or {}).items():
            error_console.print(
                f'{key.replace("_", " ")}: {display(value)}', soft_wrap=True
            )

    raise typer.Exit(code=1)


def display(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)

    return str(value)
