'''Direct access to the shared semantic-chain operation.'''

from pathlib import Path
from typing import List, Optional

import typer

from osintgpt.agentic.snowball import DEFAULT_DEPTH, DEFAULT_THRESHOLD, DEFAULT_DEDUPE, snowball
from osintgpt.agentic.snowball_output import (
    METHOD_NOTE, export_snowball, grouped_rows, snowball_data,
    usage_text, validate_grouping,
)
from .output import emit, fail
from .selection import resolve_project, state_from


def register_snowball_command(app):
    @app.command('snowball', help='Follow a semantic similarity path. ' + METHOD_NOTE)
    def command(
        context: typer.Context,
        query: str = typer.Argument(..., help='Initial query or narrative.'),
        project: Optional[str] = typer.Option(None, '--project', '-p'),
        depth: int = typer.Option(DEFAULT_DEPTH, min=1, help='Maximum accepted hits, including the first.'),
        threshold: float = typer.Option(DEFAULT_THRESHOLD, min=-1.0, max=1.0, help='Minimum local cosine score (inclusive).'),
        dedupe: str = typer.Option(DEFAULT_DEDUPE, help='Skip normalized text copies (text), or repeated chunk identities (chunk).'),
        date_field: Optional[str] = typer.Option(None, help='Indexed metadata key; defaults to timestamp.'),
        date_format: Optional[str] = typer.Option(None, help='Explicit strptime format; otherwise ISO dates.'),
        group_by: str = typer.Option('none', help='Display grouping: none, day, week, month, year. Exports retain hop order.'),
        ref: Optional[List[str]] = typer.Option(None, '--ref', help='Restrict to a file ref; repeat for multiple files.'),
        drift: bool = typer.Option(True, '--drift/--no-drift', help='Compare hits with the initial query; adds DOCUMENT embeddings.'),
        export: Optional[Path] = typer.Option(None, '--export', help='New .csv or .xlsx file; refuses overwrite.'),
        json_output: bool = typer.Option(False, '--json', help='Output full results and usage as JSON.'),
    ):
        state = state_from(context)
        try:
            validate_grouping(group_by)
            if export is not None:
                if export.suffix.lower() not in ('.csv', '.xlsx'):
                    raise ValueError('Snowball export must end in .csv or .xlsx.')
                if export.exists():
                    raise ValueError(f'Export already exists: {export}')
                if not export.parent.is_dir():
                    raise ValueError(f'Export directory does not exist: {export.parent}')
            selected = resolve_project(state.home, project)
            walk = snowball(
                selected, query, home=state.home, depth=depth, threshold=threshold,
                score_against_initial=drift, refs=ref,
                date_field=date_field, date_format=date_format, dedupe=dedupe,
            )
            data = snowball_data(walk, group_by)
            if export is not None:
                data['export'] = str(export_snowball(walk, export, group_by))
        except Exception as error:
            fail(str(error), json_output)

        def render(target):
            target.print(f'{len(walk)} hops; {walk.stopped}', markup=False)
            target.print(usage_text(walk), markup=False)
            target.print(METHOD_NOTE, markup=False)
            for period, rows in grouped_rows(data['hops'], group_by):
                target.print(period, style='bold', markup=False)
                for row in rows:
                    initial = f'{row["similarity_to_query"]:.4f}' if row['similarity_to_query'] is not None else 'off'
                    target.print(
                        f'Hop {row["hop"]}: score {row["score"]:.4f}; '
                        f'similarity to query {initial}; {row["citation"]}; '
                        f'date {row["date"] or "unknown"}', markup=False,
                    )
                    target.print(f'Seed: {row["query"]}', markup=False)
                    target.print(row['text'], markup=False)
                    target.print(f'Identical-text sources encountered: {row["identical_text_refs"]}', markup=False)
            if export is not None:
                target.print(f'Exported: {export}', markup=False)

        emit(data, json_output, render)
