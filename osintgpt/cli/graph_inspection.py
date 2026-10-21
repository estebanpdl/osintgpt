'''Graph inspection commands share the model-facing retrieval and evidence services.'''

from typing import Optional, List
import sqlite3

import typer
from rich.table import Table
from rich.text import Text

from osintgpt.graph import Edge, assertion_context, discover_entities, query_graph
from osintgpt.graph.inspection import operator_query_result, query_messages
from osintgpt.graph.status_messages import query_outcome
from .output import emit, fail


def _render(target, data, collection):
    state = data.get('graph', {})
    for message in [state.get('build_message', ''), *state.get('coverage_notes', []), query_outcome(data)]:
        if message:
            target.print(Text(message))
    if data.get('candidates'):
        table = Table('Entity ID', 'Name', 'Type', 'Direct assertions (all / selected)',
                      'Connected identities (all / selected)', 'Sources')
        for item in data['candidates']:
            counts = item.get('relationships', {})
            table.add_row(item['id'], Text(item['name']), Text(item['type']),
                          f"{counts.get('all_sources', '?')} / {counts.get('selected_sources', '?')}",
                          f"{counts.get('distinct_neighbors_all_sources', '?')} / {counts.get('distinct_neighbors_selected_sources', '?')}",
                          Text(', '.join(source['ref'] for source in item['sources'])))
        target.print(table)
    rows = data.get(collection, [])
    if data.get('limits'):
        display = operator_query_result({**data, 'path' if collection == 'edges' else 'claims': rows})
        captions, warnings = query_messages(display)
    else:
        captions = []
        warnings = ['Candidate results were limited; narrow the name or use a persistent ID.'] if data.get('truncated') else []
    for message in captions:
        target.print(Text(message))
    if rows:
        target.print('Quotations verify textual support, not truth.')
        target.print('Relation polarity values: positive, negative, unknown. Assertion status is separate.')
        table = Table('Depth', 'Assertion', 'Relationship', 'Traversal', 'Document', 'Quotations')
        for row in rows:
            edge = Edge(**{key: row[key] for key in Edge.__dataclass_fields__ if key in row})
            step = row.get('traversal', {})
            table.add_row(str(step.get('depth', '')), row['assertion_id'], Text(edge.sentence),
                          'Against arrow' if step.get('against_direction') else 'With arrow',
                          Text(row['ref']), Text('\n'.join(row['evidence_quotes'])))
        target.print(table)
    for message in warnings:
        target.print(Text(message))


def _checked(operation, json_output):
    try:
        data = operation()
    except ValueError as error:
        fail(str(error), json_output)
    if not data.get('built', True):
        message = 'The graph has not been built yet.' if data['status'] == 'unbuilt' else data.get('note', '')
        fail(message, json_output, data)
    return data


def register(app, project_for):
    from osintgpt.graph.identity_review import reconcile_identities, review_identities

    @app.command('diagnostics')
    def diagnostics_command(
        context: typer.Context,
        offset: int = typer.Option(0, '--offset', min=0, help='Saved response page offset.'),
        limit: int = typer.Option(50, '--limit', min=1, help='Responses per page.'),
        raw: bool = typer.Option(False, '--raw', help='Include original response text and opaque continuation; local sensitive project data.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Inspect retained generation responses and failure diagnostics without model calls.'''
        from osintgpt.graph.diagnostics import generation_diagnostics
        _, project = project_for(context, project_slug, json_output)
        try:
            data = generation_diagnostics(project, offset=offset, limit=limit, raw=raw)
        except (ValueError, OSError, sqlite3.Error) as error:
            fail(str(error), json_output)
        emit(data, json_output, lambda target: target.print_json(data=data))

    @app.command('revalidate')
    def revalidate_command(
        context: typer.Context,
        refs: Optional[List[str]] = typer.Option(None, '--ref', help='Saved source or record; repeat to select more.'),
        limit: int = typer.Option(100, '--limit', min=1, max=100, help='Maximum saved rejected proposals to recheck.'),
        offset: int = typer.Option(0, '--offset', min=0, help='Page offset; continue with the returned next_offset.'),
        apply: bool = typer.Option(False, '--apply', help='Publish locally supported corrections after backup; default previews only.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Recheck saved proposals with current evidence rules, without model calls or new quotations.'''
        from osintgpt.graph.revalidation import revalidate_proposals
        _, project = project_for(context, project_slug, json_output)
        try:
            data = revalidate_proposals(project, refs=refs, limit=limit, offset=offset, apply=apply)
        except (ValueError, OSError, sqlite3.Error) as error:
            fail(str(error), json_output)
        emit(data, json_output, lambda target: target.print_json(data=data))

    @app.command('reconcile')
    def reconcile_command(
        context: typer.Context,
        query: str = typer.Option('', '--query', help='Find identity candidates or inspect an ID; empty lists same-name candidates.'),
        source_id: str = typer.Option('', '--from-id', help='Identity to combine with --to-id.'),
        target_id: str = typer.Option('', '--to-id', help='Canonical identity to retain.'),
        reason: str = typer.Option('', '--reason', help='Source-based explanation required when combining identities.'),
        undo: str = typer.Option('', '--undo', help='Undo a saved operator decision by ID.'),
        split_source: str = typer.Option('', '--split-source', help='Separate this exact source contribution from --from-id into a new identity.'),
        apply: bool = typer.Option(False, '--apply', help='Apply the previewed operation, creating a SQLite backup. Default: preview only.'),
        offset: int = typer.Option(0, '--offset', min=0, help='Candidate page offset.'),
        limit: int = typer.Option(30, '--limit', min=1, max=60, help='Candidates per page.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Inspect sourced identity decisions or explicitly reconcile existing nodes without model calls.'''
        _, project = project_for(context, project_slug, json_output)
        try:
            data = (reconcile_identities(project, source_id, target_id, reason=reason, undo=undo, apply=apply,
                                        split_source=split_source)
                    if source_id or target_id or undo or split_source else review_identities(project, query, limit=limit, offset=offset))
        except (ValueError, OSError, sqlite3.Error) as error:
            fail(str(error), json_output)
        emit(data, json_output, lambda target: target.print_json(data=data))

    @app.command('entities')
    def entities_command(
        context: typer.Context,
        query: str = typer.Argument(..., help='Entity ID, name, or alias in its original language.'),
        limit: int = typer.Option(20, '--limit', min=1, max=60, help='Maximum identity candidates.'),
        max_bytes: int = typer.Option(24000, '--max-bytes', min=4096, max=65536, help='Evidence response byte budget.'),
        refs: Optional[List[str]] = typer.Option(None, '--ref', help='Source scope for relationship counts; repeat for sources, omit for all.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Discover distinct identities before selecting a graph node.'''
        _, project = project_for(context, project_slug, json_output)
        data = _checked(lambda: discover_entities(project, query, limit=limit, max_bytes=max_bytes,
                                                  allow_disabled=True, refs=refs), json_output)
        emit({'project': project.slug, **data}, json_output, lambda target: _render(target, data, 'claims'))

    @app.command('neighbors')
    def neighbors_command(
        context: typer.Context,
        entity: str = typer.Argument(..., help='Entity name, alias, or persistent ID to inspect.'),
        depth: int = typer.Option(1, '--depth', min=1, max=4, help='Traversal depth.'),
        limit: int = typer.Option(30, '--limit', min=1, max=60, help='Maximum assertions.'),
        direction: str = typer.Option('connectivity', '--direction', help='incoming, outgoing, or connectivity.'),
        max_bytes: int = typer.Option(24000, '--max-bytes', min=4096, max=65536, help='Evidence response byte budget.'),
        refs: Optional[List[str]] = typer.Option(None, '--ref', help='Restrict traversal to these sources; repeat as needed, omit for all.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Show bounded, sourced assertions around one unambiguous identity.'''
        _, project = project_for(context, project_slug, json_output)
        data = _checked(lambda: query_graph(project, entity, direction=direction, depth=depth, limit=limit,
                                            max_bytes=max_bytes, allow_disabled=True, refs=refs), json_output)
        data['results'] = data.pop('claims', [])
        for row in data['results']:
            row['depth'] = row['traversal']['depth']
        data.update(project=project.slug, entity=entity, depth=depth)
        emit(data, json_output, lambda target: _render(target, data, 'results'))

    @app.command('path')
    def path_command(
        context: typer.Context,
        entity_a: str = typer.Argument(..., help='Starting entity name, alias, or ID.'),
        entity_b: str = typer.Argument(..., help='Destination entity name, alias, or ID.'),
        max_depth: int = typer.Option(4, '--max-depth', min=1, max=4, help='Longest path to consider.'),
        direction: str = typer.Option('connectivity', '--direction', help='incoming, outgoing, or connectivity.'),
        max_bytes: int = typer.Option(24000, '--max-bytes', min=4096, max=65536, help='Evidence response byte budget.'),
        refs: Optional[List[str]] = typer.Option(None, '--ref', help='Restrict traversal to these sources; repeat as needed, omit for all.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Find a bounded path while retaining every assertion's original direction.'''
        _, project = project_for(context, project_slug, json_output)
        data = _checked(lambda: query_graph(project, entity_a, entity_b, direction=direction, depth=max_depth,
                                            max_bytes=max_bytes, allow_disabled=True, refs=refs), json_output)
        data['edges'] = data.pop('path', [])
        for row in data['edges']:
            row['depth'] = row['traversal']['depth']
        data.update(project=project.slug, source=entity_a, target=entity_b, max_depth=max_depth,
                    length=len(data['edges']), documents=list(dict.fromkeys(row['ref'] for row in data['edges'])))
        emit(data, json_output, lambda target: _render(target, data, 'edges'))

    @app.command('evidence')
    def evidence_command(
        context: typer.Context,
        assertion_id: str = typer.Argument(..., help='Assertion ID returned by neighbors or path.'),
        quote_index: int = typer.Option(0, '--quote-index', min=0, max=199, help='Supporting quotation, zero-based.'),
        occurrence: int = typer.Option(0, '--occurrence', min=0, max=10000, help='Matching source occurrence, zero-based.'),
        context_chars: int = typer.Option(800, '--context-chars', min=0, max=4000, help='Surrounding characters on each side.'),
        offset: int = typer.Option(0, '--offset', min=0, max=1000000, help='Offset within the returned evidence context.'),
        limit_chars: int = typer.Option(4000, '--limit-chars', min=1, max=8000, help='Maximum characters per page.'),
        max_bytes: int = typer.Option(24000, '--max-bytes', min=4096, max=65536, help='Evidence response byte budget.'),
        project_slug: Optional[str] = typer.Option(None, '--project', help='Project slug or id; overrides selection.'),
        json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
    ):
        '''Read the saved source passage around a selected supporting quotation.'''
        _, project = project_for(context, project_slug, json_output)
        data = _checked(lambda: assertion_context(project, assertion_id, quote_index=quote_index,
            occurrence=occurrence, context_chars=context_chars, offset=offset, limit_chars=limit_chars,
            max_bytes=max_bytes, allow_disabled=True), json_output)
        def render(target):
            target.print(Text(data.get('text', data.get('note', ''))))
            if data.get('next_offset') is not None:
                target.print(f"Continue with --offset {data['next_offset']}.")
            target.print(Text(str(data.get('verification', {}))))
        emit({'project': project.slug, **data}, json_output, render)
