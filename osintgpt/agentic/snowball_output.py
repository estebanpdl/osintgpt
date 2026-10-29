'''Shared Snowball rows, date grouping and spreadsheet exports.'''

import csv
import io
import json
from datetime import datetime
from pathlib import Path

PERIODS = ('none', 'day', 'week', 'month', 'year')
FIELDS = (
    'hop', 'query', 'text', 'ref', 'document_ref', 'sequence', 'citation',
    'score', 'similarity_to_query', 'date', 'parsed_date', 'date_status', 'period',
    'identical_text_refs', 'identical_text_refs_scope',
)
EXPORT_FIELDS = (*FIELDS, 'spreadsheet_escaped_fields')
METHOD_NOTE = (
    'A chain is a similarity path, not evidence of causation or propagation. '
    'Drift is similarity to the initial query: higher values mean less drift.'
    ' Scores depend on the embedding model; thresholds do not carry across projects.'
)
COPY_SCOPE = 'Retrieved candidates only; not an exhaustive copy inventory.'


def parse_date(value, date_format=None):
    '''Parse an explicit format or ISO date/time, preserving the stated timezone.'''
    if not value:
        return None
    try:
        if date_format:
            return datetime.strptime(value.strip(), date_format)
        return datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None


def validate_grouping(group_by):
    if group_by not in PERIODS:
        raise ValueError('Group by none, day, week, month or year.')


def _period(parsed, group_by):
    if group_by == 'none':
        return ''
    if parsed is None:
        return 'Unknown date'
    if group_by == 'week':
        year, week, _ = parsed.isocalendar()
        return f'{year:04d}-W{week:02d}'
    return parsed.strftime({'day': '%Y-%m-%d', 'month': '%Y-%m', 'year': '%Y'}[group_by])


def snowball_rows(walk, group_by='none'):
    '''Stable columns in traversal order; grouping never changes hop identity.'''
    validate_grouping(group_by)
    rows = []
    for hop in walk.hops:
        chunk = hop.result.chunk
        parsed = parse_date(hop.date, walk.date_format)
        rows.append({
            'hop': hop.depth, 'query': hop.query, 'text': hop.text,
            'ref': hop.ref, 'document_ref': chunk.document_ref,
            'sequence': chunk.sequence, 'citation': chunk.citation,
            'score': hop.result.score, 'similarity_to_query': hop.drift, 'date': hop.date,
            'parsed_date': parsed.isoformat() if parsed is not None else '',
            'date_status': 'parsed' if parsed is not None else 'invalid' if hop.date else 'missing',
            'period': _period(parsed, group_by),
            'identical_text_refs': list(hop.identical_text_refs),
            'identical_text_refs_scope': COPY_SCOPE,
        })
    return rows


def grouped_rows(rows, group_by='none'):
    '''Sort period headings chronologically and retain hop order within them.'''
    validate_grouping(group_by)
    if group_by == 'none':
        return [('Chain', rows)]
    groups = {}
    for row in rows:
        groups.setdefault(row['period'], []).append(row)
    return sorted(groups.items(), key=lambda pair: (pair[0] == 'Unknown date', pair[0]))


def snowball_data(walk, group_by='none'):
    recorder = walk.usage
    missing = walk.unrecorded_calls > 0 or (walk.embedding_calls > 0 and not recorder.calls)
    complete = recorder.cost_complete and not missing
    return {
        'query': walk.query, 'hop_count': len(walk), 'stopped': walk.stopped,
        'model': walk.model, 'depth': walk.depth, 'threshold': walk.threshold,
        'date_field': walk.date_field, 'date_format': walk.date_format,
        'group_by': group_by, 'method_note': METHOD_NOTE,
        'dedupe': walk.dedupe,
        'identical_text_refs_scope': COPY_SCOPE,
        'usage': {
            'embedding_calls': walk.embedding_calls, 'calls': recorder.calls,
            'input_tokens': recorder.input_tokens, 'output_tokens': recorder.output_tokens,
            'total_tokens': recorder.total_tokens, 'complete': complete,
            'estimated_cost_usd': recorder.estimated_billable_cost if complete else None,
            'known_cost_usd': recorder.estimated_billable_cost,
            'unpriced_calls': recorder.unpriced_calls,
            'uncounted_calls': recorder.uncounted_billable_calls,
            'unrecorded_embedding_calls': walk.unrecorded_calls,
        },
        'hops': snowball_rows(walk, group_by),
    }


def usage_text(walk):
    usage = snowball_data(walk)['usage']
    cost = (f'estimated cost ${usage["estimated_cost_usd"]:.6f}' if usage['complete']
            else f'cost unknown/incomplete (${usage["known_cost_usd"]:.6f} known)')
    return (f'{walk.embedding_calls} embedding calls; {usage["calls"]} recorded responses; '
            f'{usage["total_tokens"]:,} reported tokens; {cost}.')


def _spreadsheet_rows(walk, group_by):
    # Match findings exports: reversible apostrophe escaping, with affected
    # columns named explicitly. Numeric negative scores remain numeric.
    for row in snowball_rows(walk, group_by):
        row['identical_text_refs'] = json.dumps(row['identical_text_refs'], ensure_ascii=False)
        escaped = []
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                row[key] = "'" + value
                escaped.append(key)
        row['spreadsheet_escaped_fields'] = json.dumps(escaped)
        yield row


def snowball_csv(walk, group_by='none'):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(_spreadsheet_rows(walk, group_by))
    return '\ufeff' + output.getvalue()


def snowball_xlsx(walk, group_by='none'):
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    if len(walk) > 1_048_575:
        raise ValueError('Chain exceeds the XLSX row limit; export CSV instead.')
    book = Workbook()
    sheet = book.active
    sheet.title = 'Snowball'
    sheet.append(list(EXPORT_FIELDS))
    try:
        for row in _spreadsheet_rows(walk, group_by):
            values = [row[key] for key in EXPORT_FIELDS]
            for value in values:
                if isinstance(value, str) and (len(value) > 32767 or ILLEGAL_CHARACTERS_RE.search(value)):
                    raise ValueError('Text cannot be stored losslessly in XLSX; export CSV instead.')
            sheet.append(values)
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        output = io.BytesIO()
        book.save(output)
        return output.getvalue()
    finally:
        book.close()


def export_snowball(walk, path, group_by='none'):
    '''Write a new CSV/XLSX file; an existing export is never overwritten.'''
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ('.csv', '.xlsx'):
        raise ValueError('Snowball export must end in .csv or .xlsx.')
    data = (snowball_csv(walk, group_by).encode('utf-8') if suffix == '.csv'
            else snowball_xlsx(walk, group_by))
    with path.open('xb') as handle:
        handle.write(data)
    return path
