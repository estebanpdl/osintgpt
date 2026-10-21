'''Deterministic answer and authored-finding exports retain original text and saved evidence.'''

import csv
import hashlib
import io
import json
import re
from copy import deepcopy

from .graph.export_snapshot import project_identity


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def canon_pages(project):
    root = project.paths.canon.resolve()
    return sorted(path.relative_to(root).as_posix() for path in root.rglob('*.md')
                  if path.is_file() and path.resolve().is_relative_to(root))


def _canon_page(project, name):
    root = project.paths.canon.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != '.md' or not path.is_file():
        raise ValueError('Select an existing Markdown page inside this project’s canon directory.')
    raw = path.read_bytes()
    ref = 'canon/' + path.relative_to(root).as_posix()
    return {'kind': 'finding', 'id': ref, 'title': ref, 'text': raw.decode('utf8'),
            'source_version': 'sha256:' + hashlib.sha256(raw).hexdigest(), 'citations': [ref],
            'evidence': [], 'evidence_status': 'authored_markdown',
            'note': 'Original Markdown citations and links are preserved; their targets have not been verified.'}


def saved_evidence(answer):
    retained, graph_states = [], []
    trace = answer.get('trace')
    calls = (trace.get('calls') or []) if isinstance(trace, dict) else []
    for call in calls:
        if not isinstance(call, dict):
            continue
        payload = call.get('payload')
        if not isinstance(payload, dict):
            continue
        tool = call.get('tool', '')
        if tool in ('graph_query', 'graph_evidence'):
            retained.append({'tool': tool, 'arguments': deepcopy(call.get('arguments', {})), 'result': deepcopy(payload)})
            if payload.get('graph'):
                graph_states.append(deepcopy(payload['graph']))
        elif payload.get('passages'):
            retained.append({'tool': tool, 'arguments': deepcopy(call.get('arguments', {})),
                             'result': {'passages': deepcopy(payload['passages'])}})
    return retained, graph_states


def findings_package(project, *, conversation=None, turns=(), pages=()):
    selected = sorted(set(turns))
    if selected and (conversation is None or any(type(i) is not int or i < 0 or i >= len(conversation.turns) for i in selected)):
        raise ValueError('Select existing answer numbers in the current conversation.')
    if conversation is not None and not conversation.path.resolve().is_relative_to(project.paths.root.resolve()):
        raise ValueError('The conversation belongs to a different project.')
    items = []
    for index in selected:
        turn = conversation.turns[index]
        evidence, graph_states = saved_evidence(turn.answer)
        items.append({'kind': 'answer', 'id': f'{conversation.id}:{index + 1}',
                      'title': turn.question, 'text': turn.answer.get('answer', ''),
                      'asked_at': turn.asked_at, 'conversation_id': conversation.id,
                      'conversation_title': conversation.name, 'turn': index + 1,
                      'citations': list(turn.answer.get('sources') or []),
                      'evidence': evidence, 'graph_states': graph_states,
                      'evidence_status': 'saved_at_answer_time' if evidence else 'not_recorded',
                      'degraded': turn.answer.get('degraded', ''),
                      'note': 'Saved verification describes answer time; no current source checks were performed.'})
    items.extend(_canon_page(project, name) for name in sorted(set(pages)))
    if not items:
        raise ValueError('Select at least one answer or finding to export.')
    return {'metadata': {'format_version': 1, 'project': project_identity(project),
                         'scope': 'selected_answers_and_findings',
                         'selection': {'conversation_id': conversation.id if selected else None,
                                       'turns': [i + 1 for i in selected], 'canon_pages': sorted(set(pages))},
                         'generation_calls': 0}, 'items': items}


def _fence(text, language=''):
    length = max((len(part) for part in re.findall(r'`+', text)), default=0) + 1
    marker = '`' * max(3, length)
    return f'{marker}{language}\n{text}\n{marker}\n'


def findings_markdown(package):
    lines = ['# Selected answers and findings\n', _fence(_json(package['metadata']), 'json')]
    for number, item in enumerate(package['items'], 1):
        lines += [f'## {number}. {item["kind"].title()}\n', _fence(item['title']),
                  item['text'], '\n### Citations\n', _fence('\n'.join(item['citations']) or 'No citations recorded.'),
                  '### Evidence and context\n', _fence(_json({key: value for key, value in item.items()
                    if key not in ('text', 'title', 'citations', 'evidence')}), 'json')]
        if not item['evidence']:
            lines.append('No tool evidence was recorded for this item.\n')
        for record in item['evidence']:
            result = record['result']
            for claim in [*result.get('path', []), *result.get('claims', [])]:
                lines += [_fence(f"{claim['source']} — {claim['relation']} → {claim['target']}\n{claim['citation']}"),
                          *[_fence(quote) for quote in claim.get('evidence_quotes', [])]]
            for passage in result.get('passages', []):
                lines.append(_fence(f"{passage.get('citation', passage.get('ref', ''))}\n{passage.get('text', '')}"))
            if record['tool'] == 'graph_evidence' and result.get('text'):
                lines.append(_fence(result['text']))
            lines.append(_fence(_json(record), 'json'))
    return '\n'.join(lines) + '\n'


def findings_csv(package):
    output = io.StringIO(newline='')
    fields = ['kind', 'id', 'title', 'text', 'citations', 'evidence', 'evidence_status',
              'asked_at', 'source_version', 'graph_states', 'context', 'export_context', 'spreadsheet_escaped_fields']
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for item in package['items']:
        row = {key: item.get(key, '') for key in fields}
        for key in ('citations', 'evidence', 'graph_states'):
            row[key] = _json(item.get(key, []))
        row['context'] = _json({key: value for key, value in item.items() if key not in fields})
        row['export_context'] = _json(package['metadata'])
        # A leading apostrophe prevents spreadsheet formulas while retaining the exact value after it.
        escaped = []
        for key, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                row[key] = "'" + value
                escaped.append(key)
        row['spreadsheet_escaped_fields'] = _json(escaped)
        writer.writerow(row)
    return '\ufeff' + output.getvalue()
