'''Graph interchange formats preserve source claims and Unicode text.'''

import json
import re
from dataclasses import asdict

import pytest

from osintgpt.graph import (
    Edge,
    Entity,
    GraphStore,
    export_graph,
    merge_key,
    to_cypherl,
    to_json
)

STRING = r'"(?:\\.|[^"\\])*"'
NODE_STATEMENT = re.compile(
    rf'MERGE \(n:Entity \{{key: {STRING}\}}\) '
    rf'SET n\.name = {STRING}, n\.type = {STRING}, n\.mentions = \d+'
)
EDGE_STATEMENT = re.compile(
    rf'MATCH \(a:Entity \{{key: {STRING}\}}\), '
    rf'\(b:Entity \{{key: {STRING}\}}\) '
    rf'MERGE \(a\)-\[r:[A-Za-z_][A-Za-z0-9_]* '
    rf'\{{relation: {STRING}, ref: {STRING}, evidence: {STRING}\}}\]->\(b\)'
)


@pytest.fixture
def graph():
    store = GraphStore(':memory:')
    yield store
    store.close()


def add_claims(graph):
    entities = [
        Entity(
            key=merge_key('Société "A"'), name='Société "A"',
            type='organisation', mentions=2
        ),
        Entity(
            key=merge_key("O'Brien"), name="O'Brien", type='person',
            mentions=1
        ),
        Entity(key=merge_key('Бета'), name='Бета', mentions=1),
        Entity(key=merge_key('غاما'), name='غاما', mentions=1)
    ]
    evidence = 'Société "A" used C:\\reports.\nSecond line.'
    edges = [
        Edge(
            source='Société "A"', target="O'Brien",
            relation='funded in March', ref='first.md', evidence=evidence
        ),
        Edge(
            source='Бета', target='Société "A"', relation='финансировал',
            ref='second.md', evidence='Бета финансировал Société "A".'
        ),
        Edge(
            source='غاما', target='Société "A"', relation='دعم',
            ref='third.md', evidence='غاما دعم Société "A".'
        )
    ]
    graph.add(entities, edges)

    return evidence


def assert_balanced_statement(statement):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    quoted = False
    escaped = False
    for character in statement:
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in '([{':
            stack.append(character)
        elif character in pairs:
            assert stack and stack.pop() == pairs[character]

    assert quoted is False
    assert escaped is False
    assert stack == []


def assert_parses(statement):
    assert_balanced_statement(statement)
    grammar = (
        EDGE_STATEMENT if statement.startswith('MATCH ') else NODE_STATEMENT
    )
    assert grammar.fullmatch(statement) is not None


def property_string(statement, name):
    match = re.search(
        rf'{name}:\s*("(?:\\.|[^"\\])*")', statement
    )
    assert match is not None

    return json.loads(match.group(1))


def test_cypherl_has_valid_standalone_statements_and_escaped_values(graph):
    evidence = add_claims(graph)

    content = to_cypherl(graph)
    lines = content.splitlines()

    assert len(lines) == graph.entity_count + graph.edge_count
    assert all('MERGE ' in line for line in lines)
    for line in lines:
        assert_parses(line)

    edge_lines = [line for line in lines if line.startswith('MATCH ')]
    assert all(line.count('MATCH ') == 1 for line in edge_lines)
    assert property_string(edge_lines[0], 'evidence') == evidence
    assert '\\n' in edge_lines[0]
    assert property_string(edge_lines[0], 'relation') == 'funded in March'
    assert ':FUNDED_IN_MARCH ' in edge_lines[0]

    node = next(
        line for line in lines
        if 'Société' in line and 'SET n.name' in line
    )
    assert 'Société \\"A\\"' in node
    named = node.replace('n.name =', 'name:')
    assert property_string(named, 'name') == 'Société "A"'
    assert any("O'Brien" in line for line in lines)


def test_non_latin_relations_get_distinct_valid_types_and_keep_wording(graph):
    add_claims(graph)

    edge_lines = [
        line for line in to_cypherl(graph).splitlines()
        if line.startswith('MATCH ')
    ]
    non_latin = [line for line in edge_lines if 'RELATED_' in line]
    types = [re.search(r'\[r:([A-Za-z_][A-Za-z0-9_]*) ', line).group(1)
             for line in non_latin]

    assert len(non_latin) == 2
    assert len(set(types)) == 2
    assert all(
        re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', item) for item in types
    )
    assert {property_string(line, 'relation') for line in non_latin} == {
        'финансировал', 'دعم'
    }


def test_json_is_unicode_and_round_trips_graph_records(graph):
    add_claims(graph)

    content = to_json(graph)
    payload = json.loads(content)

    assert '\\u' not in content
    assert payload == {
        'entities': [asdict(entity) for entity in graph.entities()],
        'edges': [asdict(edge) for edge in graph.edges()]
    }


def test_refs_restrict_edges_and_their_entities_in_both_formats(graph):
    add_claims(graph)

    payload = json.loads(to_json(graph, refs=['second.md']))
    cypherl = to_cypherl(graph, refs=['second.md'])

    assert [edge['ref'] for edge in payload['edges']] == ['second.md']
    assert {entity['name'] for entity in payload['entities']} == {
        'Бета', 'Société "A"'
    }
    assert 'second.md' in cypherl
    assert 'first.md' not in cypherl
    assert 'third.md' not in cypherl
    assert "O'Brien" not in cypherl
    assert 'غاما' not in cypherl


def test_empty_graph_exports_valid_empty_formats(graph, tmp_path):
    cypherl_path = export_graph(graph, tmp_path / 'empty.cypherl')
    json_path = export_graph(graph, tmp_path / 'empty.json')

    assert cypherl_path.read_text(encoding='utf-8') == ''
    assert json.loads(json_path.read_text(encoding='utf-8')) == {
        'entities': [], 'edges': []
    }


def test_export_rejects_an_unknown_suffix(graph, tmp_path):
    with pytest.raises(ValueError, match=r'\.cypherl or \.json'):
        export_graph(graph, tmp_path / 'graph.txt')
