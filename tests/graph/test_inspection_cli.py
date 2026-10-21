'''CLI inspection exercises the same evidence contract as model tools.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.graph import Entity, graph_for
from osintgpt.projects import Registry


def invoke(case, *arguments):
    case.project.save()
    Registry.load(case.project.paths.root.parent.parent).register(case.project)
    return CliRunner().invoke(app, ['--home', str(case.project.paths.root.parent.parent),
                                   'graph', *arguments, '--project', case.project.slug])


def test_identity_direction_and_evidence_commands_without_provider(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.')
    case.project = case.project.with_settings(graph_enabled=False)
    with graph_for(case.project) as graph:
        graph.add([Entity('other_alpha', 'Alpha', description='Different identity')], [])
    result = invoke(case, 'entities', 'Alpha', '--json')
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['status'] == 'ambiguous'
    result = invoke(case, 'neighbors', 'alpha', '--direction', 'outgoing', '--json')
    row = json.loads(result.output)['results'][0]
    assert row['assertion_id'] == edge.assertion_id
    assert row['verification']['snapshot'] == 'found'
    result = invoke(case, 'path', 'beta', 'alpha', '--direction', 'incoming', '--json')
    assert json.loads(result.output)['edges'][0]['traversal']['against_direction']
    result = invoke(case, 'evidence', edge.assertion_id, '--json')
    assert result.exit_code == 0 and json.loads(result.output)['text'] == 'Alpha funds Beta.'
    assert invoke(case, 'evidence', edge.assertion_id).exit_code == 0
    assert invoke(case, 'neighbors', 'alpha', '--direction', 'nonsense').exit_code != 0


def test_human_scope_messages_keep_model_json_bounds_separate(graph_case):
    case = graph_case
    case.publish('one.md', 'Alpha supports Beta.')
    human = invoke(case, 'neighbors', 'alpha', '--depth', '1')
    assert human.exit_code == 0, human.output
    assert 'Exploration complete within the selected scope' in human.output
    assert 'depth_boundary' not in human.output and 'Limited results' not in human.output
    assert 'Relation polarity' in human.output
    machine = json.loads(invoke(case, 'neighbors', 'alpha', '--depth', '1', '--json').output)
    assert machine['truncated'] and 'depth_boundary' in machine['truncation_reasons']
    assert 'relation_polarity' in machine['results'][0]
    assert 'polarity' not in machine['results'][0]


@pytest.mark.parametrize('command', ['entities', 'neighbors', 'path', 'evidence'])
def test_inspection_help_describes_operator_behavior(command, cli_help):
    assert '--max-bytes' in cli_help('graph', command)
