'''Graph export through Typer's public command surface.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.graph import Edge, Entity, graph_for, merge_key
from osintgpt.projects import Registry


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def create_project(runner, home):
    result = invoke(runner, home, 'project', 'create', 'Case Graph')
    assert result.exit_code == 0

    return Registry.load(home).open('case-graph')


def build_graph(project):
    with graph_for(project) as graph:
        graph.add(
            [
                Entity(key=merge_key('Alpha'), name='Alpha', mentions=1),
                Entity(key=merge_key('Beta'), name='Beta', mentions=1)
            ],
            [
                Edge(
                    source='Alpha', target='Beta', relation='supports',
                    ref='evidence.md', evidence='Alpha supports Beta.'
                )
            ]
        )


def test_graph_export_writes_the_requested_file(runner, home, tmp_path):
    project = create_project(runner, home)
    build_graph(project)
    destination = tmp_path / 'graph.cypherl'

    result = invoke(
        runner, home, 'graph', 'export', str(destination),
        '--project', 'case-graph'
    )

    assert result.exit_code == 0
    assert destination.is_file()
    assert 'Graph exported' in result.output
    assert 'evidence.md' in destination.read_text(encoding='utf-8')


def test_graph_export_json_output_is_only_json(runner, home, tmp_path):
    project = create_project(runner, home)
    build_graph(project)
    destination = tmp_path / 'graph.json'

    result = invoke(
        runner, home, 'graph', 'export', str(destination),
        '--project', 'case-graph', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload['project'] == 'case-graph'
    assert payload['format'] == 'json'
    assert payload['entities'] == 2
    assert payload['edges'] == 1
    assert '\x1b[' not in result.output
    assert json.loads(destination.read_text(encoding='utf-8'))['edges'][0][
        'evidence'
    ] == 'Alpha supports Beta.'


def test_graph_export_refuses_an_unbuilt_graph(runner, home, tmp_path):
    create_project(runner, home)
    destination = tmp_path / 'graph.json'

    result = invoke(
        runner, home, 'graph', 'export', str(destination),
        '--project', 'case-graph'
    )

    assert result.exit_code != 0
    assert 'graph has not been built' in result.output
    assert destination.exists() is False


def test_graph_export_requires_a_selected_project(runner, home, tmp_path):
    result = invoke(
        runner, home, 'graph', 'export', str(tmp_path / 'graph.json')
    )

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output
