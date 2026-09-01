'''Graph export through Typer's public command surface.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.config import ENV_VARS
from osintgpt.graph import Edge, Entity, graph_for, merge_key
from osintgpt.ingestion import Corpus
from osintgpt import llm
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


def clear_credentials(monkeypatch):
    for field, name in ENV_VARS.items():
        if field.endswith('_api_key') or field.endswith('_dsn'):
            monkeypatch.delenv(name, raising=False)


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


def test_graph_verify_reports_counts_as_json(runner, home):
    project = create_project(runner, home)
    (project.paths.root / 'evidence.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    build_graph(project)

    result = invoke(
        runner, home, 'graph', 'verify',
        '--project', 'case-graph', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload['summary'] == '1 verified'
    assert payload['total'] == 1
    assert payload['found'] == 1
    assert payload['failures'] == []


def test_graph_verify_findings_exit_zero_unless_strict(runner, home):
    project = create_project(runner, home)
    (project.paths.root / 'evidence.md').write_text(
        'Something else.', encoding='utf-8'
    )
    build_graph(project)

    ordinary = invoke(
        runner, home, 'graph', 'verify',
        '--project', 'case-graph', '--json'
    )
    strict = invoke(
        runner, home, 'graph', 'verify',
        '--project', 'case-graph', '--json', '--strict'
    )

    payload = json.loads(ordinary.output)
    assert ordinary.exit_code == 0
    assert strict.exit_code == 1
    assert payload['not_found'] == 1
    assert payload['failures'][0]['ref'] == 'evidence.md'
    assert payload['failures'][0]['status'] == 'not_found'


def test_graph_verify_an_unbuilt_graph_reports_nothing(runner, home):
    create_project(runner, home)

    result = invoke(
        runner, home, 'graph', 'verify',
        '--project', 'case-graph', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload['summary'] == 'nothing to verify'
    assert payload['total'] == 0


def test_graph_verify_requires_a_selected_project(runner, home):
    result = invoke(runner, home, 'graph', 'verify')

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output


def test_graph_build_prints_cost_progress_and_report(
    runner, home, monkeypatch
):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    material = project.paths.root / 'evidence.md'
    material.write_text('Alpha supports Beta.', encoding='utf-8')
    Corpus.load(project.paths.sources).register('evidence.md')

    class Generator:
        model = 'test-chat'

        def generate(self, system, user):
            return json.dumps({
                'entities': [{'name': 'Alpha'}, {'name': 'Beta'}],
                'edges': [{
                    'source': 'Alpha', 'target': 'Beta',
                    'relation': 'supports',
                    'evidence': 'Alpha supports Beta.'
                }]
            })

    monkeypatch.setattr(
        llm, 'build_generation_provider',
        lambda provider, settings, model=None: Generator()
    )

    result = invoke(
        runner, home, 'graph', 'build', '--project', 'case-graph'
    )

    assert result.exit_code == 0
    assert 'one generation call per document' in result.output
    assert '1/1 evidence.md' in result.output
    assert '1 documents, 2 entities, 1 edges' in result.output


def test_graph_build_refuses_when_disabled_without_building_provider(
    runner, home, monkeypatch
):
    create_project(runner, home)

    def forbidden(*args, **kwargs):
        raise AssertionError('a refusal must not build a provider')

    monkeypatch.setattr(llm, 'build_generation_provider', forbidden)
    result = invoke(
        runner, home, 'graph', 'build', '--project', 'case-graph'
    )

    assert result.exit_code != 0
    assert 'graph is off' in result.output


def test_graph_build_refuses_incremental_before_first_build(
    runner, home, monkeypatch
):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()

    def forbidden(*args, **kwargs):
        raise AssertionError('a refusal must not build a provider')

    monkeypatch.setattr(llm, 'build_generation_provider', forbidden)
    result = invoke(
        runner, home, 'graph', 'build', '--incremental',
        '--project', 'case-graph'
    )

    assert result.exit_code != 0
    assert 'graph has not been built yet' in result.output


def test_graph_build_refuses_conflicting_maintenance_modes(runner, home):
    result = invoke(
        runner, home, 'graph', 'build', '--incremental', '--rebuild'
    )

    assert result.exit_code != 0
    assert 'cannot be used together' in result.output


def test_graph_neighbors_and_path_are_keyless(runner, home, monkeypatch):
    project = create_project(runner, home)
    build_graph(project)
    clear_credentials(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError('traversal must not build a provider')

    monkeypatch.setattr(llm, 'build_generation_provider', forbidden)

    nearby = invoke(
        runner, home, 'graph', 'neighbors', 'Alpha',
        '--project', 'case-graph', '--json'
    )
    connected = invoke(
        runner, home, 'graph', 'path', 'Alpha', 'Beta',
        '--project', 'case-graph', '--json'
    )

    nearby_payload = json.loads(nearby.output)
    path_payload = json.loads(connected.output)
    assert nearby.exit_code == 0
    assert connected.exit_code == 0
    assert nearby_payload['results'][0]['ref'] == 'evidence.md'
    assert nearby_payload['results'][0]['evidence'] == 'Alpha supports Beta.'
    assert path_payload['edges'][0]['ref'] == 'evidence.md'
    assert path_payload['edges'][0]['evidence'] == 'Alpha supports Beta.'


def test_graph_neighbors_walks_the_requested_depth(runner, home):
    project = create_project(runner, home)
    build_graph(project)
    with graph_for(project) as graph:
        graph.add(
            [Entity(key=merge_key('Gamma'), name='Gamma', mentions=1)],
            [Edge(
                source='Beta', target='Gamma', relation='informed',
                ref='second.md', evidence='Beta informed Gamma.'
            )]
        )

    result = invoke(
        runner, home, 'graph', 'neighbors', 'Alpha', '--depth', '2',
        '--project', 'case-graph', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert [row['depth'] for row in payload['results']] == [1, 2]
    assert [row['ref'] for row in payload['results']] == [
        'evidence.md', 'second.md'
    ]


@pytest.mark.parametrize(
    'command', [('neighbors', 'Alpha'), ('path', 'Alpha', 'Beta')]
)
def test_graph_traversal_refuses_an_unbuilt_graph(
    runner, home, command
):
    create_project(runner, home)

    result = invoke(
        runner, home, 'graph', *command, '--project', 'case-graph'
    )

    assert result.exit_code != 0
    assert 'graph has not been built' in result.output
