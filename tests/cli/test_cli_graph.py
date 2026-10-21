'''Graph export through Typer's public command surface.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.config import ENV_VARS
from osintgpt.graph import Edge, Entity, graph_for, merge_key
from osintgpt.ingestion import Corpus
from osintgpt import llm
from osintgpt.llm import Usage
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

    project = Registry.load(home).open('case-graph').with_settings(generation_model='test-chat')
    project.save()
    return project


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

        def __init__(self, recorder):
            self.recorder = recorder

        def generate(self, system, user):
            self.recorder.record(Usage(
                'stub', self.model, input_tokens=10, output_tokens=5
            ))

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
        lambda provider, settings, model=None, recorder=None: (
            Generator(recorder)
        )
    )

    result = invoke(
        runner, home, 'graph', 'build', '--project', 'case-graph'
    )

    assert result.exit_code == 0
    assert 'processes source windows with identity lookup' in result.output
    assert '1/1 evidence.md' in result.output
    assert '1 documents, 2 entities, 1 edges' in result.output
    assert 'Usage:' in result.output


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


def test_graph_build_and_status_report_windows_and_usage(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'large.md').write_text('x' * 100000)
    Corpus.load(project.paths.sources).register('large.md')

    class Generator:
        def __init__(self, recorder):
            self.recorder = recorder

        def generate(self, system, user):
            self.recorder.record(Usage('stub', 'test-chat', 10, 5))
            return '{"entities": [], "edges": []}'

    monkeypatch.setattr(llm, 'build_generation_provider',
                        lambda *args, recorder=None, **kwargs: Generator(recorder))
    result = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['status'] == 'complete'
    assert data['documents'] == 1
    assert data['calls'] == data['accounting']['windows_completed'] > 1
    assert data['usage']['total_tokens'] == data['calls'] * 15
    assert data['usage']['complete'] is False

    def forbidden(*args, **kwargs):
        raise AssertionError('Status must not create a provider')

    monkeypatch.setattr(llm, 'build_generation_provider', forbidden)
    result = invoke(runner, home, 'graph', 'status', '--project', 'case-graph', '--json')
    assert result.exit_code == 0, result.output
    status = json.loads(result.output)
    assert status['status'] == 'complete'
    assert status['available']
    assert status['counts'] == data['accounting']
    assert status['usage'] == data['usage']
    destination = home / 'empty-graph.json'
    result = invoke(runner, home, 'graph', 'export', str(destination),
                    '--project', 'case-graph', '--json')
    assert result.exit_code == 0
    exported = json.loads(destination.read_text(encoding='utf-8'))
    assert exported['entities'] == exported['edges'] == []
    assert exported['metadata']['graph']['build_status'] == 'complete'
    assert exported['metadata']['project']['id'] == project.id


def test_graph_status_does_not_create_unused_database(runner, home):
    project = create_project(runner, home)
    result = invoke(runner, home, 'graph', 'status', '--project', 'case-graph', '--json')
    assert result.exit_code == 0
    assert json.loads(result.output)['status'] == 'never_built'
    assert not (project.paths.root / 'graph.sqlite').exists()


def test_rejection_inspection_is_keyless_and_does_not_create_unused_graph(runner, home, monkeypatch):
    from osintgpt.graph.records import RejectedClaim
    project = create_project(runner, home)
    clear_credentials(monkeypatch)
    result = invoke(runner, home, 'graph', 'rejected', '--project', 'case-graph', '--json')
    assert result.exit_code == 0 and json.loads(result.output)['results'] == []
    assert not (project.paths.root / 'graph.sqlite').exists()
    with graph_for(project) as graph:
        graph.record_rejections([RejectedClaim('a.csv#one', 'a.csv', 'version', 2,
                                              {'evidence': 'Not in source'}, 'Quotation not found')])
    result = invoke(runner, home, 'graph', 'rejected', '--project', 'case-graph', '--ref', 'a.csv', '--json')
    assert result.exit_code == 0
    assert json.loads(result.output)['results'][0]['candidate']['evidence'] == 'Not in source'


def test_human_graph_rows_preserve_negative_and_alleged_status(runner, home):
    project = create_project(runner, home)
    with graph_for(project) as graph:
        graph.add([], [Edge('A', 'B', 'funded', 'a.md', 'A did not fund B.',
                            relation_polarity='negative', assertion='alleged')])
    for command in [('neighbors', 'A'), ('path', 'A', 'B')]:
        result = invoke(runner, home, 'graph', *command, '--project', 'case-graph')
        assert result.exit_code == 0
        assert 'negative' in result.output and 'alleged' in result.output


def test_invalid_graph_response_exits_nonzero_with_failed_state(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'a.md').write_text('Text')
    Corpus.load(project.paths.sources).register('a.md')

    class Generator:
        def generate(self, *args):
            return 'not json'

    monkeypatch.setattr(llm, 'build_generation_provider', lambda *args, **kwargs: Generator())
    result = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
    data = json.loads(result.output)
    assert result.exit_code == 1
    assert data['status'] == 'failed'
    assert data['accounting']['invalid_responses'] == 2
    assert data['usage']['missing_usage_requests'] == 2


def test_unchanged_cli_refresh_is_keyless_and_reports_reuse(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'a.md').write_text('Source')
    Corpus.load(project.paths.sources).register('a.md')
    class Generator:
        def generate(self, *args):
            return '{"entities": [], "edges": []}'
    monkeypatch.setattr(llm, 'build_generation_provider', lambda *args, **kwargs: Generator())
    first = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
    assert first.exit_code == 0, first.output
    clear_credentials(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reusing completed sources must not construct a provider')
    monkeypatch.setattr(llm, 'build_generation_provider', forbidden)
    result = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data['calls'] == 0 and data['skipped'] == 1
    assert data['details'][0]['action'] == 'reused'


def test_configured_fallback_model_is_part_of_graph_recipe():
    from types import SimpleNamespace
    from osintgpt.cli.graph import _LazyGenerator
    from osintgpt.config import Settings
    effective = SimpleNamespace(generation_model='', generation_provider='openai')
    first = _LazyGenerator(None, effective, Settings(openai_gpt_model='first'), None)
    second = _LazyGenerator(None, effective, Settings(openai_gpt_model='second'), None)
    assert first.graph_options()['model'] == 'first'
    assert second.graph_options()['model'] == 'second'
    assert first._built is second._built is None


@pytest.mark.parametrize('reason,exit_code', [('', 0), ('length', 1)])
def test_cli_passes_schema_and_completion_status_through_lazy_provider(runner, home, monkeypatch, reason, exit_code):
    from osintgpt.graph.contract import EXTRACTION_SCHEMA
    from osintgpt.llm.generation import GenerationResponse
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'a.md').write_text('Source')
    Corpus.load(project.paths.sources).register('a.md')
    schemas = []

    class Generator:
        supports_structured_output = True
        supports_output_limit = True

        def generate_response(self, system, user, *, max_output_tokens=None, schema=None):
            schemas.append(schema)
            return GenerationResponse('{"entities":[],"edges":[]}', status='completed', stop_reason=reason)

    monkeypatch.setattr(llm, 'build_generation_provider', lambda *args, **kwargs: Generator())
    result = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
    assert result.exit_code == exit_code, result.output
    data = json.loads(result.output)
    assert schemas == [EXTRACTION_SCHEMA]
    assert data['accounting']['structured_output_unenforced'] == 0
    assert data['accounting']['completion_status_unavailable'] == 0
    assert data['accounting']['invalid_responses'] == exit_code

    if not reason:
        def forbidden(*args, **kwargs):
            raise AssertionError('Compatible reuse must remain keyless')
        monkeypatch.setattr(llm, 'build_generation_provider', forbidden)
        again = invoke(runner, home, 'graph', 'build', '--project', 'case-graph', '--json')
        assert again.exit_code == 0, again.output
        assert json.loads(again.output)['skipped'] == 1
