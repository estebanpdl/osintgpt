'''CLI/library/tool parity through the same fake similarity graph.'''

import importlib
import json

import pytest
from typer.testing import CliRunner

from osintgpt import Project, snowball
from osintgpt.agentic import ToolContext, snowball_search
from osintgpt.agentic.snowball_output import snowball_data
from osintgpt.cli import app
from osintgpt.projects import Registry
from tests.snowball_fakes import GraphEmbedder, GraphStore

operation = importlib.import_module('osintgpt.agentic.snowball')


def test_cli_library_and_agentic_share_chain_dedupe_dates_and_usage(tmp_path, monkeypatch):
    project = Project.create('Snow', home=tmp_path).with_settings(embedding_model=GraphEmbedder.model)
    project.save()
    Registry.rebuild(tmp_path)
    stores = []

    def factory(*args):
        store = GraphStore(20, 12)
        stores.append(store)
        return store

    monkeypatch.setattr(operation, 'store_for', factory)
    monkeypatch.setattr(operation, 'build_embedding_provider', lambda *a, **k: GraphEmbedder())
    options = dict(depth=12, threshold=.5, date_field='published', date_format='%d/%m/%Y')
    library = snowball(project, 'start', home=tmp_path, **options)
    tool = snowball_search(ToolContext(project, GraphEmbedder(), store=GraphStore(20, 12)), 'start', **options)
    target = tmp_path / 'chain.xlsx'
    cli = CliRunner().invoke(app, [
        '--home', str(tmp_path), 'snowball', 'start', '--project', project.slug,
        '--depth', '12', '--threshold', '0.5', '--date-field', 'published',
        '--date-format', '%d/%m/%Y', '--export', str(target), '--json',
    ])
    assert cli.exit_code == 0, cli.output
    result = json.loads(cli.output)
    assert result['hops'] == snowball_data(library)['hops']
    assert result['usage'] == tool.payload['usage'] == snowball_data(library)['usage']
    assert [row['text'] for row in tool.payload['hops']] == [hop.text for hop in library.hops]
    assert [row['identical_text_refs'] for row in tool.payload['hops']] == [hop.identical_text_refs for hop in library.hops]
    assert [row['similarity_to_query'] for row in tool.payload['hops']] == [hop.drift for hop in library.hops]
    assert all(store.closed for store in stores) and target.is_file()


def test_cli_text_grouping_ref_scope_and_no_drift(tmp_path, monkeypatch):
    project = Project.create('Snow', home=tmp_path)
    Registry.rebuild(tmp_path)
    monkeypatch.setattr(operation, 'store_for', lambda *a: GraphStore())
    monkeypatch.setattr(operation, 'build_embedding_provider', lambda *a, **k: GraphEmbedder())
    cli = CliRunner().invoke(app, [
        '--home', str(tmp_path), 'snowball', 'start', '-p', project.slug,
        '--depth', '4', '--group-by', 'month', '--ref', 'posts.jsonl', '--no-drift',
    ])
    assert cli.exit_code == 0, cli.output
    assert '4 hops' in cli.output and 'Unknown date' in cli.output
    assert 'similarity to query off' in cli.output
    assert '4 embedding calls' in cli.output


@pytest.mark.parametrize('arguments', [
    ['--depth', '0'], ['--threshold', 'nan'], ['--group-by', 'hour'],
    ['--dedupe', 'unknown'], ['--export', 'out.txt'],
])
def test_cli_rejects_invalid_inputs_without_provider_calls(tmp_path, monkeypatch, arguments):
    project = Project.create('Snow', home=tmp_path)
    Registry.rebuild(tmp_path)
    monkeypatch.setattr(operation, 'build_embedding_provider', lambda *a, **k: pytest.fail('provider built'))
    result = CliRunner().invoke(app, ['--home', str(tmp_path), 'snowball', 'start', '-p', project.slug, *arguments, '--json'])
    assert result.exit_code != 0


def test_cli_refuses_existing_export_before_paid_work(tmp_path, monkeypatch):
    target = tmp_path / 'saved.csv'
    target.write_text('keep')
    monkeypatch.setattr(operation, 'build_embedding_provider', lambda *a, **k: pytest.fail('provider built'))
    result = CliRunner().invoke(app, ['--home', str(tmp_path), 'snowball', 'start', '--export', str(target), '--json'])
    assert result.exit_code == 1 and 'already exists' in json.loads(result.output)['error']
    assert target.read_text() == 'keep'


def test_snowball_help_describes_limits_and_parameters(cli_help):
    text = cli_help('snowball')
    for phrase in ['similarity path', 'causation', '--depth', '--threshold', '--export', '--dedupe', '--date-field']:
        assert phrase in text
