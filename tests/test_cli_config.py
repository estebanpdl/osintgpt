'''Project settings and user defaults through the CLI.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.projects import Registry, load_user_defaults


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def create_project(runner, home):
    result = invoke(runner, home, 'project', 'create', 'Case Config')
    assert result.exit_code == 0

    return Registry.load(home).open('case-config')


def test_unknown_key_fails_and_lists_dynamic_valid_keys(runner, home):
    create_project(runner, home)

    result = invoke(
        runner, home, 'config', 'get', 'imaginary',
        '--project', 'case-config'
    )

    assert result.exit_code != 0
    assert 'embedding_provider' in result.output
    assert 'cost_ceiling_usd' in result.output
    assert 'openai_api_key' in result.output


def test_get_secret_reports_status_without_printing_value(
    runner, home, monkeypatch
):
    create_project(runner, home)
    secret = 'do-not-display-this-key'
    monkeypatch.setenv('OPENAI_API_KEY', secret)

    result = invoke(
        runner, home, 'config', 'get', 'openai_api_key',
        '--project', 'case-config', '--json'
    )

    payload = json.loads(result.output)
    assert payload['value'] == {'set': True}
    assert secret not in result.output


def test_set_api_key_refuses_and_names_where_a_key_belongs(runner, home):
    create_project(runner, home)

    result = invoke(
        runner, home, 'config', 'set', 'openai_api_key', 'secret',
        '--project', 'case-config'
    )

    assert result.exit_code != 0
    assert 'OPENAI_API_KEY' in result.output
    assert 'osintgpt auth set openai' in result.output


def test_set_user_writes_defaults_and_not_project(runner, home):
    project = create_project(runner, home)

    result = invoke(
        runner, home, 'config', 'set', 'embedding_model', 'user-model',
        '--user', '--project', 'case-config', '--json'
    )

    assert result.exit_code == 0
    assert load_user_defaults(home).embedding_model == 'user-model'
    assert project.settings.embedding_model == ''
    assert Registry.load(home).open('case-config').settings.embedding_model == ''


def test_set_parses_project_boolean_and_number(runner, home):
    create_project(runner, home)

    first = invoke(
        runner, home, 'config', 'set', 'graph_enabled', 'true',
        '--project', 'case-config'
    )
    second = invoke(
        runner, home, 'config', 'set', 'cost_ceiling_usd', '12.5',
        '--project', 'case-config'
    )

    project = Registry.load(home).open('case-config')
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert project.settings.graph_enabled is True
    assert project.settings.cost_ceiling_usd == 12.5


@pytest.mark.parametrize(
    'arguments',
    [('config', 'get'), ('config', 'set', 'graph_enabled', 'true', '--user')]
)
def test_config_requires_a_selected_project(runner, home, arguments):
    result = invoke(runner, home, *arguments)

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output
