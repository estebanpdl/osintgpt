'''Quota settings persist and reach providers without becoming index inputs.'''

import json

import pytest

from osintgpt.config import Settings
from osintgpt.credentials import resolve_credentials
from osintgpt.llm import build_embedding_provider
from osintgpt.projects import ProjectSettings, Registry, save_user_defaults
from osintgpt.rate_settings import RATE_ENV_VARS
from tests.cli.test_cli_config import create_project, home, invoke, runner


@pytest.mark.parametrize('bad', [-1, 1.5, True, '2000000'])
def test_invalid_quotas_are_rejected_at_configuration_boundary(bad):
    with pytest.raises(ValueError, match='whole number'):
        Settings(openai_embedding_tpm=bad)
    with pytest.raises(ValueError, match='whole number'):
        ProjectSettings(gemini_embedding_rpm=bad)


def test_environment_parses_numbers_and_explicit_overrides_win(monkeypatch):
    monkeypatch.setenv(RATE_ENV_VARS['openai_embedding_tpm'], '2000000')
    monkeypatch.setenv(RATE_ENV_VARS['openai_embedding_quota_scope'], 'shared-organization')
    assert Settings.from_env().openai_embedding_tpm == 2_000_000
    assert Settings.from_env(openai_embedding_tpm=3_000_000).openai_embedding_tpm == 3_000_000
    assert Settings.from_env().openai_embedding_quota_scope == 'shared-organization'


def test_rate_fields_do_not_change_existing_positional_arguments():
    assert Settings('key').openai_api_key == 'key'
    assert ProjectSettings('voyage').embedding_provider == 'voyage'


def test_cli_round_trip_inheritance_and_provider_isolation(runner, home):
    project = create_project(runner, home)
    assert invoke(runner, home, 'project', 'use', project.slug).exit_code == 0
    for key, value in [('openai_embedding_tpm', '2000000'), ('openai_embedding_rpm', '5000'), ('openai_embedding_batch_tokens', '50000'), ('openai_embedding_quota_scope', 'organization-model-group')]:
        result = invoke(runner, home, 'config', 'set', key, value, '--json')
        assert result.exit_code == 0, result.output
    saved = Registry.load(home).open(project.slug)
    assert saved.settings.openai_embedding_tpm == 2_000_000
    assert saved.settings.gemini_embedding_tpm is None
    configuration = saved.settings_for(resolve_credentials(home, openai_api_key='test'))
    assert configuration.embedding_rate_state_path == str((home / 'embedding-rates.sqlite').resolve())
    embedder = build_embedding_provider('openai', configuration)
    try:
        assert embedder.pacer.configured.tpm == 2_000_000
        assert embedder.pacer.configured.rpm == 5000
        assert embedder.pacer.configured.batch_tokens == 50000
    finally:
        embedder.client.close()
    assert invoke(runner, home, 'config', 'set', 'openai_embedding_tpm', 'unset').exit_code == 0
    reloaded = Registry.load(home).open(project.slug)
    defaults = ProjectSettings(openai_embedding_tpm=1_500_000)
    assert reloaded.settings_for(Settings(), defaults).openai_embedding_tpm == 1_500_000
    assert invoke(runner, home, 'config', 'set', 'openai_embedding_tpm', '0').exit_code == 0
    assert Registry.load(home).open(project.slug).settings_for(Settings(), defaults).openai_embedding_tpm == 0
    get = invoke(runner, home, 'config', 'get', 'openai_embedding_tpm', '--json')
    assert json.loads(get.output)['value'] == 0


def test_cli_rejects_invalid_rate_without_changing_project(runner, home):
    project = create_project(runner, home)
    assert invoke(runner, home, 'project', 'use', project.slug).exit_code == 0
    original = project.paths.config.read_bytes()
    for value in ('-1', 'nan', '1.5'):
        result = invoke(runner, home, 'config', 'set', 'voyage_embedding_rpm', '--', value)
        assert result.exit_code != 0
        assert 'whole number' in result.output
    assert project.paths.config.read_bytes() == original


def test_project_and_user_limits_never_override_explicit_environment(tmp_path, monkeypatch):
    from osintgpt import Project
    from osintgpt.projects import load_user_defaults
    project = Project.create('Rate precedence', home=tmp_path).with_settings(openai_embedding_tpm=1_000_000)
    save_user_defaults(tmp_path, ProjectSettings(openai_embedding_tpm=500_000, voyage_embedding_tpm=800_000))
    monkeypatch.setenv('OSINTGPT_OPENAI_EMBEDDING_TPM', '2000000')
    config = project.settings_for(resolve_credentials(tmp_path), load_user_defaults(tmp_path))
    assert config.openai_embedding_tpm == 2_000_000
    assert config.voyage_embedding_tpm == 800_000
