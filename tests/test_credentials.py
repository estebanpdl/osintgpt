'''The credential store, its precedence, and what it refuses to print.'''

import json
import os

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.config import ENV_VARS, secret_fields
from osintgpt.credentials import (
    ENVIRONMENT,
    STORED,
    credential_names,
    credential_status,
    credentials_file,
    field_for,
    load_credentials,
    remove_credential,
    resolve_credentials,
    save_credentials,
    store_credential
)

SECRET = 'sk-not-a-real-key-90210'


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def no_ambient_credentials(monkeypatch):
    '''
    The machine running these tests may have real keys exported. Every
    assertion here is about what osintgpt resolved, so an inherited variable
    would quietly make a passing test meaningless.
    '''
    for field, name in ENV_VARS.items():
        if field in secret_fields():
            monkeypatch.delenv(name, raising=False)


def invoke(runner, home, *arguments, **kwargs):
    return runner.invoke(app, ['--home', str(home), *arguments], **kwargs)


class TestNames:
    def test_provider_names_come_from_the_settings_fields(self):
        names = credential_names()

        assert names['openai'] == 'openai_api_key'
        assert names['postgres'] == 'postgres_dsn'
        assert set(names.values()) == secret_fields()

    @pytest.mark.parametrize(
        'typed', ['openai', 'OpenAI', 'openai_api_key', 'OPENAI_API_KEY']
    )
    def test_a_credential_is_found_by_any_name_an_operator_would_type(
        self, typed
    ):
        assert field_for(typed) == 'openai_api_key'

    def test_a_non_credential_setting_is_not_a_credential(self):
        assert field_for('embedding_provider') is None
        assert field_for('') is None


class TestStore:
    def test_a_stored_credential_reads_back(self, home):
        store_credential(home, 'openai_api_key', SECRET)

        assert load_credentials(home) == {'openai_api_key': SECRET}

    def test_whitespace_around_a_pasted_key_is_dropped(self, home):
        store_credential(home, 'openai_api_key', f'  {SECRET}\n')

        assert load_credentials(home)['openai_api_key'] == SECRET

    def test_an_empty_credential_is_refused(self, home):
        with pytest.raises(ValueError):
            store_credential(home, 'openai_api_key', '   ')

    def test_a_setting_that_is_not_a_credential_is_refused(self, home):
        with pytest.raises(ValueError):
            store_credential(home, 'embedding_provider', 'openai')

    def test_removing_the_last_one_removes_the_file(self, home):
        store_credential(home, 'openai_api_key', SECRET)
        remove_credential(home, 'openai_api_key')

        # "Nothing stored" has one representation rather than two, so a file
        # left behind cannot read as a store that failed to write.
        assert not credentials_file(home).is_file()

    def test_removing_what_was_never_stored_says_so(self, home):
        assert remove_credential(home, 'openai_api_key') is False

    def test_a_key_the_running_version_does_not_know_is_dropped(self, home):
        save_credentials(home, {'openai_api_key': SECRET})
        path = credentials_file(home)
        path.write_text(
            path.read_text(encoding='utf-8') + '\nfuture_api_key = "x"\n',
            encoding='utf-8'
        )

        assert load_credentials(home) == {'openai_api_key': SECRET}

    @pytest.mark.skipif(
        os.name == 'nt', reason='Windows does not model POSIX file modes'
    )
    def test_the_file_is_written_readable_only_by_its_owner(self, home):
        store_credential(home, 'openai_api_key', SECRET)

        assert credentials_file(home).stat().st_mode & 0o077 == 0


class TestPrecedence:
    def test_a_stored_credential_reaches_settings(self, home):
        store_credential(home, 'openai_api_key', SECRET)

        assert resolve_credentials(home).openai_api_key == SECRET

    def test_the_environment_wins(self, home, monkeypatch):
        store_credential(home, 'openai_api_key', SECRET)
        monkeypatch.setenv('OPENAI_API_KEY', 'from-the-environment')

        assert resolve_credentials(home).openai_api_key == (
            'from-the-environment'
        )

    def test_a_stored_credential_still_fills_a_gap_the_environment_leaves(
        self, home, monkeypatch
    ):
        store_credential(home, 'openai_api_key', SECRET)
        store_credential(home, 'gemini_api_key', 'gemini-stored')
        monkeypatch.setenv('OPENAI_API_KEY', 'from-the-environment')

        settings = resolve_credentials(home)

        assert settings.openai_api_key == 'from-the-environment'
        assert settings.gemini_api_key == 'gemini-stored'

    def test_an_override_wins_over_both(self, home, monkeypatch):
        store_credential(home, 'openai_api_key', SECRET)
        monkeypatch.setenv('OPENAI_API_KEY', 'from-the-environment')

        settings = resolve_credentials(home, openai_api_key='explicit')

        assert settings.openai_api_key == 'explicit'

    def test_nothing_stored_leaves_the_environment_alone(
        self, home, monkeypatch
    ):
        monkeypatch.setenv('OPENAI_API_KEY', 'from-the-environment')

        assert resolve_credentials(home).openai_api_key == (
            'from-the-environment'
        )


class TestStatus:
    def test_the_source_of_each_credential_is_reported(
        self, home, monkeypatch
    ):
        store_credential(home, 'openai_api_key', SECRET)
        monkeypatch.setenv('GEMINI_API_KEY', 'g')

        by_provider = {row.provider: row for row in credential_status(home)}

        assert by_provider['openai'].source == STORED
        assert by_provider['gemini'].source == ENVIRONMENT
        assert by_provider['anthropic'].source is None
        assert by_provider['anthropic'].is_set is False

    def test_a_shadowed_credential_is_reported_as_shadowed(
        self, home, monkeypatch
    ):
        '''
        An operator who stores a key and has a stale variable exported is
        using the stale one. Nothing about re-running `auth set` would reveal
        that, so the status has to.
        '''
        store_credential(home, 'openai_api_key', SECRET)
        monkeypatch.setenv('OPENAI_API_KEY', 'stale')

        by_provider = {row.provider: row for row in credential_status(home)}

        assert by_provider['openai'].shadowed is True
        assert by_provider['gemini'].shadowed is False

    def test_status_carries_no_credential_values(self, home, monkeypatch):
        store_credential(home, 'openai_api_key', SECRET)
        monkeypatch.setenv('GEMINI_API_KEY', 'gemini-secret')

        rendered = repr(credential_status(home))

        assert SECRET not in rendered
        assert 'gemini-secret' not in rendered


class TestCommands:
    def test_a_credential_read_from_stdin_is_stored(self, runner, home):
        result = invoke(
            runner, home, 'auth', 'set', 'openai', '--stdin', '--json',
            input=SECRET
        )

        assert result.exit_code == 0
        assert load_credentials(home) == {'openai_api_key': SECRET}

    def test_the_command_never_echoes_what_it_stored(self, runner, home):
        result = invoke(
            runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET
        )

        assert SECRET not in result.output

    def test_listing_shows_presence_and_never_a_value(self, runner, home):
        invoke(runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET)

        result = invoke(runner, home, 'auth', 'list')

        assert 'OPENAI_API_KEY' in result.output
        assert SECRET not in result.output

    def test_listing_json_names_the_source_and_carries_no_value(
        self, runner, home
    ):
        invoke(runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET)

        result = invoke(runner, home, 'auth', 'list', '--json')
        payload = json.loads(result.output)
        rows = {row['provider']: row for row in payload['credentials']}

        assert rows['openai']['source'] == STORED
        assert SECRET not in result.output

    def test_listing_warns_when_the_environment_shadows_a_stored_key(
        self, runner, home, monkeypatch
    ):
        invoke(runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET)
        monkeypatch.setenv('OPENAI_API_KEY', 'stale')

        result = invoke(runner, home, 'auth', 'list')

        assert 'instead of the stored credential' in result.output

    def test_an_unknown_provider_names_the_valid_ones(self, runner, home):
        result = invoke(runner, home, 'auth', 'set', 'openai-2', '--stdin',
                        '--json', input=SECRET)
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert 'openai' in payload['valid_providers']

    def test_removing_a_credential_forgets_it(self, runner, home):
        invoke(runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET)

        result = invoke(runner, home, 'auth', 'remove', 'openai')

        assert result.exit_code == 0
        assert load_credentials(home) == {}

    def test_removing_what_is_not_stored_fails(self, runner, home):
        result = invoke(runner, home, 'auth', 'remove', 'openai')

        assert result.exit_code == 1

    def test_config_set_refuses_a_secret_and_names_the_command_that_works(
        self, runner, home
    ):
        invoke(runner, home, 'project', 'create', 'Case')

        result = invoke(
            runner, home, 'config', 'set', 'openai_api_key', SECRET,
            '--project', 'case', '--json'
        )
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert 'osintgpt auth set openai' in payload['error']
        # The old message pointed at a .env file that the CLI does not read.
        assert '.env' not in payload['error']

    def test_a_refused_secret_is_not_written_anywhere(self, runner, home):
        invoke(runner, home, 'project', 'create', 'Case')
        invoke(
            runner, home, 'config', 'set', 'openai_api_key', SECRET,
            '--project', 'case'
        )

        assert load_credentials(home) == {}
        stored = (home / 'projects' / 'case' / 'project.toml').read_text(
            encoding='utf-8'
        )
        assert SECRET not in stored


class TestReachesTheProviders:
    def test_a_stored_credential_satisfies_a_command_that_needs_one(
        self, runner, home, monkeypatch
    ):
        '''
        The point of the store. Without it this command exits 1 naming
        OPENAI_API_KEY, which is the failure the whole feature exists to fix.
        '''
        invoke(runner, home, 'project', 'create', 'Case')
        invoke(runner, home, 'auth', 'set', 'openai', '--stdin', input=SECRET)

        seen = {}

        def record(provider, settings, model=None, recorder=None):
            seen['key'] = settings.openai_api_key
            raise RuntimeError('stop here')

        monkeypatch.setattr(
            'osintgpt.cli.corpus.build_embedding_provider', record
        )
        invoke(runner, home, 'index', '--project', 'case')

        assert seen['key'] == SECRET
