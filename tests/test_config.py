# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_config.py
# Description: Settings construction, environment loading, validation, and the
#   deprecated .env-path calling convention.
# =================================================================================

# import modules
import os
import pytest

# import osintgpt config
from osintgpt.config import (
    DEFAULT_EMBEDDING_MODEL,
    ENV_VARS,
    Settings,
    resolve_settings
)

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError


class TestConstruction:
    def test_every_field_is_optional(self):
        settings = Settings()

        assert settings.openai_api_key == ''
        assert settings.qdrant_port is None

    def test_embedding_model_has_a_current_default(self):
        assert Settings().openai_embedding_model == DEFAULT_EMBEDDING_MODEL
        assert DEFAULT_EMBEDDING_MODEL != 'text-embedding-ada-002'

    def test_is_frozen(self):
        settings = Settings(openai_api_key='k')

        with pytest.raises(Exception):
            settings.openai_api_key = 'other'

    def test_with_overrides_copies_rather_than_mutates(self):
        base = Settings(openai_api_key='k1', openai_gpt_model='m')
        derived = base.with_overrides(openai_api_key='k2')

        assert derived.openai_api_key == 'k2'
        assert derived.openai_gpt_model == 'm'
        assert base.openai_api_key == 'k1'


class TestFromEnv:
    def test_reads_every_mapped_variable(self, env_file):
        settings = Settings.from_env(env_file)

        assert settings.openai_gpt_model == 'gpt-4o'
        assert settings.qdrant_host == 'localhost'

    def test_parses_the_port_as_an_integer(self, env_file):
        assert Settings.from_env(env_file).qdrant_port == 6333

    def test_rejects_a_non_numeric_port(self, tmp_path):
        path = tmp_path / 'bad.env'
        path.write_text('QDRANT_PORT=not-a-number\n', encoding='utf-8')

        with pytest.raises(ValueError, match='QDRANT_PORT'):
            Settings.from_env(str(path))

    def test_overrides_beat_the_environment(self, env_file):
        settings = Settings.from_env(env_file, openai_gpt_model='gpt-4o-mini')

        assert settings.openai_gpt_model == 'gpt-4o-mini'
        assert settings.qdrant_host == 'localhost'

    def test_process_environment_beats_the_file(self, env_file, monkeypatch):
        monkeypatch.setenv('OPENAI_GPT_MODEL', 'from-the-shell')

        assert Settings.from_env(env_file).openai_gpt_model == 'from-the-shell'

    def test_absent_variable_keeps_the_field_default(self, tmp_path):
        path = tmp_path / 'partial.env'
        path.write_text('OPENAI_API_KEY=k\n', encoding='utf-8')

        settings = Settings.from_env(str(path))

        assert settings.openai_embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_without_a_path_reads_only_the_process_environment(self, monkeypatch):
        monkeypatch.setenv('OPENAI_API_KEY', 'from-the-shell')

        assert Settings.from_env().openai_api_key == 'from-the-shell'

    def test_does_not_mutate_the_environment(self, env_file):
        Settings.from_env(env_file)

        assert os.getenv('OPENAI_GPT_MODEL') is None

    def test_two_files_stay_independent(self, env_file, tmp_path):
        other = tmp_path / 'other.env'
        other.write_text('OPENAI_GPT_MODEL=other-model\n', encoding='utf-8')

        first = Settings.from_env(env_file)
        second = Settings.from_env(str(other))
        third = Settings.from_env(env_file)

        assert first.openai_gpt_model == 'gpt-4o'
        assert second.openai_gpt_model == 'other-model'
        assert third.openai_gpt_model == 'gpt-4o'


class TestRequire:
    def test_passes_when_the_fields_have_values(self):
        settings = Settings(openai_api_key='k')

        assert settings.require('openai_api_key') is settings

    def test_names_the_variable_and_the_field(self):
        with pytest.raises(MissingEnvironmentVariableError) as excinfo:
            Settings(openai_api_key='k').require(
                'openai_api_key', 'openai_gpt_model'
            )

        message = str(excinfo.value)

        assert 'OPENAI_GPT_MODEL' in message
        assert 'Settings(openai_gpt_model=' in message

    def test_every_field_maps_to_a_variable(self):
        for field in Settings().__dataclass_fields__:
            assert field in ENV_VARS, f'{field} has no environment variable'


class TestResolveSettings:
    def test_passes_settings_through_unchanged(self):
        settings = Settings(openai_api_key='k')

        assert resolve_settings(settings) is settings

    def test_accepts_a_path_with_a_deprecation_warning(self, env_file):
        with pytest.warns(DeprecationWarning, match='Settings'):
            settings = resolve_settings(env_file)

        assert settings.openai_gpt_model == 'gpt-4o'

    @pytest.mark.parametrize('value', [None, 42, {'openai_api_key': 'k'}, []])
    def test_rejects_anything_else(self, value):
        with pytest.raises(TypeError):
            resolve_settings(value)
