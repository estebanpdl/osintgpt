# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: config.py
# Description: The config.py file contains the Settings class, the configuration
#   osintgpt classes receive as an argument. Library code reads values from a
#   Settings instance and never from the process environment.
# =================================================================================

# import modules
import os
import warnings

# import submodules
from dataclasses import dataclass, replace
from dotenv import dotenv_values

# type hints
from typing import Optional, Union

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

# Current, cheap, and available to every account.
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'

# Maps each setting to the environment variable `Settings.from_env` reads it
# from. Single source of truth for both the loader and the error messages.
ENV_VARS = {
    'openai_api_key': 'OPENAI_API_KEY',
    'openai_gpt_model': 'OPENAI_GPT_MODEL',
    'openai_embedding_model': 'OPENAI_EMBEDDING_MODEL',
    'sql_db_file_path': 'SQL_DB_FILE_PATH',
    'qdrant_api_key': 'QDRANT_API_KEY',
    'qdrant_url': 'QDRANT_URL',
    'qdrant_host': 'QDRANT_HOST',
    'qdrant_port': 'QDRANT_PORT'
}

# Settings class
@dataclass(frozen=True)
class Settings:
    '''
    Settings class

    Configuration handed to osintgpt classes. Every field is optional at
    construction; classes declare what they need via `require`, so a Settings
    carrying only an OpenAI key is valid for embeddings and rejected by Qdrant.
    '''
    openai_api_key: str = ''
    openai_gpt_model: str = ''
    openai_embedding_model: str = ''
    sql_db_file_path: str = ''
    qdrant_api_key: str = ''
    qdrant_url: str = ''
    qdrant_host: str = ''
    qdrant_port: Optional[int] = None

    # build settings from the environment
    @classmethod
    def from_env(cls, env_file_path: Optional[str] = None, **overrides):
        '''
        Build Settings from environment variables, optionally reading a .env
        file. The one place osintgpt reads the environment at all.

        Precedence is overrides, then the process environment, then the file.
        The file is parsed, never loaded: `os.environ` is left untouched, so two
        calls with two different files return two different Settings in the same
        process.

        Args:
            env_file_path (str, optional): Path to a .env file. When omitted, \
                only variables already present in the environment are read; no \
                file is searched for.
            **overrides: Field values that win over everything else.

        Returns:
            Settings: A new instance.
        '''
        from_file = dotenv_values(env_file_path) if env_file_path else {}

        # Empty values are omitted rather than passed as '' so that fields with
        # a real default (the embedding model) keep it instead of being blanked.
        values = {}
        for field, name in ENV_VARS.items():
            value = os.getenv(name) or from_file.get(name)
            if value:
                values[field] = value

        if 'qdrant_port' in values:
            values['qdrant_port'] = _parse_port(values['qdrant_port'])

        values.update(overrides)

        return cls(**values)

    # check that required settings carry a value
    def require(self, *fields: str):
        '''
        Assert that every named field has a value.

        Args:
            *fields (str): Setting names to check.

        Raises:
            MissingEnvironmentVariableError: If any named field is empty.

        Returns:
            Settings: self, so callers can validate and assign in one line.
        '''
        for field in fields:
            if getattr(self, field) in (None, ''):
                raise MissingEnvironmentVariableError(
                    ENV_VARS.get(field, field),
                    hint=f'pass Settings({field}=...) or set it in your .env file'
                )

        return self

    # copy with replaced fields
    def with_overrides(self, **changes):
        '''
        A copy of these settings with fields replaced.

        Args:
            **changes: Field values to replace.

        Returns:
            Settings: A new instance; the original is unchanged.
        '''
        return replace(self, **changes)


# parse a port value
def _parse_port(value: Union[str, int, None]):
    '''
    Parse a port into an int, treating an empty value as unset.

    Args:
        value (Union[str, int, None]): Raw port value.

    Raises:
        ValueError: If the value is present but not a number.

    Returns:
        Optional[int]: Port number, or None when unset.
    '''
    if value in (None, ''):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{ENV_VARS["qdrant_port"]} must be a number, got {value!r}'
        ) from None


# resolve a config argument into Settings
def resolve_settings(config: Union[Settings, str, None]):
    '''
    Normalize the `config` argument osintgpt classes accept.

    A string is read as a path to a .env file — the pre-0.2 calling
    convention, kept working and warned about. None is rejected: reading the
    environment implicitly is the behaviour these settings exist to remove.

    Args:
        config (Union[Settings, str]): Settings, or a path to a .env file.

    Raises:
        TypeError: If config is None or an unsupported type.

    Returns:
        Settings: The resolved settings.
    '''
    if isinstance(config, Settings):
        return config

    if isinstance(config, str):
        warnings.warn(
            'Passing a .env file path is deprecated and will be removed in a '
            'future release. Pass Settings instead, e.g. '
            f'Settings.from_env({config!r}).',
            DeprecationWarning,
            stacklevel=3
        )
        return Settings.from_env(config)

    raise TypeError(
        'Expected Settings, e.g. Settings.from_env(".env") or '
        f'Settings(openai_api_key=...), got {type(config).__name__}.'
    )
