# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: conftest.py
# Description: Shared fixtures. Provider clients are stubbed rather than called,
#   so the suite never reaches the network and never needs a key.
# =================================================================================

# import modules
import os
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import ENV_VARS, Settings

# A key-shaped string that is not a key. Constructing an OpenAI client requires
# one; nothing here ever sends it anywhere.
FAKE_KEY = 'sk-test-not-a-real-key'


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    '''
    Clear every variable osintgpt reads.

    Without this the developer's own shell decides what the tests see, and a
    suite that passes on one machine fails on another for reasons nobody can
    reproduce.
    '''
    for name in ENV_VARS.values():
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path):
    '''Settings sufficient for the OpenAI and SQLite paths.'''
    return Settings(
        openai_api_key=FAKE_KEY,
        openai_gpt_model='gpt-4o',
        sql_db_file_path=str(tmp_path / 'conversations.db')
    )


@pytest.fixture
def env_file(tmp_path):
    '''
    A .env file carrying a full configuration.

    Returns the path as a string — the deprecated calling convention takes one,
    and `Settings.from_env` is tested against it.
    '''
    path = tmp_path / '.env'
    path.write_text(
        f'OPENAI_API_KEY={FAKE_KEY}\n'
        'OPENAI_GPT_MODEL=gpt-4o\n'
        'QDRANT_HOST=localhost\n'
        'QDRANT_PORT=6333\n'
        f'SQL_DB_FILE_PATH={(tmp_path / "from_env.db").as_posix()}\n',
        encoding='utf-8'
    )

    return str(path)


class StubEmbeddings:
    '''Records the models it was asked for; returns deterministic vectors.'''

    def __init__(self):
        self.models = []
        self.batches = []

    def create(self, *, model, input):
        self.models.append(model)
        self.batches.append(len(input))

        return SimpleNamespace(
            model=model,
            data=[
                SimpleNamespace(index=i, embedding=[float(i), 0.2, 0.3])
                for i in range(len(input))
            ]
        )


class StubUsage:
    def model_dump(self, exclude_none=False):
        return {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}


class StubCompletions:
    '''Records every request; replies with a fixed message.'''

    REPLY = 'STUB REPLY'

    def __init__(self):
        self.calls = []

    def create(self, *, model, messages, **kwargs):
        self.calls.append({'model': model, 'messages': messages, **kwargs})

        return SimpleNamespace(
            id='chatcmpl-stub',
            created=1_700_000_000,
            model=model,
            usage=StubUsage(),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role='assistant', content=self.REPLY
                    )
                )
            ]
        )


class StubOpenAI:
    '''Stands in for openai.OpenAI across both call surfaces.'''

    def __init__(self):
        self.embeddings = StubEmbeddings()
        self.chat = SimpleNamespace(completions=StubCompletions())


@pytest.fixture
def stub_client():
    return StubOpenAI()
