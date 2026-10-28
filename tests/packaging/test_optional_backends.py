'''Absent-extra simulations and injected backends; no service or model downloads.'''

import builtins
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from osintgpt import Settings
from osintgpt.llm import build_embedding_provider
from osintgpt.vector_store import store_for


OPTIONAL = {'qdrant_client', 'psycopg', 'pgvector', 'sentence_transformers', 'torch'}


def test_core_exports_and_app_without_any_runtime_extra():
    source = '''
import importlib.abc
import sys
class NoExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in OPTIONAL:
            raise ModuleNotFoundError('simulated missing extra', name=fullname)
sys.meta_path.insert(0, NoExtras())
import osintgpt
import osintgpt.cli
import osintgpt.app.main
from osintgpt.llm import build_embedding_provider, build_generation_provider
from osintgpt.llm.local import SentenceTransformerEmbedding
from osintgpt.vector_store.pgvector_store import PgVectorStore
for name in osintgpt.__all__:
    assert getattr(osintgpt, name) is not None, name
assert not OPTIONAL.intersection(sys.modules)
'''
    result = subprocess.run(
        [sys.executable, '-c', f'OPTIONAL = {OPTIONAL!r}\n' + source],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('missing,extra', [
    ('psycopg', 'postgres'), ('pgvector', 'postgres'), ('sentence_transformers', 'local'),
])
def test_missing_extra_is_reported_by_the_selected_factory(monkeypatch, missing, extra):
    original = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.split('.')[0] == missing:
            raise ModuleNotFoundError('simulated missing extra', name=missing)
        return original(name, *args, **kwargs)

    if missing == 'pgvector':
        monkeypatch.setitem(sys.modules, 'psycopg', ModuleType('psycopg'))
    monkeypatch.setattr(builtins, '__import__', refuse)
    with pytest.raises(ImportError, match=rf'osintgpt\[{extra}\]'):
        if extra == 'local':
            build_embedding_provider('sentence-transformers', Settings())
        else:
            project = SimpleNamespace(slug='case', settings=SimpleNamespace(storage_backend='postgres'))
            store_for(project, Settings())


def test_postgres_factory_uses_available_driver_modules(monkeypatch):
    from osintgpt.vector_store.pgvector_store import PgVectorStore

    psycopg = ModuleType('psycopg')
    connection = Mock()
    psycopg.connect = Mock(return_value=connection)
    pgvector = ModuleType('pgvector')
    pgvector.__path__ = []
    driver = ModuleType('pgvector.psycopg')
    driver.register_vector = Mock()
    for name, module in [('psycopg', psycopg), ('pgvector', pgvector), ('pgvector.psycopg', driver)]:
        monkeypatch.setitem(sys.modules, name, module)
    # Schema operations need a real server; this probe stops at the driver boundary.
    prepare = Mock()
    monkeypatch.setattr(PgVectorStore, '_prepare', prepare)
    settings = Settings(postgres_dsn='postgresql://offline.invalid/test')
    project = SimpleNamespace(slug='case', settings=SimpleNamespace(storage_backend='postgres'))
    with store_for(project, settings) as store:
        assert isinstance(store, PgVectorStore)
        assert store.connection is connection
        assert store.register_vector is driver.register_vector
        assert store.collection == 'case'
    psycopg.connect.assert_called_once_with(settings.postgres_dsn)
    prepare.assert_called_once()
    connection.close.assert_called_once()


def test_local_factory_uses_available_encoder_module(monkeypatch):
    from osintgpt.llm.local import SentenceTransformerEmbedding

    module = ModuleType('sentence_transformers')
    encoder = Mock()
    encoder.encode.return_value = [[0.25, 0.75]]
    module.SentenceTransformer = Mock(return_value=encoder)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', module)
    provider = build_embedding_provider('sentence-transformers', Settings(), model='offline-model')
    assert isinstance(provider, SentenceTransformerEmbedding)
    assert provider.embed(['evidence']) == [[0.25, 0.75]]
    module.SentenceTransformer.assert_called_once_with('offline-model', device=None)
    encoder.encode.assert_called_once_with(['evidence'])
