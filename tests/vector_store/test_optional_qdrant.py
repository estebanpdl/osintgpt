'''Core installs work without Qdrant; optional imports retain their contracts.'''

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
BLOCK_QDRANT = '''
import importlib.abc
import sys

class BlockQdrant(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qdrant_client' or fullname.startswith('qdrant_client.'):
            raise ModuleNotFoundError('blocked optional client', name='qdrant_client')

sys.meta_path.insert(0, BlockQdrant())
'''


class TestOptionalQdrant(unittest.TestCase):
    def run_source(self, source, *, prefix=BLOCK_QDRANT):
        result = subprocess.run(
            [sys.executable, '-c', prefix + textwrap.dedent(source)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_core_import_and_default_sqlite_workflow(self):
        self.run_source('''
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import osintgpt
            from osintgpt.vector_store import BACKENDS, StoredChunk, store_for

            assert 'qdrant' in BACKENDS
            assert 'osintgpt.vector_store.qdrant_store' not in sys.modules
            assert 'osintgpt.vector_store.qdrant' not in sys.modules
            assert 'osintgpt.vector_store.connection' not in sys.modules
            with TemporaryDirectory() as home:
                project = osintgpt.Project.create('Case', home=Path(home))
                chunk = StoredChunk(ref='source', sequence=0, text='evidence',
                                    embedding_model='synthetic')
                with store_for(project) as store:
                    assert store.upsert('source', [chunk], [[1.0, 0.0]]) == 1
                    assert store.search([1.0, 0.0], 'synthetic')[0].chunk == chunk
                with store_for(project) as store:
                    assert store.chunks_for('source') == [chunk]
                    assert store.delete(['source']) == 1
                    assert store.count() == 0
            assert 'qdrant_client' not in sys.modules
        ''')

    def test_every_cli_command_offers_help_without_qdrant(self):
        self.run_source('''
            from pathlib import Path
            from tempfile import TemporaryDirectory
            import click
            from typer.main import get_command
            from typer.testing import CliRunner
            from osintgpt.cli import app

            def command_paths(command, prefix=()):
                yield prefix
                if isinstance(command, click.Group):
                    for name, child in command.commands.items():
                        yield from command_paths(child, (*prefix, name))

            runner = CliRunner()
            with TemporaryDirectory() as home:
                for path in command_paths(get_command(app)):
                    result = runner.invoke(app, ['--home', home, *path, '--help'])
                    assert result.exit_code == 0, (path, result.output, result.exception)
                    assert 'Usage' in result.output, (path, result.output)
            assert 'qdrant_client' not in sys.modules
        ''')

    def test_missing_extra_has_install_guidance_at_each_import_boundary(self):
        self.run_source('''
            import osintgpt.vector_store as stores

            statements = [
                'from osintgpt.vector_store import QdrantVectorStore',
                'from osintgpt.vector_store import Qdrant',
                'from osintgpt.vector_store import DEFAULT_COLLECTION',
                'from osintgpt.vector_store.qdrant_store import QdrantVectorStore',
                'from osintgpt.vector_store.qdrant import Qdrant',
                'from osintgpt.vector_store.connection import connect',
            ]
            for statement in statements:
                try:
                    exec(statement, {})
                except ImportError as error:
                    assert 'qdrant-client' in str(error), statement
                    assert 'pip install "osintgpt[qdrant]"' in str(error), statement
                    assert error.__suppress_context__, statement
                else:
                    raise AssertionError(statement)
            assert not hasattr(stores, 'unknown_backend')
        ''')

    def test_factory_reports_missing_extra_before_connecting(self):
        self.run_source('''
            from types import SimpleNamespace
            from osintgpt import Settings
            from osintgpt.vector_store import store_for

            project = SimpleNamespace(slug='case', settings=SimpleNamespace(
                storage_backend='  Qdrant '))
            try:
                store_for(project, Settings())
            except ImportError as error:
                assert 'pip install "osintgpt[qdrant]"' in str(error)
            else:
                raise AssertionError('missing client was accepted')
        ''')

    def test_cli_backend_error_is_actionable_without_traceback(self):
        self.run_source('''
            import json
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from types import SimpleNamespace
            from unittest.mock import patch
            from typer.testing import CliRunner
            from osintgpt import Settings
            from osintgpt.cli import app
            from osintgpt.cli import corpus
            from osintgpt.projects import Registry

            runner = CliRunner()
            with TemporaryDirectory() as home:
                result = runner.invoke(app, ['--home', home, 'project', 'create', 'Case'])
                assert result.exit_code == 0, result.output
                project = Registry.load(Path(home)).open('case')
                project.with_settings(storage_backend='qdrant').save()
                with patch.object(corpus, 'resolve_credentials', return_value=Settings()), \\
                     patch.object(corpus, 'build_embedding_provider',
                                  return_value=SimpleNamespace(model='synthetic')):
                    for flags in ([], ['--json']):
                        result = runner.invoke(app, [
                            '--home', home, 'index', '--project', 'case', *flags])
                        assert result.exit_code == 1, (result.output, result.exception)
                        message = json.loads(result.output)['error'] if flags else result.output
                        assert 'pip install "osintgpt[qdrant]"' in message, result.output
                        assert 'Traceback' not in result.output
                        if flags:
                            assert 'qdrant-client' in json.loads(result.output)['error']
        ''')

    def test_public_names_and_factory_resolve_with_available_client(self):
        self.run_source('''
            import sys
            from types import ModuleType, SimpleNamespace
            from unittest.mock import Mock
            import osintgpt.vector_store as stores
            from osintgpt import Settings

            assert 'osintgpt.vector_store.qdrant_store' not in sys.modules
            assert 'osintgpt.vector_store.qdrant' not in sys.modules
            client = Mock()
            package = ModuleType('qdrant_client')
            package.__path__ = []
            package.QdrantClient = Mock(return_value=client)
            http = ModuleType('qdrant_client.http')
            http.__path__ = []
            http.models = ModuleType('qdrant_client.http.models')
            sys.modules.update({
                'qdrant_client': package,
                'qdrant_client.http': http,
                'qdrant_client.http.models': http.models,
            })

            from osintgpt.vector_store import QdrantVectorStore, DEFAULT_COLLECTION
            from osintgpt.vector_store.qdrant_store import QdrantVectorStore as modern
            assert QdrantVectorStore is modern
            assert DEFAULT_COLLECTION == 'osintgpt'
            assert 'osintgpt.vector_store.qdrant' not in sys.modules
            package.QdrantClient.assert_not_called()

            from osintgpt.vector_store import Qdrant
            from osintgpt.vector_store.qdrant import Qdrant as legacy
            assert Qdrant is legacy
            exports = {}
            exec('from osintgpt.vector_store import *', exports)
            assert exports['Qdrant'] is legacy
            assert exports['QdrantVectorStore'] is modern

            config = Settings(qdrant_host='localhost', qdrant_port=6333)
            project = SimpleNamespace(slug='case', settings=SimpleNamespace(
                storage_backend='  Qdrant '))
            engine = stores.store_for(project, config)
            assert isinstance(engine, modern)
            assert engine.collection == 'case'
            assert engine.settings is config
            assert engine.client is client
            package.QdrantClient.assert_called_once_with(
                host='localhost', port=6333, timeout=60)
            assert legacy(config).get_client() is client
        ''', prefix='')

    def test_unrelated_missing_dependency_is_not_relabelled(self):
        self.run_source('''
            import importlib.abc
            import sys

            class BrokenClient(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'qdrant_client':
                        raise ModuleNotFoundError('broken dependency', name='grpc')

            sys.meta_path.insert(0, BrokenClient())
            try:
                from osintgpt.vector_store import Qdrant
            except ModuleNotFoundError as error:
                assert error.name == 'grpc'
                assert 'osintgpt[qdrant]' not in str(error)
            else:
                raise AssertionError('broken client was accepted')
        ''', prefix='')


if __name__ == '__main__':
    unittest.main(verbosity=2)
