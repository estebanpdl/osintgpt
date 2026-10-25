'''Offline checks executed outside the source tree by wheel_smoke.py.'''

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory


def main():
    optional = ['qdrant_client', 'psycopg', 'pgvector', 'sentence_transformers', 'torch']
    for name in optional:
        assert importlib.util.find_spec(name) is None, f'{name} must be absent for core-only proof'
    assert getattr(socket, '_osintgpt_offline', False), 'offline guard was not loaded'

    import osintgpt
    import osintgpt.app.main
    from osintgpt.databases import SQLDatabaseManager
    from osintgpt.embeddings import OpenAIEmbeddingGenerator
    from osintgpt.llms import OpenAIGPT
    from osintgpt.semantic_operations import SemanticOperations
    from osintgpt.prompts.templates import available, prompt, variables_of
    from osintgpt.vector_store import StoredChunk, store_for

    prefix = Path(sys.prefix).resolve()
    origin = Path(osintgpt.__file__).resolve()
    assert origin.is_relative_to(prefix), (origin, prefix)
    dist = importlib.metadata.distribution('osintgpt')
    assert Path(dist.locate_file('')).resolve().is_relative_to(prefix)
    assert osintgpt.__version__ == dist.version
    for name in osintgpt.__all__:
        assert getattr(osintgpt, name) is not None
    assert all([SQLDatabaseManager, OpenAIEmbeddingGenerator, OpenAIGPT, SemanticOperations])
    for name in available():
        assert prompt(name, **{variable: 'smoke' for variable in variables_of(name)}), name
    assert len(available()) == 10
    for name in ['osintgpt.png', 'graph_layout.mjs', 'graph_downloads.mjs',
                 'graph_diagram.mjs', 'graph_diagram.css']:
        assert (origin.parent / 'app' / 'assets' / name).stat().st_size > 0

    with TemporaryDirectory(dir=Path.cwd()) as directory:
        project = osintgpt.Project.create('Wheel case', home=Path(directory))
        chunk = StoredChunk(ref='source', sequence=0, text='installed evidence', embedding_model='synthetic')
        with store_for(project) as store:
            assert store.upsert('source', [chunk], [[1.0, 0.0]]) == 1
            assert store.search([1.0, 0.0], 'synthetic')[0].chunk == chunk
        with store_for(project) as store:
            assert store.chunks_for('source') == [chunk]
            assert store.delete(['source']) == 1
            assert store.count() == 0
        for backend in ['qdrant', 'postgres']:
            try:
                store_for(project.with_settings(storage_backend=backend), osintgpt.Settings())
            except ImportError as error:
                assert f'osintgpt[{backend}]' in str(error)
            else:
                raise AssertionError(f'{backend} unexpectedly available')
        from osintgpt.llm import build_embedding_provider
        try:
            build_embedding_provider('sentence-transformers', osintgpt.Settings())
        except ImportError as error:
            assert 'osintgpt[local]' in str(error)
        else:
            raise AssertionError('local encoder unexpectedly available')
    assert not set(optional).intersection(sys.modules)
    print(json.dumps({'python': sys.executable, 'origin': str(origin), 'version': dist.version,
                      'absent_extras': optional, 'prompts': available(), 'app_assets': 5,
                      'legacy_imports': 'passed', 'sqlite_persistence': 'passed',
                      'missing_extra_guidance': 'passed'}))


if __name__ == '__main__':
    main()
