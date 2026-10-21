'''Operators can inspect saved recovery work without constructing a provider.'''

import pytest

from osintgpt.app.session import VIEW
from osintgpt.graph import build_graph
from osintgpt.ingestion import Corpus
from test_app_renders import home, project, run


def test_graph_shows_recovery_details_without_duplicating_them_in_settings(project, monkeypatch):
    project = project.with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'long.md').write_text('text ' * 15000)
    Corpus.load(project.paths.sources).register('long.md')
    class Generator:
        def __init__(self, interrupt=False):
            self.calls, self.interrupt = 0, interrupt
        def generate(self, *args):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise KeyboardInterrupt()
            return '{"entities": [], "edges": []}'
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(interrupt=True))
    def forbidden(*args, **kwargs):
        raise AssertionError('Reading graph recovery must be provider-free')
    monkeypatch.setattr('osintgpt.app.session.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Settings'})
    assert not app.exception
    assert 'Source refresh details' not in [item.label for item in app.expander]
    assert 'Published extraction models' not in [item.label for item in app.expander]
    assert any('Relationship graph:' in item.value for item in app.caption)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    assert '1 unfinished sources' in captions and '1 windows saved' in captions
    assert 'Source refresh details' in [item.label for item in app.expander]
    build_graph(project, Generator())
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    assert '1 saved windows reused' in captions and 'ten previous runs' in captions


def test_failed_generation_diagnostics_are_available_without_published_graph(project, monkeypatch):
    project = project.with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'bad.md').write_text('Original source.')
    Corpus.load(project.paths.sources).register('bad.md')
    class Generator:
        def generate(self, *args):
            return 'Synthetic malformed response'
    build_graph(project, Generator())
    def forbidden(*args, **kwargs):
        pytest.fail('Diagnostics must not construct providers')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    assert 'Generation diagnostics' in [item.label for item in app.expander]
    next(item for item in app.button if item.label == 'Inspect generation responses').click().run()
    assert not app.exception
    assert all('Synthetic malformed response' not in str(item.value) for item in app.json)
