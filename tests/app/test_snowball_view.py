'''Real Streamlit renders with a fake embedder and similarity graph.'''

import importlib

import pytest

from osintgpt import Project
from osintgpt.app.session import VIEW, SELECTED
from osintgpt.config import Settings
from tests.snowball_fakes import GraphEmbedder, GraphStore
from test_app_renders import home, project, run

operation = importlib.import_module('osintgpt.agentic.snowball')


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def test_page_renders_without_building_any_provider(project, monkeypatch):
    monkeypatch.setattr(operation, 'build_embedding_provider', lambda *a, **k: pytest.fail('provider built'))
    app = run(project, **{VIEW: 'Snowball'})
    assert not app.exception
    assert app.sidebar.radio[0].options == ['Projects', 'Material', 'Ask', 'Graph', 'Snowball', 'Settings']
    assert widget(app.number_input, 'Depth').value == 25
    assert widget(app.selectbox, 'Deduplicate by').value == 'text'
    assert widget(app.number_input, 'Similarity threshold').value == .5
    assert any('causation' in item.value for item in app.info)


def test_run_group_and_download_reuse_results_and_project_configuration(project, home, monkeypatch):
    project = project.with_settings(embedding_model=GraphEmbedder.model, storage_backend='postgres')
    project.save()
    synthetic = Settings(postgres_dsn='postgresql://synthetic-only')
    monkeypatch.setattr('osintgpt.app.session.resolve_credentials', lambda *a: synthetic)
    calls, stores = [], []

    def build(provider, config, model=None):
        assert model == GraphEmbedder.model
        assert config.postgres_dsn == synthetic.postgres_dsn
        calls.append(model)
        return GraphEmbedder()

    def open_store(selected, config):
        assert selected.id == project.id
        assert config.postgres_dsn == synthetic.postgres_dsn
        store = GraphStore(4, 12)
        stores.append(store)
        return store

    monkeypatch.setattr(operation, 'build_embedding_provider', build)
    monkeypatch.setattr(operation, 'store_for', open_store)
    app = run(project, **{VIEW: 'Snowball'})
    widget(app.text_area, 'Initial query').set_value('start')
    widget(app.number_input, 'Depth').set_value(4)
    widget(app.button, 'Run Snowball').click().run()
    assert not app.exception, app.exception
    result = app.session_state[f'snowball:{project.id}:result']
    assert len(result) == 4 and len(result.hops[0].identical_text_refs) == 13
    assert stores[0].closed and calls == [GraphEmbedder.model]
    assert any('estimated cost' in item.value for item in app.caption)
    assert any('Similarity to query' in item.value for item in app.caption)
    widget(app.selectbox, 'Group by period').set_value('month').run()
    assert any(item.value == 'Unknown date' for item in app.subheader)
    widget(app.button, 'Prepare downloads').click().run()
    assert len(app.download_button) == 2 and not app.exception
    ready = app.session_state[f'snowball:{project.id}:downloads']
    assert 'similarity_to_query' in ready['csv'] and ready['xlsx'][:2] == b'PK'
    app.download_button(key=f'snowball:{project.id}:csv').click().run()
    assert len(calls) == 1
    widget(app.selectbox, 'Group by period').set_value('year').run()
    assert not app.download_button and len(calls) == 1
    widget(app.text_area, 'Initial query').set_value('   ')
    widget(app.button, 'Run Snowball').click().run()
    assert any('nonempty' in item.value for item in app.error)
    assert not app.download_button and len(calls) == 1


def test_switching_project_does_not_show_another_projects_result(project, home):
    other = Project.create('Case Beta', home=home)
    app = run(project, **{VIEW: 'Snowball'})
    widget(app.selectbox, 'Project').set_value(other.slug).run()
    assert not app.exception
    assert app.session_state[SELECTED] == other.slug
    assert widget(app.selectbox, 'Project').value == other.slug
    assert not app.download_button
