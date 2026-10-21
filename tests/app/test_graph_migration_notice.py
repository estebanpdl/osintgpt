'''Inspection names the required migration and never upgrades an operator file implicitly.'''

import sqlite3

import pytest

from osintgpt.app.session import VIEW
from test_app_renders import home, project, run
from test_graph_view import published_graph


@pytest.mark.parametrize('view', ['Graph', 'Settings'])
def test_schema_seven_notice_does_not_modify_the_graph(published_graph, view, monkeypatch):
    project, _ = published_graph
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=7')
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('Viewing an old graph must not construct providers.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: view})
    assert not app.exception
    messages = [item.value for group in (app.error, app.warning, app.info, app.caption) for item in group]
    assert any('schema version 7' in value and 'current version is 8' in value and 'migrate --dry-run' in value
               for value in messages)
    assert path.read_bytes() == before and not list(path.parent.glob('graph.backup-*'))
