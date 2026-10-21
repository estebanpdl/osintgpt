'''Graph tab notices separate coverage from search outcomes and live build activity.'''

import json

import pytest

from osintgpt.app.session import VIEW
from osintgpt.graph import Entity, graph_for
from osintgpt.graph.accounting import BuildCounts
from osintgpt.graph.state import build_lock
from test_app_renders import home, project, run
from test_graph_view import published_graph, button, widget, choose_start


def test_missing_and_isolated_identities_are_not_described_as_running_builds(published_graph, monkeypatch):
    project, _ = published_graph
    with graph_for(project) as graph:
        graph.add([Entity('separate', 'Separate identity')], [])
        graph.state.finish('complete_with_gaps')
        with graph.connection:
            graph.connection.execute('UPDATE graph_build SET counts=? WHERE id=1',
                                     (json.dumps({'documents_failed': 1}),))
    def forbidden(*args, **kwargs):
        pytest.fail('Inspection must not construct providers.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    assert any(item.value == 'The last graph build finished with gaps.' for item in app.caption)
    assert any('coverage may be incomplete' in item.value for item in app.warning)
    assert any('unknown documents completed; 1 failed' in item.value for item in app.caption)
    assert any('Window progress has not been recorded' in item.value for item in app.caption)
    widget(app.text_input, 'Starting entity').set_value('Absent').run()
    app.button(key=f'graph-explore:{project.id}:start:find').click().run()
    assert any(item.value == 'No matching identity was found in the published graph.' for item in app.info)
    choose_start(app, project, 'Separate identity', 'separate')
    button(app, 'Explore neighborhood').click().run()
    assert not app.exception
    assert any(item.value == 'This identity has no accepted relationships in the published graph.' for item in app.info)
    messages = [item.value for group in (app.info, app.caption, app.warning) for item in group]
    assert not any('The build is incomplete' in item or 'build is running' in item for item in messages)
    assert not any('This identity has no accepted relationships' in item and 'Quotations verify' in item for item in messages)


def test_running_and_abandoned_builds_use_checked_activity(published_graph):
    project, _ = published_graph
    with build_lock(project.paths.root / 'graph.sqlite'):
        with graph_for(project) as graph:
            graph.state.begin(BuildCounts())
        app = run(project, **{VIEW: 'Graph'})
        assert not app.exception
        assert any(item.value == 'A graph build is running.' for item in app.caption)
        assert not any('The last graph build was interrupted' in item.value for item in app.caption)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    assert any(item.value == 'The last graph build was interrupted.' for item in app.caption)
    assert not any('build is running' in item.value for item in app.caption)


def test_stale_and_disabled_graph_has_independent_notices(published_graph):
    project, _ = published_graph
    (project.paths.root / 'funding.md').write_text('Changed source.', encoding='utf-8')
    project = project.with_settings(graph_enabled=False)
    project.save()
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    assert any('Updating it requires an explicit build' in item.value for item in app.warning)
    assert any('Graph retrieval is disabled; any published graph remains available' in item.value for item in app.caption)
    assert not any('until their replacements finish processing' in item.value for item in app.warning)
    choose_start(app, project)
    button(app, 'Explore neighborhood').click().run()
    assert not app.exception
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 2  # Direct connections by default.
