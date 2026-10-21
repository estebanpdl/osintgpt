'''Progressive controls keep query, page, diagram, and evidence selections distinct.'''

import json

from osintgpt.app.session import VIEW
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.ingestion import Corpus
from osintgpt.app.views import graph_chart
from test_app_renders import home, project, run
from test_graph_view import button, widget, choose_start


def build(project, count=155, chain=False):
    project = project.with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'fixture.md').write_text('\n'.join(f'Quoted statement {i}.' for i in range(count)), encoding='utf8')
    Corpus.load(project.paths.sources).register('fixture.md')
    names = ['Alpha', *[f'Identity {i}' for i in range(1, count + 1)]]
    ids = ['alpha', *[f'node-{i}' for i in range(1, count + 1)]]
    with graph_for(project) as graph:
        graph.add([Entity(key, name) for key, name in zip(ids, names)], [
            Edge(names[i if chain else 0], names[i + 1], 'links', 'fixture.md', f'Quoted statement {i}.',
                 source_id=ids[i if chain else 0], target_id=ids[i + 1]) for i in range(count)])
        graph.connection.execute("UPDATE graph_build SET started_at='2026-01-01' WHERE id=1")
        graph.connection.commit()
    return project


def complete(app, project):
    for _ in range(20):
        assert not app.exception
        if app.session_state[f'graph-explore:{project.id}:inspection'].state != 'running':
            return
        app.run()
    raise AssertionError('Small fixture did not complete within 20 test-driven ticks.')


def test_pause_continue_pages_and_independent_diagram_selection(project, monkeypatch):
    project = build(project)
    captured = []
    original = graph_chart.render
    def chart(st, claims, *args, **kwargs):
        captured.append([row['assertion_id'] for row in claims])
        return original(st, claims, *args, **kwargs)
    monkeypatch.setattr(graph_chart, 'render', chart)
    def forbidden(*args, **kwargs):
        raise AssertionError('Exploration cannot construct a provider.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    choose_start(app, project)
    assert widget(app.number_input, 'Traversal depth').value == 1
    assert not any(item.label == 'Maximum assertions' for item in app.number_input)
    assert not any(item.label == 'Expand one more step' for item in app.button)
    button(app, 'Explore neighborhood').click().run()
    query = app.session_state[f'graph-explore:{project.id}:inspection']
    assert query.state == 'running' and 0 < len(query.rows) < 155
    button(app, 'Stop exploration').click().run()
    assert query.state == 'paused'
    count = len(query.rows); app.run()
    assert len(query.rows) == count
    app.button(key=f'graph-export:{project.id}:selection:prepare').click().run()
    package = json.loads(app.session_state[f'graph-export:{project.id}:selection:ready']['json'])
    assert package['metadata']['truncated'] and package['metadata']['query']['progress']['state'] == 'paused'
    button(app, 'Continue exploration').click().run(); complete(app, project)
    assert len(query.rows) == 155 and query.state == 'complete'
    assert len(captured[-1]) == 30
    first = next(frame.value for frame in app.dataframe if 'Assertion ID' in frame.value.columns)
    assert len(first) == 30
    assert widget(app.selectbox, 'Diagram selection').value == 'Current table page'
    button(app, 'Next page').click().run()
    second = next(frame.value for frame in app.dataframe if 'Assertion ID' in frame.value.columns)
    assert set(first['Assertion ID']).isdisjoint(second['Assertion ID'])
    assert captured[-1] == list(second['Assertion ID'])
    widget(app.selectbox, 'Diagram selection').select('First matching assertions').run()
    assert captured[-1] == list(first['Assertion ID'])
    widget(app.selectbox, 'Diagram selection').select('Current table page').run()
    assert captured[-1] == list(second['Assertion ID'])
    widget(app.selectbox, 'Diagram selection').select('Choose assertions').run()
    widget(app.multiselect, 'Assertions from this table page').select(second['Assertion ID'].iloc[0]).run()
    assert captured[-1] == [second['Assertion ID'].iloc[0]]
    button(app, 'Previous page').click().run()
    widget(app.multiselect, 'Assertions from this table page').select(first['Assertion ID'].iloc[0]).run()
    assert set(captured[-1]) == {first['Assertion ID'].iloc[0], second['Assertion ID'].iloc[0]}
    widget(app.text_input, 'Search relationships').set_value('no matching evidence').run()
    assert not app.get('bidi_component')
    widget(app.text_input, 'Search relationships').set_value('').run()
    assert set(captured[-1]) == {first['Assertion ID'].iloc[0], second['Assertion ID'].iloc[0]}
    widget(app.selectbox, 'Diagram selection').select('Hidden').run()
    assert not app.get('bidi_component')
    widget(app.selectbox, 'Inspect assertion').select(first['Assertion ID'].iloc[1]).run()
    assert not app.exception
    widget(app.selectbox, 'Diagram selection').select('First matching assertions').run()
    widget(app.number_input, 'Assertions in diagram').set_value(75).run()
    assert len(captured[-1]) == 75  # No renderer-side 60-assertion slice.
    widget(app.number_input, 'Table page').set_value(6).run()
    last = next(frame.value for frame in app.dataframe if 'Assertion ID' in frame.value.columns)
    assert len(last) == 5 and len(captured[-1]) == 75
    app.button(key=f'graph-export:{project.id}:selection:prepare').click().run()
    package = json.loads(app.session_state[f'graph-export:{project.id}:selection:ready']['json'])
    assert len(package['edges']) == 155 and not package['metadata']['truncated']
    assert not app.exception
    button(app, 'Refresh graph status').click().run()
    assert f'graph-explore:{project.id}:inspection' not in app.session_state
    assert query._graph is None and not app.get('bidi_component')


def test_depth_ten_expansion_direct_reset_and_changed_graph(project):
    project = build(project, 12, chain=True)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    choose_start(app, project)
    button(app, 'Direct connections').click().run(); complete(app, project)
    query = app.session_state[f'graph-explore:{project.id}:inspection']
    assert len(query.rows) == 1
    widget(app.number_input, 'Traversal depth').set_value(2).run()
    assert query.cursor.depth == 1  # Editing the value alone does not run a query.
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    assert app.session_state[f'graph-explore:{project.id}:inspection'] is query
    assert query.cursor.depth == 2 and len(query.rows) == 2
    widget(app.number_input, 'Traversal depth').set_value(10).run()
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    assert app.session_state[f'graph-explore:{project.id}:inspection'] is query
    assert len(query.rows) == 10 and query.cursor.depth == 10
    widget(app.number_input, 'Traversal depth').set_value(2).run()
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    smaller = app.session_state[f'graph-explore:{project.id}:inspection']
    assert smaller is not query and len(smaller.rows) == 2
    query = smaller
    with graph_for(project) as graph:
        graph.connection.execute("UPDATE edges SET relation='updated' WHERE id=1")
        graph.connection.commit()
    app.run()
    assert query.state == 'invalidated'
    assert any('graph changed' in item.value.lower() for item in app.warning)
    assert not app.get('bidi_component')
    button(app, 'Direct connections').click().run(); complete(app, project)
    assert widget(app.number_input, 'Traversal depth').value == 1
    assert len(app.session_state[f'graph-explore:{project.id}:inspection'].rows) == 1
    assert not app.exception
