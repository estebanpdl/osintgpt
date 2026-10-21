'''App exports disclose their precise page/diagram/filter scope and stale checks.'''

import json
import os

from osintgpt.app.session import VIEW
from test_app_renders import home, project, run
from test_graph_progressive import build, complete
from test_graph_view import button, widget, choose_start


def test_export_scopes_follow_pages_diagram_and_full_filters(project, monkeypatch):
    project = build(project, 75)
    def forbidden(*args, **kwargs):
        raise AssertionError('Operator inspection must not call providers.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    choose_start(app, project)
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    export_key = f'graph-export:{project.id}:selection:'
    def prepare():
        app.button(key=export_key + 'prepare').click().run()
        assert not app.exception
        return json.loads(app.session_state[export_key + 'ready']['json'])
    all_matches = prepare()
    assert len(all_matches['edges']) == 75 and all_matches['metadata']['scope'] == 'all_matching'
    button(app, 'Next page').click().run()
    assert app.download_button(key=export_key + 'json')  # All-matches export survives page changes.
    widget(app.radio, 'Export scope').set_value('Current table page').run()
    assert not any(item.key == export_key + 'json' for item in app.download_button)
    page = prepare()
    table = next(frame.value for frame in app.dataframe if 'Assertion ID' in frame.value.columns)
    assert [row['assertion_id'] for row in page['edges']] == list(table['Assertion ID'])
    assert page['metadata']['selection']['display'] == {'table_page': 2, 'rows_per_page': 30}
    button(app, 'Next page').click().run()
    assert not any(item.key == export_key + 'json' for item in app.download_button)
    assert len(prepare()['edges']) == 15
    widget(app.selectbox, 'Diagram selection').select('First matching assertions').run()
    widget(app.number_input, 'Assertions in diagram').set_value(10).run()
    widget(app.radio, 'Export scope').set_value('Displayed diagram').run()
    diagram = prepare()
    assert len(diagram['edges']) == 10
    assert diagram['metadata']['selection']['display'] == {'diagram_selection': 'First matching assertions'}
    widget(app.selectbox, 'Diagram selection').select('Hidden').run()
    assert not any(item.key == export_key + 'json' for item in app.download_button)
    assert prepare()['edges'] == []
    widget(app.text_input, 'Search relationships').set_value('statement 74.').run()
    widget(app.radio, 'Export scope').set_value('All matching results').run()
    assert [row['evidence'] for row in prepare()['edges']] == ['Quoted statement 74.']
    widget(app.text_input, 'Search relationships').set_value('does not occur').run()
    empty = prepare()
    assert empty['edges'] == [] and empty['metadata']['selection']['matching_total'] == 0
    assert any('75 discovered assertions checked' in row.value for row in app.caption)
    app.session_state[f'graph-explore:{project.id}:inspection'].close()


def test_source_edit_hides_prepared_exports_and_old_evidence(project):
    project = build(project, 3)
    app = run(project, **{VIEW: 'Graph'})
    choose_start(app, project)
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    export_key = f'graph-export:{project.id}:selection:'
    app.button(key=export_key + 'prepare').click().run()
    previous = app.session_state[export_key + 'ready']['json']
    path = project.paths.root / 'fixture.md'
    stat = path.stat()
    path.write_text(path.read_text(encoding='utf8').replace('Quoted', 'Edited'), encoding='utf8')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    app.run()
    assert not app.exception
    assert any('Source content or availability changed' in row.value for row in app.warning)
    assert not any(item.key == export_key + 'json' for item in app.download_button)
    assert not any('Quotation preview' in frame.value.columns for frame in app.dataframe)
    assert app.session_state[export_key + 'ready']['json'] == previous
    button(app, 'Explore neighborhood').click().run(); complete(app, project)
    assert not app.exception
    widget(app.selectbox, 'Current quotation verification').select('not_found').run()
    app.button(key=export_key + 'prepare').click().run()
    current = json.loads(app.session_state[export_key + 'ready']['json'])
    assert len(current['edges']) == 3
    assert all(row['verification']['current'] == 'not_found' for row in current['edges'])
    app.session_state[f'graph-explore:{project.id}:inspection'].close()
