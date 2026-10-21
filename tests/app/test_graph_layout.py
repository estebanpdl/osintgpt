'''Layout scopes and selection stay local without widening traversal or changing claims.'''

from copy import deepcopy
from types import SimpleNamespace

from osintgpt.app.views import graph_chart


def claims():
    return [dict(assertion_id=key, source_id=source, target_id=target, source='同名', target='同名',
                 relation='relación', ref='synthetic.md', relation_polarity=polarity, assertion='alleged')
            for key, source, target, polarity in [('one', 'a', 'b', 'positive'), ('two', 'b', 'c', 'negative')]]


def test_filters_keep_scope_and_hidden_positions_but_new_query_resets():
    rows = claims()
    original = deepcopy(rows)
    all_data = graph_chart.diagram_data(rows, starting_id='a', scope=['project', 'query'])
    filtered = graph_chart.diagram_data(rows[1:], all_claims=rows, starting_id='a', scope=['project', 'query'])
    assert all_data['scope'] == filtered['scope']
    assert filtered['all_ids'] == ['a', 'b', 'c']
    assert [node['id'] for node in filtered['nodes']] == ['b', 'c']
    assert filtered['starting_id'] == 'a'  # Do not highlight a namesake when the start is filtered out.
    layout = {'scope': all_data['scope'], 'positions': [{'id': 'a', 'x': -40, 'y': 900}],
              'view': {'x': 10, 'y': 15, 'k': 2}}
    assert graph_chart.checked_layout(layout, filtered) == layout
    new = graph_chart.diagram_data(rows, starting_id='c', scope=['project', 'query'])
    assert graph_chart.checked_layout(layout, new) is None
    assert rows == original


def test_invalid_layout_or_stale_selection_cannot_enter_graph_state():
    data = graph_chart.diagram_data(claims(), starting_id='a')
    value = {'scope': data['scope'], 'positions': [{'id': 'unknown', 'x': 0, 'y': 0},
             {'id': 'a', 'x': float('nan'), 'y': 0}, {'id': 'b', 'x': 30, 'y': 40}],
             'view': {'x': 0, 'y': 0, 'k': float('inf')}}
    checked = graph_chart.checked_layout(value, data)
    assert checked['positions'] == [{'id': 'b', 'x': 30, 'y': 40}]
    assert checked['view'] == {'x': 0, 'y': 0, 'k': 1}
    assert graph_chart.selected_assertion({'selected': {'scope': 'old', 'assertion_id': 'one'}}, data) is None
    assert graph_chart.selected_assertion({'selected': {'scope': data['scope'], 'assertion_id': 'missing'}}, data) is None


def test_layout_and_selection_callbacks_preserve_same_result_without_graph_services(monkeypatch):
    state, captured, selected = {}, {}, []
    def mount(**args):
        captured.update(args)
        return state.get(args['key'], {})
    monkeypatch.setattr(graph_chart, 'component', lambda: mount)
    st = SimpleNamespace(session_state=state, caption=lambda text: None)
    graph_chart.render(st, claims(), 'chart', starting_id='a', on_select=selected.append)
    data = captured['data']
    layout = {'scope': data['scope'], 'positions': [{'id': 'a', 'x': 3, 'y': 4}],
              'view': {'x': 0, 'y': 0, 'k': 1}}
    state['chart'] = {'layout': layout, 'selected': {'scope': data['scope'], 'assertion_id': 'one'}}
    captured['on_layout_change']()
    captured['on_selected_change']()
    assert selected == ['one'] and state['chart:layout'] == layout
    # A zero-result display filter can unmount the widget; the separate layout survives.
    del state['chart']
    graph_chart.render(st, claims()[1:], 'chart', starting_id='a', all_claims=claims())
    assert captured['data']['layout'] == layout
    assert captured['data']['edges'][0]['assertion_id'] == 'two'


def test_bundled_assets_are_declared_for_installed_packages():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from pathlib import Path
    project = tomllib.loads((Path(__file__).parents[2] / 'pyproject.toml').read_text(encoding='utf8'))
    patterns = project['tool']['setuptools']['package-data']['osintgpt.app']
    assert 'assets/*.mjs' in patterns and 'assets/*.css' in patterns
    for name in ('graph_layout.mjs', 'graph_downloads.mjs', 'graph_diagram.mjs', 'graph_diagram.css'):
        assert (graph_chart.ASSETS / name).is_file()
