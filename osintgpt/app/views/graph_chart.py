'''Local diagram interactions over the existing bounded assertion selection.'''

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

ASSETS = Path(__file__).parents[1] / 'assets'


def diagram_data(claims, *, starting_id, all_claims=None, scope=None, selected_id=None):
    '''Stable IDs, not labels, identify nodes; display filtering never changes query scope.'''
    all_claims = claims if all_claims is None else all_claims
    names = {row[side + '_id']: row[side] for row in all_claims for side in ('source', 'target')}
    fields = ('assertion_id', 'source_id', 'target_id', 'source', 'target', 'relation',
              'ref', 'relation_polarity', 'assertion')
    edges = [{name: row[name] for name in fields} for row in claims]
    visible = {row[side + '_id'] for row in edges for side in ('source', 'target')}
    identity = json.dumps([scope, starting_id, sorted(row['assertion_id'] for row in all_claims) if scope is None else None],
                          ensure_ascii=False, sort_keys=True)
    return {'scope': hashlib.sha256(identity.encode('utf8')).hexdigest(),
            'starting_id': starting_id, 'selected_id': selected_id,
            'all_ids': sorted(names),
            'nodes': [{'id': key, 'name': names[key]} for key in sorted(visible)], 'edges': edges}


def checked_layout(value, data):
    '''Keep only finite UI coordinates for this query; never store them in graph.sqlite.'''
    if not isinstance(value, dict) or value.get('scope') != data['scope']:
        return None
    allowed = set(data['all_ids'])
    positions = []
    for row in value.get('positions', []) if isinstance(value.get('positions'), list) else []:
        if (isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'] in allowed
                and all(type(row.get(axis)) in (int, float) and math.isfinite(row[axis])
                        and abs(row[axis]) <= 1e7 for axis in ('x', 'y'))):
            positions.append({name: row[name] for name in ('id', 'x', 'y')})
    view = value.get('view', {})
    if not (isinstance(view, dict)
            and all(type(view.get(axis)) in (int, float) and math.isfinite(view[axis])
                    and abs(view[axis]) <= 1e7 for axis in ('x', 'y', 'k'))
            and .15 <= view['k'] <= 6):
        view = {'x': 0, 'y': 0, 'k': 1}
    return {'scope': data['scope'], 'positions': positions, 'view': view}


def selected_assertion(event, data):
    selected = event.get('selected') if event else None
    if isinstance(selected, dict) and selected.get('scope') == data['scope']:
        key = selected.get('assertion_id')
        return key if key in {row['assertion_id'] for row in data['edges']} else None
    return None


@lru_cache(maxsize=1)
def component_assets():
    return {'js': '\n'.join((ASSETS / name).read_text(encoding='utf8')
                            for name in ('graph_layout.mjs', 'graph_downloads.mjs', 'graph_diagram.mjs')),
            'css': (ASSETS / 'graph_diagram.css').read_text(encoding='utf8')}


def component():
    from streamlit.components.v2 import component as register
    # Only trusted bundled code goes into the component. Source/model text goes in data.
    # Registration is idempotent; a callable must belong to the active Streamlit runtime.
    return register('relationship_diagram', **component_assets())


def render(st, claims, key, *, starting_id, all_claims=None, scope=None, selected_id=None, on_select=None):
    data = diagram_data(claims, starting_id=starting_id, all_claims=all_claims, scope=scope,
                        selected_id=selected_id)
    layout_key = key + ':layout'
    data['layout'] = checked_layout(st.session_state.get(layout_key), data)

    def layout_changed():
        value = checked_layout(st.session_state[key].get('layout'), data)
        if value is not None:
            st.session_state[layout_key] = value

    def selection_changed():
        selected = selected_assertion(st.session_state[key], data)
        if selected is not None and on_select is not None:
            on_select(selected)

    st.caption('Drag a node to move it; drag empty space to pan and scroll to zoom. '
               'Click a relationship dot to inspect evidence. Arrows retain the original assertion direction; '
               'layout distance and position do not indicate importance or evidential strength.')
    result = component()(data=data, key=key, on_layout_change=layout_changed,
                         on_selected_change=selection_changed, height='content', width='stretch')
    return selected_assertion(result, data)
