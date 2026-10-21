'''Synthetic browser harness: no project, database, credentials, or provider calls.'''

import json
import streamlit as st
from osintgpt.app.views import graph_chart

st.set_page_config(layout='wide')
st.title('Offline graph layout checks')
names = {'a': '同名', 'b': '同名', 'c': '<img src=x onerror=alert(1)>', 'd': 'منظمة تجريبية'}
rows = [dict(assertion_id=f'assert-{i}', source_id=source, target_id=target,
             source=names[source], target=names[target], relation=f'relación {i}', ref='synthetic.md',
             relation_polarity=polarity, assertion=qualification)
        for i, (source, target, polarity, qualification) in enumerate([
            ('a', 'b', 'positive', 'stated'), ('b', 'a', 'negative', 'alleged'),
            ('a', 'b', 'unknown', 'uncertain'), ('a', 'a', 'positive', 'hypothetical'),
            ('b', 'c', 'positive', 'stated'), ('c', 'd', 'positive', 'stated')])]
result = st.selectbox('Result', ['First', 'Second'])
filter_value = st.selectbox('Display filter', ['All', 'Indirect only', 'None'])
filtered = rows if filter_value == 'All' else rows[4:] if filter_value == 'Indirect only' else []
st.button('Unrelated rerun')
if st.session_state.get('chosen') not in [row['assertion_id'] for row in filtered]:
    st.session_state['chosen'] = filtered[0]['assertion_id'] if filtered else None
if filtered:
    graph_chart.render(st, filtered, 'diagram', all_claims=rows, starting_id='a' if result == 'First' else 'b',
                       scope=result, selected_id=st.session_state['chosen'],
                       on_select=lambda key: st.session_state.__setitem__('chosen', key))
    st.selectbox('Inspect assertion', [row['assertion_id'] for row in filtered], key='chosen')
st.write('Evidence assertion: ' + str(st.session_state['chosen']))
st.code(json.dumps(st.session_state.get('diagram:layout'), ensure_ascii=False), language='json')
