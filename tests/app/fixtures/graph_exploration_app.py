'''Real exploration UI over an isolated synthetic graph, without model providers.'''

import os
import tempfile
from pathlib import Path

import streamlit as st

from osintgpt import Project
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.ingestion import Corpus
from osintgpt.app.views.graph import explore

st.set_page_config(layout='wide')
st.title('Offline progressive graph checks')
if 'fixture-project' not in st.session_state:
    project = Project.create('Synthetic graph', home=Path(tempfile.mkdtemp(prefix='osintgpt-exploration-browser-')))
    project = project.with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'notes.md').write_text('Example evidence.', encoding='utf8')
    Corpus.load(project.paths.sources).register('notes.md')
    count = int(os.environ.get('OSINTGPT_TEST_ASSERTIONS', '805'))
    with graph_for(project) as graph:
        graph.add([Entity('alpha', 'Alpha'), *[Entity(f'n{i}', f'Identity {i}') for i in range(count)]],
                  [Edge('Alpha', f'Identity {i}', 'links', 'notes.md', 'Example evidence.',
                        source_id='alpha', target_id=f'n{i}') for i in range(count)])
    st.session_state['fixture-project'] = project
explore(st, st.session_state['fixture-project'], st.session_state, ['notes.md'], 'fixture')
