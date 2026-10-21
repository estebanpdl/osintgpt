'''Checkpoint invalidation, publication, and pending sources are separate outcomes.'''

import pytest

from osintgpt import Project
from osintgpt.graph import build_graph, saved_graph_status
from osintgpt.graph.events import GraphCancelled
from osintgpt.ingestion import Corpus
from test_extraction_quality import Script


def test_cancelled_source_is_not_reported_running_and_unstarted_source_stays_pending(tmp_path):
    project = Project.create('Source outcomes', home=tmp_path).with_settings(graph_enabled=True)
    corpus = Corpus.load(project.paths.sources)
    for ref in ('one.md', 'two.md'):
        (project.paths.root / ref).write_text('Fictional source text.', encoding='utf-8')
        corpus.register(ref)
    def stop(event):
        if event.kind == 'window_saved':
            raise GraphCancelled('Injected stop after saving paid work.')
    with pytest.raises(GraphCancelled):
        build_graph(project, Script({'entities': [], 'edges': []}), on_event=stop)
    state = saved_graph_status(project)
    assert sorted(item['action'] for item in state['details']) == ['interrupted', 'pending']
    assert state['checkpoints']['windows_saved'] == 1
    completed = build_graph(project, Script({'entities': [], 'edges': []}))
    assert completed.status == 'complete' and completed.accounting.windows_reused == 1
    assert all(item['action'] == 'published' for item in completed.details)
