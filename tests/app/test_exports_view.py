'''Graph and findings download controls bind to the exact reviewed selections.'''

import json

from osintgpt.app.session import VIEW
from osintgpt.projects import start_conversation, append_turn
from osintgpt.graph import query_graph
from test_app_renders import home, project, run
from test_graph_view import published_graph, button, widget, choose_start


def test_full_and_filtered_graph_downloads_are_prepared_without_models(published_graph, monkeypatch):
    project, _ = published_graph
    def forbidden(*a, **k):
        raise AssertionError('Exports must not build a provider')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    app.button(key=f'graph-export:{project.id}:full:prepare').click().run()
    full = app.session_state[f'graph-export:{project.id}:full:ready']
    assert len(json.loads(full['json'])['edges']) == 3 and not app.exception
    assert len(app.download_button) == 2
    choose_start(app, project)
    widget(app.radio, 'Source scope').set_value('Selected sources').run()
    widget(app.multiselect, 'Source filter').select('denial.md').run()
    button(app, 'Explore neighborhood').click().run()
    widget(app.text_input, 'Search relationships').set_value('does not').run()
    widget(app.radio, 'Export scope').set_value('All matching results').run()
    app.button(key=f'graph-export:{project.id}:selection:prepare').click().run()
    assert not app.exception
    ready = app.session_state[f'graph-export:{project.id}:selection:ready']
    package = json.loads(ready['json'])
    assert len(package['edges']) == 1 and package['edges'][0]['relation_polarity'] == 'negative'
    assert package['metadata']['query']['request']['refs'] == ['denial.md']
    assert package['metadata']['filters']['text'] == 'does not'
    assert len(app.download_button) == 4
    app.download_button(key=f'graph-export:{project.id}:selection:json').click().run()
    assert app.session_state[f'graph-export:{project.id}:selection:ready']['json'] == ready['json']
    widget(app.text_input, 'Search relationships').set_value('no results').run()
    assert len(app.download_button) == 2
    assert any('selection changed' in item.value for item in app.info)


def test_selected_answer_and_canon_downloads_preserve_saved_graph_evidence(published_graph):
    project, _ = published_graph
    conversation = start_conversation(project, 'First')
    append_turn(conversation, 'First', {'answer': 'Qualified answer [funding.md].', 'sources': ['funding.md'],
                                     'trace': {'calls': [{'tool': 'graph_query', 'payload': query_graph(project, 'alpha', 'gamma')}]}})
    append_turn(conversation, 'Unselected', {'answer': 'Keep out of export'})
    (project.paths.canon / 'note.md').write_text('# Finding\nA finding [source](../funding.md).')
    app = run(project, **{VIEW: 'Ask', 'conversation_id': conversation.id})
    widget(app.multiselect, 'Answers to export').select(0).run()
    widget(app.multiselect, 'Canon findings to export').select('note.md').run()
    button(app, 'Prepare findings downloads').click().run()
    assert not app.exception and len(app.download_button) == 2
    key = f'findings-export:{project.id}:{conversation.id}:ready'
    ready = app.session_state[key]
    assert 'Alpha funds Beta.' in ready['md'] and 'Beta owns Gamma.' in ready['csv']
    assert 'Keep out of export' not in ready['md'] + ready['csv']
    widget(app.multiselect, 'Answers to export').unselect(0).run()
    assert len(app.download_button) == 0
    button(app, 'Prepare findings downloads').click().run()
    assert len(app.session_state[key]['package']['items']) == 1


def test_authored_findings_can_export_without_a_conversation(project):
    (project.paths.canon / 'note.md').write_text('# Operator finding\nEvidence is recorded inline.')
    app = run(project, **{VIEW: 'Ask'})
    widget(app.multiselect, 'Canon findings to export').select('note.md').run()
    button(app, 'Prepare findings downloads').click().run()
    assert not app.exception and len(app.download_button) == 2
