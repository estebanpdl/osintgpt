'''The graph workflow renders and responds to operator actions without provider calls.'''

import json
from threading import Event

import pytest

from osintgpt.app.session import VIEW
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot
from test_app_renders import home, project, run


@pytest.fixture
def published_graph(project):
    project = project.with_settings(graph_enabled=True)
    project.save()
    edges = []
    for ref, source, target, quote, relation_polarity in [
        ('funding.md', 'Alpha', 'Beta', 'Alpha funds Beta.', 'positive'),
        ('denial.md', 'Alpha', 'Beta', 'Alpha does not fund Beta.', 'negative'),
        ('ownership.md', 'Beta', 'Gamma', 'Beta owns Gamma.', 'positive')]:
        path = project.paths.root / ref
        path.write_text('# Findings\n\n' + quote + '\n\nSource context.', encoding='utf-8')
        Corpus.load(project.paths.sources).register(ref)
        snapshot = load_snapshot(path, ref=ref, cache_dir=project.paths.extracts)
        provenance = snapshot.provenance(snapshot.document(ref), quote)
        provenance['supporting'] = [{'quote': quote, 'matches': provenance['matches']}]
        edge = Edge(source, target, 'owns' if target == 'Gamma' else 'funds', ref, quote, provenance,
                    source.casefold(), target.casefold(), relation_polarity=relation_polarity, evidence_quotes=(quote,))
        edges.append(edge)
        with graph_for(project) as graph:
            graph.add([Entity(source.casefold(), source), Entity(target.casefold(), target)], [edge])
            graph.state.completed(ref, snapshot.fingerprint, snapshot.version)
    with graph_for(project) as graph:
        graph.connection.execute("UPDATE graph_build SET started_at='2026-01-01' WHERE id=1")
        graph.connection.commit()
    return project, edges


def button(app, label):
    return next(item for item in app.button if item.label == label)


def widget(items, label):
    return next(item for item in items if item.label == label)


def choose_start(app, project, value='Alpha', identity='alpha'):
    widget(app.text_input, 'Starting entity').set_value(value).run()
    app.button(key=f'graph-explore:{project.id}:start:find').click().run()
    app.selectbox(key=f'graph-explore:{project.id}:start:choice').select(identity).run()


def test_empty_graph_view_and_preflight_do_not_build_provider(project, monkeypatch):
    project = project.with_settings(generation_model='offline-graph')
    project.save()
    def forbidden(*args, **kwargs):
        raise AssertionError('Inspection must not construct a provider')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    assert button(app, 'Build graph').disabled
    button(app, 'Preview build workload').click().run()
    assert not app.exception
    assert any('Preview:' in item.value for item in app.info)
    assert not (project.paths.root / 'graph.sqlite').exists()


def test_neighborhood_filters_and_saved_source_inspector(published_graph):
    project, edges = published_graph
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception
    choose_start(app, project)
    assert widget(app.number_input, 'Traversal depth').value == 1
    widget(app.number_input, 'Traversal depth').set_value(2).run()
    button(app, 'Explore neighborhood').click().run()
    assert not app.exception
    assert len(app.get('bidi_component')) == 1
    assert any('Direct assertions involving this identity: 2 across all sources; 2 in the active source scope.'
               in item.value for item in app.caption)
    assert any('Distinct connected identities: 1 across all sources; 1 in the active source scope.'
               in item.value for item in app.caption)
    assert any('Neighborhood: 3 assertions found; 3 returned.' in item.value for item in app.caption)
    assert any('2 assertions involve the selected identity directly; 1 connect other identities' in item.value
               for item in app.caption)
    assert not any('depth_boundary' in item.value for item in app.warning)
    assert any('Exploration complete within the selected scope: up to 2 relationship steps' in item.value
               for item in app.caption)
    assert any('Relation polarity:' in item.value for item in app.caption)
    assert any('Relation polarity' in frame.value.columns for frame in app.dataframe)
    assert any('Other assertions for this entity pair' in item.label for item in app.expander)
    widget(app.text_input, 'Search relationships').set_value('does not').run()
    assert any('of 1 filter matches among 3 discovered assertions.'
               in item.value for item in app.caption)
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 1
    button(app, 'Read source context').click().run()
    assert any('Source context.' in item.value for item in app.text)
    widget(app.text_input, 'Search relationships').set_value('').run()
    widget(app.multiselect, 'Relationship filter').select('owns').run()
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 1
    widget(app.multiselect, 'Relationship filter').unselect('owns').run()
    widget(app.radio, 'Source scope').set_value('Selected sources').run()
    widget(app.multiselect, 'Source filter').select('denial.md').run()
    assert any('Run the query again' in item.value for item in app.info)
    button(app, 'Explore neighborhood').click().run()
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 1
    assert not app.exception


def test_connection_shows_each_qualified_link(published_graph):
    project, _ = published_graph
    app = run(project, **{VIEW: 'Graph'})
    widget(app.radio, 'Explore').set_value('Connection').run()
    choose_start(app, project)
    widget(app.text_input, 'Destination entity').set_value('Gamma').run()
    app.button(key=f'graph-explore:{project.id}:end:find').click().run()
    app.selectbox(key=f'graph-explore:{project.id}:end:choice').select('gamma').run()
    widget(app.selectbox, 'Traversal direction').select('outgoing').run()
    widget(app.number_input, 'Traversal depth').set_value(2).run()
    button(app, 'Find connection').click().run()
    assert not app.exception
    assert any('Complete supporting path: 2 steps' in item.value for item in app.caption)
    frame = next(item.value for item in app.dataframe if 'Path step' in item.value.columns)
    assert list(frame['Path step']) == [1, 2]
    assert any('Alpha funds Beta.' in item.value for item in app.text)
    assert any('Beta owns Gamma.' in value for value in frame['Quotation preview'])


def test_large_evidence_keeps_app_rows_search_and_exports(published_graph, monkeypatch):
    from dataclasses import replace
    from osintgpt.graph.inspection import OperatorQuery
    project, edges = published_graph
    text = 'Alpha funds Beta. ' + 'Supporting material. ' * 4000 + ' Único final.'
    path = project.paths.root / 'funding.md'
    path.write_text(text, encoding='utf-8')
    snapshot = load_snapshot(path, ref='funding.md', cache_dir=project.paths.extracts)
    provenance = snapshot.provenance(snapshot.document('funding.md'), text)
    provenance['supporting'] = [{'quote': text, 'matches': provenance['matches']}]
    large = replace(edges[0], evidence=text, evidence_quotes=(text,), provenance=provenance)
    with graph_for(project) as graph:
        graph.forget(['funding.md'])
        graph.add([], [large])
        graph.state.completed('funding.md', snapshot.fingerprint, snapshot.version)

    def forbidden(*args, **kwargs):
        raise AssertionError('Inspection must not construct a provider.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    hydrated = []
    full_result = OperatorQuery.full_result

    def record_hydration(query):
        hydrated.append(True)
        return full_result(query)
    monkeypatch.setattr(OperatorQuery, 'full_result', record_hydration)
    app = run(project, **{VIEW: 'Graph'})
    choose_start(app, project)
    widget(app.number_input, 'Traversal depth').set_value(2).run()
    button(app, 'Explore neighborhood').click().run()
    assert not app.exception
    assert any('Neighborhood: 3 assertions found; 3 returned.' in item.value for item in app.caption)
    assert not any('response size limit' in item.value.lower() for item in app.warning)
    frame = next(item.value for item in app.dataframe if 'Quotation preview' in item.value.columns)
    assert len(frame) == 3 and max(map(len, frame['Quotation preview'])) == 240
    assert not hydrated
    widget(app.text_input, 'Search relationships').set_value('ÚNICO FINAL').run()
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 1
    assert any(text in item.value for item in app.text)
    widget(app.radio, 'Export scope').set_value('All matching results').run()
    app.button(key=f'graph-export:{project.id}:selection:prepare').click().run()
    assert not app.exception and not hydrated  # Export expands only the filtered selection.
    package = json.loads(app.session_state[f'graph-export:{project.id}:selection:ready']['json'])
    assert package['edges'][0]['evidence_quotes'] == [text]
    assert package['edges'][0]['provenance'] == provenance
    assert package['edges'][0]['assertion_id'] == large.assertion_id
    assert 'quotation_preview' not in json.dumps(package)


def test_settings_persist_and_source_allowance_does_not_cap_continuation(project):
    app = run(project, **{VIEW: 'Graph'})
    widget(app.number_input, 'Generation requests per minute').set_value(12).run()
    button(app, 'Save build settings').click().run()
    from osintgpt.graph.run_settings import GraphRunSettings
    assert GraphRunSettings.load(project).rpm == 12
    fresh = run(project, **{VIEW: 'Graph'})
    assert widget(fresh.number_input, 'Generation requests per minute').value == 12
    widget(fresh.number_input, 'Source window allowance (UTF-8 bytes)').set_value(100).run()
    assert not fresh.exception
    button(fresh, 'Save build settings').click().run()
    assert GraphRunSettings.load(project).context_budget == 100
    assert not any(item.label in {'Entity context allowance', 'Maximum calls per record'} for item in fresh.number_input)


def test_full_verification_and_audit_are_available_in_app(published_graph):
    project, _ = published_graph
    app = run(project, **{VIEW: 'Graph'})
    button(app, 'Verify all graph evidence').click().run()
    assert not app.exception and any('3 verified' in item.value for item in app.info)
    button(app, 'Inspect rejected proposals').click().run()
    assert not app.exception


def test_empty_selected_scope_blocks_query_and_review_is_provider_free(published_graph, monkeypatch):
    project, _ = published_graph
    def forbidden(*args, **kwargs):
        raise AssertionError('Inspection must not construct a provider.')
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider', forbidden)
    app = run(project, **{VIEW: 'Graph'})
    choose_start(app, project)
    widget(app.radio, 'Source scope').set_value('Selected sources').run()
    assert button(app, 'Explore neighborhood').disabled
    assert any('Select at least one source' in item.value for item in app.info)
    widget(app.radio, 'Source scope').set_value('All sources').run()
    assert not button(app, 'Explore neighborhood').disabled
    widget(app.text_input, 'Identity name or ID').set_value('Alpha').run()
    button(app, 'Review identities').click().run()
    assert button(app, 'Preview identity decision') and not app.exception
    button(app, 'Preview local revalidation').click().run()
    assert not app.exception and button(app, 'Apply supported corrections').disabled


def test_chat_previews_survive_real_trace_serialization(published_graph):
    from osintgpt.agentic import AgenticAnswer, Trace, answer_from_dict, answer_to_dict
    from osintgpt.app.views.chat import passages_of
    from osintgpt.graph import query_graph
    project, _ = published_graph
    payload = query_graph(project, 'Alpha', 'Gamma')
    trace = Trace()
    trace.record(1, 'graph_query', {}, payload=payload)
    answer = AgenticAnswer('How?', 'Linked through Beta.', trace=trace, sources=['funding.md', 'ownership.md'])
    data = json.loads(json.dumps(answer_to_dict(answer)))
    restored = answer_from_dict(data)
    assert len(passages_of(restored)) == 2
    assert passages_of(restored)[0]['graph_claim']['source_version']
    from osintgpt.projects import start_conversation, append_turn
    conversation = start_conversation(project, answer.question)
    append_turn(conversation, answer.question, data)
    app = run(project, **{VIEW: 'Ask', 'conversation_id': conversation.id})
    assert not app.exception
    assert any('Alpha funds Beta.' in item.value for item in app.text)
    assert any('Saved evidence: found' in item.value for item in app.caption)


def test_build_survives_navigation_and_second_session_then_resumes(project, monkeypatch):
    from osintgpt.graph.jobs import GraphJobs
    from osintgpt.graph import saved_graph_status
    project = project.with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'a.md').write_text('Alpha funds Beta.')
    Corpus.load(project.paths.sources).register('a.md')
    jobs, entered, release = GraphJobs(), Event(), Event()
    calls = []
    class Generator:
        def graph_options(self):
            return {'provider': 'fixture', 'model': 'test'}
        def generate(self, *args):
            calls.append(1)
            entered.set()
            assert release.wait(30)
            return '{"entities":[],"edges":[]}'
    monkeypatch.setattr('osintgpt.app.views.graph_controls.JOBS', jobs)
    monkeypatch.setattr('osintgpt.app.views.graph_controls.generator_factory', lambda *args: Generator)
    app = run(project, **{VIEW: 'Graph'})
    button(app, 'Build graph').click().run()
    try:
        assert entered.wait(10)
        identity = jobs.get(project)['id']
        widget(app.radio, 'View').set_value('Material').run()
        widget(app.radio, 'View').set_value('Graph').run()
        fresh = run(project, **{VIEW: 'Graph'})
        assert not fresh.exception and not app.exception
        assert jobs.get(project)['id'] == identity and calls == [1]
        assert fresh.button(key=f'graph-build:{project.id}:start').disabled
        button(fresh, 'Cancel build').click().run()
        assert jobs.get(project)['cancel_requested']
    finally:
        release.set()
        jobs.jobs[jobs.key(project)].thread.join(10)
    assert saved_graph_status(project)['checkpoints']['windows_saved'] == 1
    fresh.run()
    widget(fresh.selectbox, 'Build action').select('Resume saved work').run()
    button(fresh, 'Resume build').click().run()
    jobs.jobs[jobs.key(project)].thread.join(10)
    fresh.run()
    assert not fresh.exception
    assert jobs.get(project)['status'] == 'complete' and calls == [1]


def test_changed_source_verification_and_explicit_identity_choice(published_graph):
    project, _ = published_graph
    (project.paths.root / 'denial.md').write_text('No named organizations remain here.')
    app = run(project, **{VIEW: 'Graph'})
    assert next(item for item in app.metric if item.label == 'Current sources').value == '2/3'
    widget(app.text_input, 'Starting entity').set_value('Alpha').run()
    app.button(key=f'graph-explore:{project.id}:start:find').click().run()
    assert app.selectbox(key=f'graph-explore:{project.id}:start:choice').value is None
    assert button(app, 'Explore neighborhood').disabled
    app.selectbox(key=f'graph-explore:{project.id}:start:choice').select('alpha').run()
    button(app, 'Explore neighborhood').click().run()
    widget(app.selectbox, 'Current quotation verification').select('not_found').run()
    assert len(widget(app.selectbox, 'Inspect assertion').options) == 1
    assert any('Saved evidence: found; current evidence: not_found' in item.value for item in app.caption)


def test_ask_reads_graph_and_saves_previews_without_an_embedding_client(published_graph, monkeypatch):
    from osintgpt.llm.calling import ModelTurn, ToolCall
    from osintgpt.projects import open_conversation
    project, _ = published_graph
    project = project.with_settings(suggest_followups=False)
    project.save()
    class Model:
        supports_tools = True
        def generate_with_tools(self, system, question, tools, history, **kwargs):
            if not history:
                return ModelTurn(calls=(ToolCall('graph', 'graph_query', {'entity': 'alpha', 'target': 'gamma'}),))
            return ModelTurn(text='Alpha connects to Gamma through Beta [funding.md; ownership.md].')
    monkeypatch.setattr('osintgpt.app.session.build_embedding_provider', lambda *a, **k: pytest.fail('Unneeded embedding client'))
    monkeypatch.setattr('osintgpt.app.session.build_generation_provider', lambda *a, **k: Model())
    app = run(project, **{VIEW: 'Ask'})
    app.chat_input[0].set_value('What connects Alpha to Gamma?').run()
    assert not app.exception
    assert any('Alpha funds Beta.' in item.value for item in app.text)
    assert any('Beta owns Gamma.' in item.value for item in app.text)
    saved = open_conversation(project, app.session_state['conversation_id'])
    assert saved.turns[-1].answer['trace']['calls'][0]['payload']['path']


@pytest.mark.parametrize('invalid', ['unsupported', 'corrupt'])
def test_invalid_graph_is_reported_without_modifying_it(project, invalid):
    import sqlite3
    path = project.paths.root / 'graph.sqlite'
    if invalid == 'unsupported':
        with sqlite3.connect(path) as connection:
            connection.execute('PRAGMA user_version=99')
    else:
        path.write_text('not a SQLite database')
    original = path.read_bytes()
    app = run(project, **{VIEW: 'Graph'})
    assert not app.exception and any('Cannot open graph' in item.value for item in app.error)
    assert any(f'graph migrate --project {project.slug} --dry-run' in item.value for item in app.caption)
    assert path.read_bytes() == original


@pytest.mark.parametrize('view', ['Graph', 'Settings'])
def test_legacy_records_show_reextraction_notice(project, home, view):
    (project.paths.root / 'source.md').write_text('Alpha supports Beta.')
    Corpus.load(project.paths.sources).register('source.md')
    with graph_for(project) as graph:
        graph.add([Entity('alpha', 'Alpha', provenance={'legacy_schema': 0, 'requires_reextraction': True})],
                  [Edge('Alpha', 'Beta', 'supports', 'source.md', 'Alpha supports Beta.',
                        provenance={'legacy_schema': 0, 'requires_reextraction': True})])
    app = run(project, **{VIEW: view})
    assert not app.exception
    assert any('Migrated legacy data: 1 entities and 1 relationships' in item.value for item in app.warning)


def test_build_inherits_project_model_and_stricter_cost_ceiling(project, home, monkeypatch):
    from types import SimpleNamespace
    from osintgpt.config import Settings
    from osintgpt.app.views.graph_controls import generator_factory
    from osintgpt.graph.run_settings import GraphRunSettings
    from osintgpt.graph.provider import LazyGraphGenerator
    from osintgpt.cli.graph_generation import _LazyGenerator
    project = project.with_settings(generation_provider='openai', generation_model='project-model', cost_ceiling_usd=0.25)
    runtime = SimpleNamespace(project=project, config=Settings())
    monkeypatch.setattr('osintgpt.app.views.graph_controls.build_generation_provider',
                        lambda *a, **k: pytest.fail('Metadata must remain lazy'))
    generator = generator_factory(runtime, home, GraphRunSettings(cost_ceiling_usd=0.5))()
    assert generator.graph_options()['model'] == 'project-model'
    assert generator.recorder.cost_ceiling_usd == 0.25
    assert _LazyGenerator is LazyGraphGenerator
