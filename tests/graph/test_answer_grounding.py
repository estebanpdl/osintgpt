'''The answer loop receives selected identities, qualified links, and source context.'''

import json
from types import SimpleNamespace

import pytest

from osintgpt import agentic_answer, index_project
from osintgpt.agentic.registry import run_tool, tools_for
from osintgpt.agentic.tools import ToolContext
from osintgpt.graph import Entity, graph_for
from osintgpt.llm.calling import ModelTurn, ToolCall


def request(name, **arguments):
    return ModelTurn(calls=[ToolCall(id=name, name=name, arguments=arguments)])


def last_result(history):
    return json.loads(next(iter(history[-1].results.values())))


def test_graph_only_answer_can_select_identity_read_path_and_cite_every_link(graph_case):
    case = graph_case
    first = case.publish('a.md', 'According to Witness, Álex allegedly funds Beta in 2024.', source='Álex', source_id='alex_one',
                         assertion='alleged', attributed_to='Witness', event_time='2024')
    second = case.publish('b.md', 'Beta funds Gamma.', source='Beta', target='Gamma')
    with graph_for(case.project) as graph:
        graph.add([Entity('alex_two', 'Álex', description='A different person')], [])
    project = case.project.with_settings(suggest_followups=False)

    class Model:
        supports_tools = True

        def generate_with_tools(self, system, question, tools, history):
            assert 'assertion ID for every link used in an answer' in system
            assert {'graph_entities', 'graph_query', 'graph_evidence'} <= {tool.name for tool in tools}
            if not history:
                return request('graph_entities', query='Álex')
            result = last_result(history)
            if len(history) == 1:
                assert result['status'] == 'ambiguous'
                return request('graph_query', entity='alex_one', target='Gamma', direction='outgoing')
            if len(history) == 2:
                assert [row['assertion_id'] for row in result['path']] == [first.assertion_id, second.assertion_id]
                assert result['path'][0]['assertion'] == 'alleged'
                assert result['path'][0]['event_time'] == '2024'
                return request('graph_evidence', assertion_id=first.assertion_id)
            assert result['text'] == first.evidence and result['verification']['snapshot'] == 'found'
            return ModelTurn(text=f'Witness alleges Álex funded Beta in 2024 [{first.ref}; {first.assertion_id}]. '
                            f'Beta funds Gamma [{second.ref}; {second.assertion_id}].')

    answer = agentic_answer(project, 'What connects the first Álex to Gamma?', None, Model(),
                            store=SimpleNamespace(count=lambda: 0))
    assert set(answer.sources) == {'a.md', 'b.md'}
    assert first.assertion_id in answer.text and second.assertion_id in answer.text
    assert not answer.trace.failures
    from osintgpt.agentic import answer_from_dict, answer_to_dict
    restored = answer_from_dict(answer_to_dict(answer))
    assert restored.trace.entries[1].payload['path'][0]['assertion_id'] == first.assertion_id
    assert restored.trace.entries[2].payload['text'] == first.evidence


@pytest.mark.parametrize('state', ['disabled', 'unavailable', 'unbuilt', 'no_connection'])
def test_graph_outcomes_leave_semantic_and_lexical_tools_operational(graph_case, state):
    case = graph_case
    case.publish('a.md', 'Alpha funds Beta.')
    case.publish('b.md', 'Other funds Separate.', source='Other', target='Separate')

    class Embedder:
        model = 'offline'
        supports_images = False

        def embed(self, texts, **kwargs):
            return [[1.0, 0.0] for text in texts]

    embedder = Embedder()
    index_project(case.project, embedder)
    project = case.project
    path = project.paths.root / 'graph.sqlite'
    if state == 'disabled':
        project = project.with_settings(graph_enabled=False)
    elif state in ('unavailable', 'unbuilt'):
        path.unlink()
        if state == 'unavailable':
            path.write_text('invalid database')
    context = ToolContext(project, embedder)
    result = run_tool(context, 'graph_query', {'entity': 'Alpha', 'target': 'Other'})
    assert result.payload['status'] == state
    assert result.payload['fallback_tools'] == ['semantic_search', 'exact_search']
    assert run_tool(context, 'semantic_search', {'query': 'Alpha'}).payload['passages']
    assert run_tool(context, 'exact_search', {'terms': ['Alpha']}).payload['passages']
    if state == 'disabled':
        assert not any(tool.name.startswith('graph_') for tool in tools_for(project))
        for name, arguments in [('graph_entities', {'query': 'Alpha'}),
                                ('graph_evidence', {'assertion_id': 'assert_missing'})]:
            assert run_tool(context, name, arguments).payload['status'] == 'disabled'
