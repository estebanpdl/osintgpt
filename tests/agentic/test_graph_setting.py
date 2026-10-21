'''Disabling relationship retrieval hides its tool and rejects direct execution.'''

from types import SimpleNamespace

import pytest

from osintgpt import Project, agentic_answer
from osintgpt.agentic.registry import run_tool
from osintgpt.agentic.tools import ToolContext
from osintgpt.graph import Edge, graph_for
from osintgpt.llm.calling import ModelTurn


@pytest.mark.parametrize('enabled', [False, True])
def test_graph_tool_offered_only_when_enabled(tmp_path, enabled):
    project = Project.create('Graph tools', home=tmp_path).with_settings(
        graph_enabled=enabled, suggest_followups=False
    )

    class Model:
        supports_tools = True

        def generate_with_tools(self, system, question, tools, history):
            assert ('graph_query' in [tool.name for tool in tools]) is enabled
            return ModelTurn(text='No evidence requested.')

    agentic_answer(project, 'Question', None, Model(), store=SimpleNamespace(count=lambda: 1))


def test_disabled_handler_preserves_graph_and_reenable_restores_access(tmp_path):
    project = Project.create('Graph tools', home=tmp_path)
    with graph_for(project) as graph:
        graph.add([], [Edge('Alpha', 'Beta', 'supports', 'a.md', 'Alpha supports Beta.')])
    context = ToolContext(project=project, embedder=None)
    result = run_tool(context, 'graph_query', {'entity': 'Alpha'})
    assert result.payload['enabled'] is False
    assert 'claims' not in result.payload
    with graph_for(project) as graph:
        assert graph.edge_count == 1
    context.project = project.with_settings(graph_enabled=True)
    result = run_tool(context, 'graph_query', {'entity': 'Alpha'})
    assert len(result.payload['claims']) == 1


def test_disabled_handler_does_not_open_graph(tmp_path, monkeypatch):
    project = Project.create('Graph tools', home=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled retrieval must not open the graph')

    monkeypatch.setattr('osintgpt.graph.retrieval.GraphStore', forbidden)
    result = run_tool(ToolContext(project, None), 'graph_query', {'entity': 'Alpha'})
    assert result.payload['enabled'] is False
