'''Recover successful native responses across cost stops and unfinished quote repair.'''

import json

import httpx
import pytest
from openai import OpenAI

from osintgpt.graph import ExtractionOptions, build_graph, graph_for, saved_graph_status
from osintgpt.graph.events import GraphCancelled
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder
from osintgpt.llm.generation_pacing import GenerationRequestError
from tests.conftest import FAKE_KEY
from tests.llm.test_generation_contract import provider_fixture
from test_build_status import project
from test_extraction_contract import complete_proposal


@pytest.mark.parametrize('kind', ['responses', 'anthropic', 'gemini', 'ollama'])
def test_native_response_saved_before_cost_stop_and_replayed_at_record_limit(project, kind):
    provider, requests = provider_fixture(kind)
    provider.recorder.cost_ceiling_usd = 0
    options = ExtractionOptions(max_calls=1)
    try:
        with pytest.raises(CostLimitReached):
            build_graph(project, provider, options=options)
        status = saved_graph_status(project)
        assert status['checkpoints']['responses_saved'] == 1
        assert status['checkpoints']['windows_saved'] == 0
        assert status['counts']['requests_succeeded'] == status['usage']['calls'] == 1
        provider.recorder = UsageRecorder(cost_ceiling_usd=0)
        report = build_graph(project, provider, options=options)
        assert report.status == 'complete' and len(requests) == 1
        assert report.accounting.responses_reused == 1
        assert report.calls == report.usage['calls'] == 0
        assert saved_graph_status(project)['checkpoints']['responses_saved'] == 0
    finally:
        provider.client.close()


def scripted_native(*replies):
    provider, _ = provider_fixture('gemini')
    provider.client.close()
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(reply, BaseException):
            raise reply
        from copy import deepcopy
        reply = deepcopy(reply)
        for entity in reply.get('entities', []):
            entity.setdefault('resolution', {'action': 'create', 'method': 'new', 'reason': 'New sourced entity.',
                                             'evidence': [], 'candidate_evidence': [], 'type_compatible': False, 'type_reason': ''})
        message = {'role': 'assistant', 'content': json.dumps(reply)}
        submission = any(tool.get('function', {}).get('name') == 'submit_extraction' for tool in calls[-1].get('tools', []))
        if submission:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'submit', 'type': 'function',
                        'function': {'name': 'submit_extraction', 'arguments': json.dumps(reply)}}]}
        return httpx.Response(200, json={
            'id': 'response', 'object': 'chat.completion', 'created': 0, 'model': provider.model,
            'choices': [{'index': 0, 'finish_reason': 'tool_calls' if submission else 'stop', 'message': message}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
    provider.client = OpenAI(api_key=FAKE_KEY, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    provider.model = 'gpt-4o-mini'
    return provider, calls


@pytest.mark.parametrize('stop_at', ['extraction', 'repair'])
def test_repair_reuses_each_saved_response_without_duplicate_audit(project, stop_at):
    draft = complete_proposal('Invented.')
    correction = {'corrections': [{'index': 0, 'evidence': ['Alpha supports Beta.']}]}
    first, requests = scripted_native(draft, correction)
    cost = Usage('gemini', first.model, 10, 5).estimated_cost
    first.recorder.cost_ceiling_usd = cost * (0.5 if stop_at == 'extraction' else 1.5)
    try:
        with pytest.raises(CostLimitReached):
            build_graph(project, first)
        expected = 1 if stop_at == 'extraction' else 2
        assert len(requests) == saved_graph_status(project)['checkpoints']['responses_saved'] == expected
    finally:
        first.client.close()
    second, requests = scripted_native(correction)
    try:
        report = build_graph(project, second)
        assert report.status == 'complete' and report.edges == 1
        assert report.accounting.responses_reused == expected
        assert len(requests) == report.usage['calls'] == 2 - expected
        with graph_for(project) as graph:
            assert len(graph.rejections()) == 1 and graph.rejections()[0].status == 'repaired'
    finally:
        second.client.close()


def test_cancel_after_receipt_keeps_response_and_usage_before_window_commit(project):
    provider, requests = provider_fixture('responses')
    def stop(event):
        if event.kind == 'response_saved':
            raise GraphCancelled('Stopped by operator')
    try:
        with pytest.raises(GraphCancelled):
            build_graph(project, provider, on_event=stop)
        status = saved_graph_status(project)
        assert status['status'] == 'interrupted'
        assert status['checkpoints']['responses_saved'] == 1
        assert status['usage']['calls'] == 1
        report = build_graph(project, provider)
        assert report.accounting.responses_reused == 1 and len(requests) == 1
        assert report.usage['calls'] == 0
    finally:
        provider.client.close()


def test_malformed_response_is_audited_and_resume_does_not_repeat_invalid_work(project):
    first, requests = scripted_native({'wrong': []})
    try:
        result = build_graph(project, first)
        assert result.accounting.invalid_responses == 2 and result.failed
        from osintgpt.graph.diagnostics import generation_diagnostics
        diagnostics = generation_diagnostics(project, raw=True)
        assert len(diagnostics['results']) == 2
        assert all(row['state'] == 'invalid' for row in diagnostics['results'])
        before = len(requests)
        assert build_graph(project, first).failed
        assert len(requests) == before
    finally:
        first.client.close()


def test_journal_failure_does_not_repeat_successful_http_request(project, monkeypatch):
    provider, requests = provider_fixture('responses')
    def fail(*args):
        raise OSError('Checkpoint storage unavailable')
    monkeypatch.setattr('osintgpt.graph.checkpoints.RecordCheckpoint.save_response', fail)
    try:
        result = build_graph(project, provider)
        assert result.failed and 'storage unavailable' in result.failed[0].problem
        assert len(requests) == result.calls == result.usage['calls'] == 1
        assert result.accounting.requests_succeeded == 1 and result.accounting.requests_failed == 0
    finally:
        provider.client.close()


def test_permanent_provider_failure_stops_before_other_sources_and_preserves_publication(project):
    provider, _ = scripted_native(complete_proposal('Alpha supports Beta.'))
    try:
        assert build_graph(project, provider).edges == 1
    finally:
        provider.client.close()
    (project.paths.root / 'material' / 'b.md').write_text('Another source.')
    provider, _ = scripted_native({})
    provider.client.close()
    attempts = []
    def fail(request):
        attempts.append(request)
        return httpx.Response(401, json={'error': {'code': 'invalid_api_key', 'message': 'Invalid credential'}})
    provider.client = OpenAI(api_key=FAKE_KEY, http_client=httpx.Client(transport=httpx.MockTransport(fail)))
    try:
        with pytest.raises(GenerationRequestError):
            build_graph(project, provider, rebuild=True)
        status = saved_graph_status(project)
        assert len(attempts) == 1 and status['status'] == 'failed'
        assert status['available'] and status['edges'] == 1
        assert status['counts']['requests_failed'] == 1
    finally:
        provider.client.close()
