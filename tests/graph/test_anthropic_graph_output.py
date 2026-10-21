'''Model discovery, streaming, and recovery use native SDK serialization offline.'''

import json

import httpx2
import pytest
from anthropic import Anthropic

from osintgpt.graph import build_graph, preflight_graph, saved_graph_status
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.provider import LazyGraphGenerator
from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.generation_pacing import GenerationRequestError
from osintgpt.llm.usage import UsageRecorder
from osintgpt.config import Settings
from tests.conftest import FAKE_KEY
from tests.llm.anthropic_fixtures import message_stream, model_metadata
from test_build_status import project
from test_extraction_contract import complete_proposal


def native(*, maximum=128000, context=1000000, complete=True, reason='end_turn', repair=False, discovery_failure=None):
    calls, discovery, timeouts = [], [], []

    def respond(request):
        if request.method == 'GET':
            discovery.append(str(request.url))
            if discovery_failure:
                return httpx2.Response(discovery_failure, json={'type': 'error',
                    'error': {'type': 'invalid_request_error', 'message': 'Model metadata unavailable'}})
            return httpx2.Response(200, json=model_metadata('resolved-model', maximum, context))
        data = json.loads(request.content)
        calls.append(data)
        timeouts.append(request.extensions.get('timeout'))
        proposal = complete_proposal('Invented.' if repair else 'Alpha supports Beta.')
        for entity in proposal['entities']:
            entity['resolution'] = {'action': 'create', 'method': 'new', 'reason': 'New sourced entity.',
                                    'evidence': [], 'candidate_evidence': [], 'type_compatible': False, 'type_reason': ''}
        if len(calls) == 2:
            proposal = {'corrections': [{'index': 0, 'evidence': ['Alpha supports Beta.']}]}
        body = {'id': 'msg_fixture', 'type': 'message', 'model': 'resolved-model', 'role': 'assistant',
                'content': [{'type': 'text', 'text': json.dumps(proposal)}], 'stop_sequence': None,
                'stop_reason': reason, 'usage': {'input_tokens': 30, 'output_tokens': 20}}
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'},
                               content=message_stream(body, complete=complete))

    client = Anthropic(api_key=FAKE_KEY, http_client=httpx2.Client(transport=httpx2.MockTransport(respond)))
    provider = AnthropicGeneration('model-alias', FAKE_KEY, client=client, recorder=UsageRecorder())
    return provider, calls, discovery, timeouts


@pytest.mark.parametrize('maximum', [32000, 64000, 128000])
def test_graph_limits_and_quote_repair_use_streaming_and_final_usage(project, maximum):
    provider, calls, discovery, timeouts = native(maximum=maximum, repair=True)
    try:
        preview = preflight_graph(project, provider)
        assert not discovery and not calls
        assert preview['output_policy']['max_tokens'] is None
        report = build_graph(project, provider)
        assert report.status == 'complete' and report.edges == 1
        assert len(discovery) == 1 and len(calls) == 2
        assert all(call['max_tokens'] == maximum and call['stream'] for call in calls)
        assert report.usage['total_tokens'] == 100 and report.accounting.repaired_claims == 1
        assert all(timeout['read'] == 600 for timeout in timeouts)
        assert report.usage['output_policy']['resolved_model'] == 'resolved-model'
    finally:
        provider.client.close()


@pytest.mark.parametrize('maximum,context', [(None, None), (0, 200000), (64000, None), (64000, 60000)])
def test_missing_or_impossible_model_limits_fail_before_generation(project, maximum, context):
    provider, calls, _, _ = native(maximum=maximum, context=context)
    try:
        with pytest.raises(ValueError):
            build_graph(project, provider)
        assert calls == []
    finally:
        provider.client.close()


def test_broken_stream_is_not_a_reusable_or_published_response(project):
    provider, calls, _, _ = native(complete=False)
    try:
        with pytest.raises(GenerationRequestError, match='message_stop'):
            build_graph(project, provider)
        status = saved_graph_status(project)
        assert len(calls) == status['counts']['requests_failed'] == 1
        assert status['counts']['requests_succeeded'] == status['checkpoints']['responses_saved'] == 0
        assert status['usage']['missing_usage_requests'] == 1
        assert not status['usage']['tokens_complete'] and status['entities'] == 0
    finally:
        provider.client.close()


@pytest.mark.parametrize('reason', ['max_tokens', 'refusal'])
def test_provider_stop_reason_is_retained_even_when_json_is_valid(project, reason):
    provider, _, _, _ = native(reason=reason)
    try:
        report = build_graph(project, provider)
        assert report.status == 'failed' and report.accounting.invalid_responses == 1
        assert report.usage['total_tokens'] == 50 and report.edges == 0
    finally:
        provider.client.close()


def test_saved_limits_allow_keyless_preview_and_unchanged_build(project):
    provider, calls, _, _ = native()
    try:
        assert build_graph(project, provider).status == 'complete'
        fresh, _, _, _ = native()
        try:
            fresh.graph_model_limits = lambda: pytest.fail('Must reuse exact provider/model metadata')
            fresh.client.close()
            assert preflight_graph(project, fresh)['counts']['documents_unchanged'] == 1
            assert build_graph(project, fresh).skipped == 1
        finally:
            fresh.client.close()
        assert len(calls) == 1
    finally:
        provider.client.close()


def test_cancelled_stream_is_not_published_or_reused(project):
    provider, calls, _, _ = native()
    try:
        with pytest.raises(GraphCancelled):
            build_graph(project, provider, should_cancel=lambda: bool(calls))
        status = saved_graph_status(project)
        assert status['status'] == 'interrupted'
        assert status['counts']['requests_interrupted'] == 1
        assert status['counts']['requests_succeeded'] == status['checkpoints']['responses_saved'] == 0
        assert status['entities'] == 0 and not status['usage']['tokens_complete']
    finally:
        provider.client.close()


def test_lazy_graph_provider_discovers_once_and_forwards_automatic_allowance(project):
    provider, calls, discovery, _ = native()
    factories = []
    def factory(*args, **kwargs):
        factories.append(args)
        return provider
    lazy = LazyGraphGenerator(factory, project.settings, Settings(), provider.recorder,
                              provider='anthropic', model='model-alias')
    try:
        assert not preflight_graph(project, lazy)['model_context_verified']
        assert not factories and not discovery
        assert build_graph(project, lazy).status == 'complete'
        assert len(factories) == len(discovery) == len(calls) == 1
        assert calls[0]['max_tokens'] == 128000 and calls[0]['stream'] is True
        fresh = LazyGraphGenerator(lambda *a, **k: pytest.fail('No provider needed'),
                                   project.settings, Settings(), UsageRecorder(),
                                   provider='anthropic', model='model-alias')
        assert preflight_graph(project, fresh)['model_context_verified']
        assert build_graph(project, fresh).skipped == 1
    finally:
        provider.client.close()


def test_changed_model_cannot_reuse_cached_capabilities(project):
    from osintgpt.graph.output_policy import prepare_policy
    provider, _, discovery, _ = native()
    try:
        prepare_policy(provider, resolve=True)
        provider.model = 'other-model'
        assert not prepare_policy(provider).resolved
        prepare_policy(provider, resolve=True)
        assert len(discovery) == 2 and discovery[-1].endswith('/other-model')
    finally:
        provider.client.close()


@pytest.mark.parametrize('status', [401, 404, 429, 500])
def test_metadata_http_error_is_actionable_without_generation(project, status):
    provider, calls, discovery, _ = native(discovery_failure=status)
    try:
        with pytest.raises(ValueError, match='No graph generation request was sent'):
            build_graph(project, provider)
        assert not calls and len(discovery) == 1 and provider.recorder.calls == 0
    finally:
        provider.client.close()
