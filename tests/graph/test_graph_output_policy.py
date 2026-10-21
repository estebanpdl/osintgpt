'''Complete outputs, automatic model limits, and compatibility without paid calls.'''

from dataclasses import asdict
import json
import warnings

import pytest

from osintgpt.graph import ExtractionOptions, build_graph, graph_for, preflight_graph, saved_graph_status
from osintgpt.graph.export import to_json
from osintgpt.graph.grounding import parse_response
from osintgpt.graph.contract import REPAIR_SCHEMA
from osintgpt.graph.output_policy import OutputPolicy
from osintgpt.graph.run_settings import GraphRunSettings
from osintgpt.llm.generation_pacing import GenerationLimits
from osintgpt.rate_settings import RateLimits
from test_build_status import project
from test_extraction_contract import StructuredScript, complete_proposal
from tests.llm.test_generation_contract import provider_fixture


@pytest.mark.parametrize('kind', ['responses', 'gemini', 'anthropic', 'ollama'])
def test_graph_native_parameters_and_publication(project, kind):
    provider, requests = provider_fixture(kind)
    try:
        preview = preflight_graph(project, provider)
        assert requests == []
        assert preview['budget_cost_estimate_usd'] is None or kind == 'ollama'
        report = build_graph(project, provider)
        assert report.status == 'complete' and report.calls == 1
        request = requests[0]
        if kind == 'anthropic':
            assert request['max_tokens'] == 64000 and request['stream'] is True
            assert report.usage['output_policy']['max_tokens'] == 64000
        else:
            assert 'max_tokens' not in request and 'max_output_tokens' not in request
        assert report.usage['total_tokens'] == 15
        second = build_graph(project, provider)
        assert second.skipped == 1 and len(requests) == 1
    finally:
        provider.client.close()


def test_model_allowance_does_not_consume_application_input_budget():
    policy = OutputPolicy('model_maximum', 128000, 1000000, 'model')
    options = policy.options(ExtractionOptions())
    assert options.input_units == 872000
    assert options.context_budget == 32768
    narrow = OutputPolicy('model_maximum', 64000, 80000, 'narrow').options(ExtractionOptions())
    assert narrow.input_units == 16000
    with pytest.raises(ValueError, match='insufficient context'):
        OutputPolicy('model_maximum', 64000, 64000).options(ExtractionOptions())


def test_retired_options_and_saved_settings(project):
    with pytest.raises(ValueError, match='removed'):
        ExtractionOptions(output_tokens=4096)
    with pytest.raises(ValueError, match='removed'):
        ExtractionOptions(32768, 4096, 1024, 500, True)
    path = project.paths.root / 'graph-build.toml'
    path.write_text('output_tokens = 4096\ncontext_budget = 24000\n')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        for _ in range(3):
            settings = GraphRunSettings.load(project)
    assert settings.context_budget == 24000
    assert 'output_tokens' not in asdict(settings.extraction())
    settings.save(project)
    assert 'output_tokens' not in path.read_text()


def large_proposal():
    sentences = [f'企业{i} 支持 企业{i+1}。' for i in range(205)]
    entities = [{'id': f'e{i}', 'name': f'企业{i}', 'type': '组织', 'description': '说明' * 600,
                 'existing_id': '', 'aliases': [], 'identifiers': [],
                 'evidence': [sentences[min(i, 204)]]} for i in range(206)]
    edges = [{'source': f'e{i}', 'target': f'e{i+1}', 'relation': '支持', 'evidence': [quote],
              'relation_polarity': 'positive', 'assertion': 'stated', 'attributed_to': '', 'event_time': '',
              'statement_time': ''} for i, quote in enumerate(sentences)]
    return '\n'.join(sentences), {'entities': entities, 'edges': edges}


def test_large_multilingual_valid_response_survives_publication_export_and_reuse(project):
    text, proposal = large_proposal()
    assert len(json.dumps(proposal, ensure_ascii=False).encode()) > 262144
    (project.paths.root / 'material/a.md').write_text(text, encoding='utf-8')
    generator = StructuredScript(proposal)
    report = build_graph(project, generator)
    assert report.status == 'complete'
    assert report.entities == 206 and report.edges == 205
    assert report.accounting.invalid_responses == report.accounting.rejected_claims == 0
    with graph_for(project) as graph:
        exported = json.loads(to_json(graph))
    assert len(exported['entities']) == 206 and len(exported['edges']) == 205
    assert len(exported['entities'][0]['description']) == 1200
    second = build_graph(project, generator)
    assert second.skipped == 1 and second.calls == 0


def test_correction_parser_has_no_unadvertised_array_or_quote_limits():
    response = {'corrections': [{'index': i, 'evidence': ['证据' * 2400]} for i in range(205)]}
    assert parse_response(json.dumps(response), 'corrections', schema=REPAIR_SCHEMA) == response


def test_supported_aliases_identifiers_and_quotations_have_no_hidden_count_limits(project):
    quote = 'Alpha supports Beta. ' + ' '.join(f'Alias{i} Issuer{i} Value{i}.' for i in range(15))
    (project.paths.root / 'material/a.md').write_text(quote)
    proposal = complete_proposal(quote)
    entity = proposal['entities'][0]
    entity['aliases'] = [{'name': f'Alias{i}', 'evidence': [quote]} for i in range(15)]
    entity['identifiers'] = [{'issuer': f'Issuer{i}', 'value': f'Value{i}', 'evidence': [quote]} for i in range(15)]
    report = build_graph(project, StructuredScript(proposal))
    assert report.status == 'complete' and report.edges == 1
    assert len(report.extracted[0].entities[0].aliases) == 15


def test_per_minute_budget_does_not_silently_reduce_provider_allowance(project):
    provider, requests = provider_fixture('anthropic')
    try:
        report = build_graph(project, provider, request_limits=GenerationLimits(RateLimits(tpm=100)))
        assert report.status == 'failed' and 'budget' in report.failed[0].problem.lower()
        assert requests == []
        assert saved_graph_status(project)['counts']['requests_attempted'] == 0
    finally:
        provider.client.close()


def test_provider_default_reservation_is_only_a_pacing_estimate():
    assert OutputPolicy().reservation(2000) == 4512
    assert OutputPolicy('model_maximum', 64000, 200000).reservation(2000) == 66512
