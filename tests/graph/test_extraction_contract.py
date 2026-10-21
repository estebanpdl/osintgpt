'''Completion signals and native schemas never replace local evidence checks.'''

import json
from dataclasses import replace

import pytest

from osintgpt.graph import ExtractionOptions, build_graph, extract_document, graph_for, saved_graph_status
from osintgpt.graph.contract import EXTRACTION_SCHEMA, REPAIR_SCHEMA
from osintgpt.graph.grounding import parse_response
from osintgpt.graph.recipes import extraction_recipe, fingerprint
from osintgpt.graph.windowing import encode
from osintgpt.llm.generation import GenerationResponse
from test_build_status import project
from test_extraction_quality import Script, proposal

EMPTY = {'entities': [], 'edges': []}


class StructuredScript(Script):
    supports_structured_output = True

    def __init__(self, *replies, response=None):
        super().__init__(*replies)
        self.response = response or GenerationResponse('', status='completed')
        self.schemas = []

    def generate_response(self, system, user, *, max_output_tokens=None, schema=None):
        self.schemas.append(schema)
        return replace(self.response, text=self.generate(system, user, max_output_tokens=max_output_tokens))


def complete_proposal(quote):
    data = proposal(quote)
    for entity in data['entities']:
        entity.update(type='', description='', existing_id='', aliases=[], identifiers=[])
    data['edges'][0].update(evidence=[quote], relation_polarity='unknown', assertion='unknown',
                            attributed_to='', event_time='', statement_time='')
    return data


def test_schema_is_sent_on_extraction_and_quote_correction():
    generator = StructuredScript(complete_proposal('Invented.'),
                                 {'corrections': [{'index': 0, 'evidence': ['Alpha funds Beta.']}]})
    result = extract_document(generator, 'one.md', 'Alpha funds Beta.')
    assert result.ok and len(result.edges) == 1
    assert result.edges[0].evidence == 'Alpha funds Beta.'
    assert generator.schemas == [EXTRACTION_SCHEMA, REPAIR_SCHEMA]
    assert result.accounting.structured_output_unenforced == 0
    assert result.accounting.completion_status_unavailable == 0


@pytest.mark.parametrize('response,problem', [
    (GenerationResponse('', status='incomplete', stop_reason='max_output_tokens'), 'truncated'),
    (GenerationResponse('', stop_reason='length'), 'truncated'),
    (GenerationResponse('', stop_reason='max_tokens'), 'truncated'),
    (GenerationResponse('', stop_reason='model_context_window_exceeded'), 'truncated'),
    (GenerationResponse('', refused=True), 'refused'),
    (GenerationResponse('', stop_reason='content_filter'), 'without completion'),
    (GenerationResponse('', status='failed'), 'not complete'),
])
def test_valid_json_cannot_override_incomplete_or_refused_response(project, response, problem):
    generator = StructuredScript(EMPTY, response=response)
    report = build_graph(project, generator)
    assert report.status == 'failed' and problem in report.failed[0].problem
    assert report.accounting.requests_succeeded == report.accounting.invalid_responses == 1
    assert report.accounting.requests_failed == report.accounting.windows_completed == 0
    status = saved_graph_status(project)
    assert not status['available'] and status['checkpoints']['windows_saved'] == 0


def test_incomplete_replacement_preserves_published_graph(project):
    generator = StructuredScript(complete_proposal('Alpha supports Beta.'))
    assert build_graph(project, generator).status == 'complete'
    with graph_for(project) as graph:
        original = list(graph.edges())
    stopped = StructuredScript(EMPTY, response=GenerationResponse('', stop_reason='length'))
    report = build_graph(project, stopped, rebuild=True)
    assert report.failed and report.accounting.windows_completed == 0
    with graph_for(project) as graph:
        assert list(graph.edges()) == original


def test_truncated_correction_is_audited_without_publishing_it():
    generator = StructuredScript(complete_proposal('Invented.'),
                                 {'corrections': [{'index': 0, 'evidence': ['Alpha funds Beta.']}]})
    generate = generator.generate_response
    def respond(*args, **kwargs):
        reply = generate(*args, **kwargs)
        return replace(reply, stop_reason='length') if len(generator.calls) == 2 else reply
    generator.generate_response = respond
    result = extract_document(generator, 'one.md', 'Alpha funds Beta.')
    assert result.ok and not result.edges
    assert 'truncated' in result.rejected[0].repair_problem
    assert result.accounting.requests_succeeded == 2 and result.accounting.invalid_responses == 1


def test_unknown_completion_and_schema_support_are_visible_for_legacy_provider():
    result = extract_document(Script(EMPTY), 'one.md', 'Source.')
    assert result.ok
    assert result.accounting.structured_output_unenforced == 1
    assert result.accounting.completion_status_unavailable == 1


def test_native_schema_is_validated_locally_even_when_provider_ignores_it():
    generator = StructuredScript(proposal('Alpha funds Beta.'))
    result = extract_document(generator, 'one.md', 'Alpha funds Beta.')
    assert not result.ok and 'schema' in result.problem
    assert result.accounting.invalid_responses == 2 and not result.edges


def test_native_schema_does_not_bypass_quote_validation():
    result = extract_document(StructuredScript(complete_proposal('Invented.')), 'one.md', 'Alpha funds Beta.',
                              options=ExtractionOptions(repair_quotes=False))
    assert result.ok and not result.edges and result.accounting.rejected_claims == 1


@pytest.mark.parametrize('data', [
    {'corrections': [{'index': True, 'evidence': ['real']}]},
    {'corrections': [{'index': 0, 'evidence': 'real'}]},
    {'corrections': [{'index': 0, 'evidence': ['real'], 'relation': 'changed'}]},
])
def test_repair_schema_rejects_wrong_types_and_unrequested_fields(data):
    with pytest.raises(ValueError):
        parse_response(json.dumps(data), 'corrections', schema=REPAIR_SCHEMA)


def test_schema_is_included_in_every_multilingual_request_budget():
    generator = StructuredScript(EMPTY)
    options = ExtractionOptions(context_budget=10000, context_units=500)
    text = '甲公司没有资助乙公司。' * 2000
    result = extract_document(generator, 'long.md', text, options=options)
    assert result.ok and len(generator.calls) > 1
    for system, payload, cap in generator.calls:
        assert options.fits(system, encode(payload), EXTRACTION_SCHEMA)
        assert cap is None


def test_schema_support_and_window_policy_change_checkpoint_recipe(monkeypatch):
    generator = StructuredScript(EMPTY)
    original = extraction_recipe(generator)
    generator.supports_structured_output = False
    assert fingerprint(extraction_recipe(generator)) != fingerprint(original)
    generator.supports_structured_output = True
    monkeypatch.setattr('osintgpt.graph.recipes.WINDOWING_VERSION', 999)
    assert fingerprint(extraction_recipe(generator)) != fingerprint(original)
