'''Renamed fields recover exact saved requests without changing their historical audit.'''

import json
import pytest

from osintgpt.graph import ExtractionOptions, graph_for
from osintgpt.graph.accounting import ExtractionCounts
from osintgpt.graph.calling import ExtractionCalls
from osintgpt.graph.checkpoints import Checkpoints
from osintgpt.graph.contract import EXTRACTION_SCHEMA, REPAIR_SCHEMA, legacy_field_request, normalize_saved_extraction
from osintgpt.graph.extraction import plan_windows
from osintgpt.graph.identity import EntityRegistry
from osintgpt.graph.identity_session import SUBMIT, extract_with_lookup
from osintgpt.graph.migration import migrate_graph
from osintgpt.graph.output_policy import policy_data
from osintgpt.graph.recipes import extraction_recipe, fingerprint
from osintgpt.graph.request_context import RequestContext
from osintgpt.graph.windowing import encode, units
from osintgpt.llm.generation import GenerationResponse
from osintgpt.prompts import prompt
from test_continuation_capacity import search
from test_polarity_migration import downgrade, table_rows
from test_relation_polarity import NoGeneration, legacy_recipe


@pytest.fixture(params=['fixture', 'responses', 'gemini', 'anthropic'])
def saved_provider(request, monkeypatch):
    from tests.llm.test_generation_contract import provider_fixture
    provider = NoGeneration() if request.param == 'fixture' else provider_fixture(request.param)[0]
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery must not issue a new provider request.')
    monkeypatch.setattr(provider, 'generate_response', forbidden)
    try:
        yield provider
    finally:
        if hasattr(provider, 'client'):
            provider.client.close()


@pytest.mark.parametrize('submission', [False, True])
def test_migrated_paid_response_keeps_exact_hash_and_raw_continuation(graph_case, submission, saved_provider):
    project, generator = graph_case.project, saved_provider
    recipe = legacy_recipe(extraction_recipe(generator))
    data = {'entities': [], 'edges': [{'source': 'a', 'target': 'b', 'relation': 'denies links to',
            'polarity': 'negative', 'assertion': 'alleged', 'evidence': ['A denies links to B.']}]}
    reply = (GenerationResponse('', status='completed', tool_calls=(
        {'id': 'submit', 'name': 'submit_extraction', 'arguments': data},),
        continuation=({'opaque': '原始 polarity'},))
        if submission else GenerationResponse(json.dumps(data), status='completed'))
    system, user = prompt('graph_extraction'), '{"source":{"text":"A denies links to B."}}'
    schema, tools = (None, [SUBMIT]) if submission else (EXTRACTION_SCHEMA, None)
    with graph_for(project) as graph:
        journal = Checkpoints(graph)
        journal.prepare('one.md', 'v1', fingerprint(recipe), recipe)
        record = journal.record('one.md', 'one.md', (), 1)
        calls = ExtractionCalls(generator, ExtractionOptions(), ExtractionCounts(), None, record)
        old_system, old_schema, old_tools = legacy_field_request(system, schema, tools)
        key = fingerprint({'request': calls.context.arguments(old_system, user, old_schema, old_tools, None),
                           'output_policy': policy_data(generator)})
        record.save_response(1, 'extraction', key, reply)
    path = downgrade(project)
    raw = table_rows(path)['graph_responses']
    migrate_graph(project)
    with graph_for(project) as graph:
        record = Checkpoints(graph).record('one.md', 'one.md', (), 1)
        counts = ExtractionCounts()
        # Recovery does not need to fit the renamed request into a smaller capacity.
        calls = ExtractionCalls(generator, ExtractionOptions(model_input_units=1), counts, None, record)
        calls.position = 1
        assert calls.has_saved_response(system, user, schema, tools=tools)
        recovered = calls.generate(system, user, schema, tools=tools, return_response=True)
        normalized = recovered.tool_calls[0]['arguments'] if submission else json.loads(recovered.text)
        assert normalized['edges'][0]['relation_polarity'] == 'negative'
        assert 'polarity' not in normalized['edges'][0]
        assert recovered.continuation == reply.continuation
        assert counts.requests_attempted == 0 and counts.responses_recovered == 1
        assert not calls.has_saved_response(system, 'changed source', schema, tools=tools)
        assert not calls.has_saved_response(system + ' Additional extraction rule.', user, schema, tools=tools)
    assert table_rows(path)['graph_responses'] == raw


def test_migrated_repair_request_recovers_only_structured_candidate_field(graph_case):
    project, provider = graph_case.project, NoGeneration()
    recipe = legacy_recipe(extraction_recipe(provider))
    old_candidate = {'source': 'a', 'polarity': 'negative', 'target': 'b',
                     'evidence': ['The literal "relation_polarity" is quoted.']}
    old_payload = {'source': {'text': 'The literal "relation_polarity" is quoted.'},
                   'candidates': [{'index': 0, 'candidate': old_candidate}]}
    normalized = normalize_saved_extraction(GenerationResponse(encode({'edges': [old_candidate]})))
    candidate = json.loads(normalized.text)['edges'][0]
    payload = {**old_payload, 'candidates': [{'index': 0, 'candidate': candidate}]}
    response = GenerationResponse('{"corrections":[{"index":0,"evidence":["original quotation"]}]}', status='completed')
    with graph_for(project) as graph:
        journal = Checkpoints(graph)
        journal.prepare('one.md', 'v1', fingerprint(recipe), recipe)
        record = journal.record('one.md', 'one.md', (), 1)
        calls = ExtractionCalls(provider, ExtractionOptions(), ExtractionCounts(), None, record)
        old_system, schema, _ = legacy_field_request(prompt('graph_quote_repair'), REPAIR_SCHEMA, None)
        key = fingerprint({'request': calls.context.arguments(old_system, encode(old_payload), schema, None, None),
                           'output_policy': policy_data(provider)})
        record.save_response(1, 'repair', key, response)
    path = downgrade(project)
    original = table_rows(path)['graph_responses']
    migrate_graph(project)
    with graph_for(project) as graph:
        record = Checkpoints(graph).record('one.md', 'one.md', (), 1)
        calls = ExtractionCalls(provider, ExtractionOptions(model_input_units=1), ExtractionCounts(), None, record)
        calls.position = 1
        assert calls.fits(prompt('graph_quote_repair'), encode(payload), REPAIR_SCHEMA, purpose='repair')
        assert calls.generate(prompt('graph_quote_repair'), encode(payload), REPAIR_SCHEMA, purpose='repair') == response.text
        assert calls.counts.responses_recovered == 1 and not calls.counts.requests_attempted
    assert table_rows(path)['graph_responses'] == original


@pytest.mark.parametrize('version', [4, 5])
def test_rename_keeps_capacity_sensitive_source_boundaries(monkeypatch, version):
    provider = NoGeneration()
    recipe = legacy_recipe(extraction_recipe(provider))
    recipe['version'] = version
    recipe['options']['context_units'] = 4096
    text = '甲乙丙丁' * 500
    current_system = prompt('graph_extraction')
    old_system, old_schema, _ = legacy_field_request(current_system, EXTRACTION_SCHEMA, None)
    envelope = {'document_ref': 'source.md', 'source_version': 'v1',
                'source': {'start': 0, 'end': len(text), 'text': ''}, 'known_entities': []}
    overhead = (512 + units(old_system) + units(encode(envelope)) + 4096 + units(encode(old_schema))
                if version == 4 else RequestContext(provider, ExtractionOptions()).measure(old_system, encode(envelope), old_schema))
    options = ExtractionOptions(context_budget=40000, model_input_units=overhead + units(encode(text)))
    with monkeypatch.context() as patch:
        patch.setattr('osintgpt.graph.extraction.prompt', lambda _: old_system)
        _, _, expected = plan_windows('source.md', text, 'v1', options, old_schema, legacy=recipe,
                                      request_context=RequestContext(provider, options))
    system, _, actual = plan_windows('source.md', text, 'v1', options, EXTRACTION_SCHEMA, legacy=recipe,
                                     request_context=RequestContext(provider, options))
    assert system == current_system  # New requests continue to use the current contract.
    assert actual == expected and len(actual) == 1
    # Without compatibility sizing, the longer instructions split this paid window.
    current_recipe = extraction_recipe(provider)
    current_recipe.update(version=version, options=recipe['options'])
    _, _, fresh = plan_windows('source.md', text, 'v1', options, EXTRACTION_SCHEMA, legacy=current_recipe,
                               request_context=RequestContext(provider, options))
    assert fresh != expected


def test_migrated_lookup_sequence_reuses_paid_turns_before_context_restart(graph_case, monkeypatch):
    import osintgpt.graph.identity_session as session

    class Model(NoGeneration):
        def __init__(self):
            self.responses = iter([
                GenerationResponse('', status='completed', tool_calls=(search('Álpha'),),
                                   continuation=({'opaque': 'original lookup'},)),
                GenerationResponse('{"entities":[],"edges":[]}', status='completed'),
            ])
            self.requests = 0

        def generate_response(self, system, user, *, tools=None, history=None, **kwargs):
            self.requests += 1
            return next(self.responses)

    project, model = graph_case.project, Model()
    recipe = legacy_recipe(extraction_recipe(model))
    system = prompt('graph_extraction')
    old_system, old_schema, _ = legacy_field_request(system, session.RESOLVING_SCHEMA, None)
    payload = {'document_ref': 'one.md', 'source_version': 'v1', 'known_entities': [],
               'source': {'start': 0, 'end': 15, 'text': 'Álpha existe.'}}
    with graph_for(project) as graph:
        journal = Checkpoints(graph)
        journal.prepare('one.md', 'v1', fingerprint(recipe), recipe)
        record = journal.record('one.md', 'one.md', (), 1)
        calls = ExtractionCalls(model, ExtractionOptions(), ExtractionCounts(), None, record)
        calls.position = 1
        with monkeypatch.context() as patch:
            patch.setattr(session, 'RESOLVING_SCHEMA', old_schema)
            extract_with_lookup(calls, old_system, payload, EntityRegistry())
        assert model.requests == 2
    path = downgrade(project)
    original = sorted((row[3], row[4], json.loads(row[5])['text']) for row in table_rows(path)['graph_responses'])
    migrate_graph(project)
    with graph_for(project) as graph:
        record = Checkpoints(graph).record('one.md', 'one.md', (), 1)
        counts = ExtractionCounts()
        calls = ExtractionCalls(model, ExtractionOptions(model_input_units=1), counts, None, record)
        calls.position = 1
        result, _ = extract_with_lookup(calls, system, payload, EntityRegistry())
        assert json.loads(result) == {'entities': [], 'edges': []}
        assert counts.responses_recovered == 2 and counts.entity_lookups_reused == 1
        assert counts.requests_attempted == counts.context_restarts == 0 and model.requests == 2
    assert sorted((row[3], row[4], json.loads(row[5])['text']) for row in table_rows(path)['graph_responses']) == original
