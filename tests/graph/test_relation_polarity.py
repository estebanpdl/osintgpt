'''A field rename preserves evidence, qualifications, request provenance, and paid work.'''

import json
from dataclasses import asdict, replace

import pytest

from osintgpt.graph import Edge, ExtractionOptions, GraphStore, build_graph, graph_for, query_graph
from osintgpt.graph.accounting import ExtractionCounts
from osintgpt.graph.calling import ExtractionCalls
from osintgpt.graph.checkpoints import Checkpoints, edge_from
from osintgpt.graph.contract import EXTRACTION_SCHEMA, legacy_field_request, normalize_saved_extraction, validate_structure
from osintgpt.graph.identity_tools import SUBMIT
from osintgpt.graph.output_policy import policy_data
from osintgpt.graph.recipes import RELATION_FIELD_RENAME, compatible_recipe, extraction_recipe, fingerprint
from osintgpt.graph.state import SCHEMA_VERSION
from osintgpt.llm.generation import GenerationResponse
from osintgpt.prompts import prompt
from test_extraction_quality import extract, proposal


class NoGeneration:
    supports_structured_output = True

    def graph_options(self):
        return {'provider': 'fixture', 'model': 'test'}

    def generate_response(self, *args, **kwargs):
        raise AssertionError('Unexpected new generation')


def legacy_recipe(current):
    return {**current, **{name: pair[0] for name, pair in RELATION_FIELD_RENAME.items()}}


@pytest.mark.parametrize('value', ['positive', 'negative', 'unknown'])
def test_new_contract_preserves_values_and_legacy_records_preserve_ids(value):
    edge = Edge('A', 'B', 'relates to', 'one.md', 'A relates to B.', relation_polarity=value)
    current = asdict(edge)
    old = {**current, 'polarity': current['relation_polarity']}
    del old['relation_polarity']
    assert asdict(Edge.from_dict(old)) == current
    assert edge_from(old).assertion_id == edge.assertion_id
    schema = EXTRACTION_SCHEMA['properties']['edges']['items']
    assert schema['properties']['relation_polarity']['enum'] == ['negative', 'positive', 'unknown']
    assert 'polarity' not in schema['properties']
    with pytest.raises(ValueError, match='Conflicting'):
        Edge.from_dict({**old, 'relation_polarity': 'different'})


def test_existing_graph_read_and_current_export_do_not_rewrite_storage(graph_case):
    case = graph_case
    edge = case.publish('one.md', 'Alpha does not fund Beta.', relation_polarity='negative', assertion='alleged')
    path = case.project.paths.root / 'graph.sqlite'
    with graph_for(case.project) as graph:
        old = json.loads(graph.connection.execute('SELECT data FROM edges').fetchone()[0])
        old['polarity'] = old.pop('relation_polarity')
        raw = json.dumps(old)
        graph.connection.execute('UPDATE edges SET data=?', (raw,))
        graph.connection.commit()
    before = path.read_bytes()
    claim = query_graph(case.project, 'Alpha')['claims'][0]
    assert claim['relation_polarity'] == 'negative' and 'polarity' not in claim
    assert claim['assertion'] == 'alleged' and claim['assertion_id'] == edge.assertion_id
    assert path.read_bytes() == before
    with GraphStore(path, read_only=True) as graph:
        assert graph.connection.execute('SELECT data FROM edges').fetchone()[0] == raw
        assert graph.connection.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION


def test_new_model_output_cannot_silently_drop_the_old_field():
    _, result = extract('Alpha does not fund Beta.', proposal('Alpha does not fund Beta.', polarity='negative'))
    assert not result.edges and 'retired polarity key' in result.rejected[0].reason


def test_recipe_equivalence_is_only_the_reviewed_rename():
    current = extraction_recipe(NoGeneration())
    assert all(current[name] == pair[1] for name, pair in RELATION_FIELD_RENAME.items())
    previous = legacy_recipe(current)
    assert previous != current and compatible_recipe(previous, current)
    for field in ('extraction_prompt', 'response_schema', 'provider', 'evidence_policy'):
        assert not compatible_recipe(previous, {**current, field: 'changed'})


@pytest.mark.parametrize('unfinished', [False, True])
def test_build_reuses_completed_sources_and_old_saved_windows(graph_case, unfinished):
    case = graph_case
    edge = case.publish('one.md', 'Alpha supports Beta.', relation_polarity='positive')
    provider = NoGeneration()
    previous = legacy_recipe(extraction_recipe(provider))
    with graph_for(case.project) as graph:
        graph.state.completed(edge.ref, edge.provenance['source_fingerprint'], edge.provenance['source_version'],
                              recipe=fingerprint(previous), recipe_data=previous)
        if unfinished:
            journal = Checkpoints(graph)
            journal.prepare(edge.ref, edge.provenance['source_version'], fingerprint(previous), previous)
            record = journal.record(edge.ref, edge.ref, (), 1)
            # Staged entities carry source provenance; the bare retrieval fixture does not.
            record.save(1, [replace(entity, provenance=edge.provenance) for entity in graph.entities()], [edge], [])
            data = json.loads(graph.connection.execute('SELECT data FROM graph_windows').fetchone()[0])
            data['edges'][0]['polarity'] = data['edges'][0].pop('relation_polarity')
            graph.connection.execute('UPDATE graph_windows SET data=?', (json.dumps(data),))
            graph.connection.commit()
    report = build_graph(case.project, provider)
    assert report.status == 'complete', report.failed
    assert report.calls == 0
    assert report.accounting.windows_reused == int(unfinished)
    assert report.skipped == int(not unfinished)
    assert query_graph(case.project, 'Alpha')['claims'][0]['assertion_id'] == edge.assertion_id


@pytest.mark.parametrize('submission', [False, True])
def test_saved_old_request_is_recovered_without_changing_its_audit(submission):
    generator = NoGeneration()
    current = extraction_recipe(generator)
    data = {'entities': [], 'edges': [{'source': 'a', 'target': 'b', 'relation': 'denies links to',
             'evidence': ['A denies links to B.'], 'polarity': 'negative', 'assertion': 'stated',
             'attributed_to': '', 'event_time': '', 'statement_time': ''}]}
    reply = (GenerationResponse('', status='completed', stop_reason='tool_calls', tool_calls=(
        {'id': 'submit', 'name': 'submit_extraction', 'arguments': data},), continuation=({'opaque': 'unchanged'},))
        if submission else GenerationResponse(json.dumps(data), status='completed', stop_reason='stop'))
    with GraphStore(':memory:') as graph:
        journal = Checkpoints(graph)
        journal.prepare('one.md', 'version', fingerprint(legacy_recipe(current)), legacy_recipe(current))
        checkpoint = journal.record('one.md', 'one.md', (), 1)
        counts = ExtractionCounts()
        calls = ExtractionCalls(generator, ExtractionOptions(), counts, None, checkpoint)
        calls.position = 1
        system, user = prompt('graph_extraction'), '{"source":{"text":"A denies links to B."}}'
        schema, tools = (None, [SUBMIT]) if submission else (EXTRACTION_SCHEMA, None)
        old_system, old_schema, old_tools = legacy_field_request(system, schema, tools)
        old_key = fingerprint({'request': calls.context.arguments(old_system, user, old_schema, old_tools, None),
                               'output_policy': policy_data(generator)})
        checkpoint.save_response(1, 'extraction', old_key, reply)
        before = [tuple(row) for row in graph.connection.execute('SELECT request_hash,data FROM graph_responses')]
        recovered = calls.generate(system, user, schema, tools=tools, return_response=submission)
        result = recovered.tool_calls[0]['arguments'] if submission else json.loads(recovered)
        validate_structure(result, EXTRACTION_SCHEMA)
        assert result['edges'][0]['relation_polarity'] == 'negative'
        assert counts.responses_recovered == 1 and counts.requests_attempted == 0
        assert [tuple(row) for row in graph.connection.execute('SELECT request_hash,data FROM graph_responses')] == before
        if submission:
            assert recovered.continuation == reply.continuation
        with pytest.raises(AssertionError, match='Unexpected new generation'):
            calls.generate(system, '{"source":{"text":"Different source."}}', schema, tools=tools, return_response=submission)


def test_malformed_and_non_extraction_audits_are_not_rewritten():
    malformed = GenerationResponse('{broken', status='completed')
    assert normalize_saved_extraction(malformed) == malformed
    correction = GenerationResponse('{"corrections":[{"index":0,"evidence":["polarity"]}]}')
    assert normalize_saved_extraction(correction) == correction


def test_saved_fenced_completion_keeps_quotation_wording():
    data = {'entities': [], 'edges': [{'polarity': 'negative', 'evidence': ['The field polarity is quoted.']}]}
    response = GenerationResponse('```json\n' + json.dumps(data) + '\n```')
    normalized = json.loads(normalize_saved_extraction(response).text)
    assert normalized['edges'] == [{'relation_polarity': 'negative', 'evidence': data['edges'][0]['evidence']}]
    assert '"polarity"' in response.text
