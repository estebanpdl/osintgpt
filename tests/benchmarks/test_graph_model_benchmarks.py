'''Model-role routing, saved evidence, and one shared budget through local native transports.'''

import json
from copy import deepcopy

import pytest

from osintgpt.llm.usage import Usage, UsageRecorder
from graph_answer_evidence import validate_bundle
from graph_answers import run
from graph_offline import RecordedFixtureEmbedder, generation

FIXTURE = {'documents': [{'ref': 'a.md', 'text': 'Alpha funds Beta.', 'source': 'Alpha', 'target': 'Beta', 'relation': 'funds'}],
           'questions': [{'name': 'funding', 'query': 'Who funds Beta?', 'terms': ['Alpha'], 'entity': 'Alpha',
                          'target': 'Beta', 'direction': 'outgoing', 'expected': ['a.md']}]}


def providers(recorder, answer='answer-A', extract='extract-B'):
    return (generation(FIXTURE, recorder, role='answering', model=answer),
            generation(FIXTURE, recorder, role='extraction', model=extract), RecordedFixtureEmbedder(recorder))


@pytest.fixture
def prepared(tmp_path):
    recorder = UsageRecorder()
    answerer, extractor, embedder = providers(recorder)
    try:
        output = tmp_path / 'prepared.json'
        report = run(FIXTURE, answerer, embedder, recorder, extractor=extractor, output=output, execution='offline_transport')
        assert report['status'] == 'complete_pending_review', report.get('problem')
        assert json.loads(output.read_text(encoding='utf8')) == report
        yield report
    finally:
        answerer.client.close()
        extractor.client.close()


def test_native_extraction_and_answering_receive_distinct_models_and_report_usage(prepared):
    roles = prepared['usage_by_role']
    assert roles['extraction']['calls'] == 1 and roles['answering']['calls'] == 2 and roles['embedding']['calls'] >= 2
    assert {row['model'] for row in roles['extraction']['records']} == {'extract-B'}
    assert {row['model'] for row in roles['answering']['records']} == {'answer-A'}
    assert all(row['answering_model']['model'] == 'answer-A' for row in prepared['answers'])
    assert prepared['calls'] == sum(row['calls'] for row in roles.values())
    assert prepared['actual_provider_spend_usd'] == 0
    assert prepared['estimated_cost_usd'] is None and not prepared['cost_complete']
    assert prepared['evidence_bundle']['graph_snapshot']['edges']
    assert prepared['evidence_bundle']['extraction_recipe']['provider']['model'] == 'extract-B'


def test_answer_model_comparison_reuses_identical_graph_and_inputs_without_other_calls(prepared, monkeypatch):
    recorder = UsageRecorder()
    answerer = generation(FIXTURE, recorder, role='answering', model='answer-C')
    monkeypatch.setattr('graph_answers.prepare', lambda *a, **k: pytest.fail('Reused evidence must not prepare again'))
    try:
        report = run(FIXTURE, answerer, None, recorder, evidence=prepared['evidence_bundle'])
    finally:
        answerer.client.close()
    assert report['status'] == 'complete_pending_review'
    assert report['evidence_reused'] and report['evidence_sha256'] == prepared['evidence_sha256']
    assert report['usage_by_role']['extraction']['calls'] == report['usage_by_role']['embedding']['calls'] == 0
    assert report['models']['extraction']['model'] == 'extract-B'
    assert report['models']['answering']['model'] == 'answer-C'
    for first, second in zip(prepared['answers'], report['answers']):
        assert (first['context'], first['system_input'], first['question']) == (second['context'], second['system_input'], second['question'])


def test_extraction_comparison_holds_answering_fixture_and_retrieval_controls_fixed(prepared):
    recorder = UsageRecorder()
    answerer, extractor, embedder = providers(recorder, extract='extract-D')
    try:
        report = run(FIXTURE, answerer, embedder, recorder, extractor=extractor)
    finally:
        answerer.client.close()
        extractor.client.close()
    assert report['status'] == 'complete_pending_review'
    assert report['models']['answering'] == prepared['models']['answering']
    assert report['comparison_controls'] == prepared['comparison_controls']
    assert report['models']['extraction']['model'] == 'extract-D'
    assert report['evidence_sha256'] != prepared['evidence_sha256']


@pytest.mark.parametrize('change', ['contents', 'fixture', 'budget', 'output_budget'])
def test_reuse_rejects_changed_evidence_or_controls_before_answering(prepared, change):
    bundle, fixture, budget = deepcopy(prepared['evidence_bundle']), deepcopy(FIXTURE), 2048
    output_tokens = 1024
    if change == 'contents':
        bundle['rows'][0]['context']['passages'][0]['text'] = 'Changed text'
    elif change == 'fixture':
        fixture['questions'][0]['terms'] = ['Beta']
    elif change == 'budget':
        budget += 1
    else:
        output_tokens += 1
    with pytest.raises(ValueError):
        validate_bundle(bundle, fixture, budget, output_tokens)


def test_one_cost_ceiling_covers_extraction_and_answering_and_saves_stopped_usage(tmp_path, monkeypatch):
    monkeypatch.setattr('osintgpt.llm.usage.estimate_cost', lambda *a, **k: 0.01)
    recorder = UsageRecorder(cost_ceiling_usd=0.03)
    answerer, extractor, embedder = providers(recorder)
    try:
        path = tmp_path / 'stopped.json'
        report = run(FIXTURE, answerer, embedder, recorder, extractor=extractor, output=path)
    finally:
        answerer.client.close()
        extractor.client.close()
    assert report['status'] == 'cost_stopped'
    assert report['usage_by_role']['extraction']['calls'] == report['usage_by_role']['answering']['calls'] == 1
    assert report['usage_by_role']['extraction']['estimated_cost_usd'] == 0.02
    assert report['usage_by_role']['answering']['estimated_cost_usd'] == 0.02
    assert report['budget_usage']['estimated_cost_usd'] == 0.04
    assert json.loads(path.read_text(encoding='utf8'))['status'] == 'cost_stopped'
    assert report['evidence_bundle'] and not report['answers']


@pytest.mark.parametrize('owner', ['separate', 'missing'])
def test_separate_recorders_cannot_bypass_shared_ceiling(owner):
    recorder = UsageRecorder(cost_ceiling_usd=0.03)
    answerer, extractor, embedder = providers(recorder)
    extractor.recorder = UsageRecorder() if owner == 'separate' else None
    try:
        with pytest.raises(ValueError, match='share one usage recorder'):
            run(FIXTURE, answerer, embedder, recorder, extractor=extractor)
    finally:
        answerer.client.close()
        extractor.client.close()


@pytest.mark.parametrize('stage', ['embedding', 'extraction'])
def test_preparation_cost_stop_prevents_answer_calls_and_retains_role_usage(tmp_path, monkeypatch, stage):
    monkeypatch.setattr('osintgpt.llm.usage.estimate_cost', lambda *a, **k: 0.01)
    recorder = UsageRecorder(cost_ceiling_usd=0.001)
    answerer, extractor, embedder = providers(recorder)
    if stage == 'embedding':
        monkeypatch.setattr(embedder, 'embed', lambda texts, **kwargs: recorder.record(Usage('offline', 'embedding', 10)))
    try:
        report = run(FIXTURE, answerer, embedder, recorder, extractor=extractor, output=tmp_path / 'stopped.json')
    finally:
        answerer.client.close()
        extractor.client.close()
    assert report['status'] == 'cost_stopped'
    assert not report['answers'] and report['usage_by_role']['answering']['calls'] == 0
    assert report['usage_by_role'][stage]['calls'] == 1
    if stage == 'extraction':
        assert report['graph_build']['status'] == 'interrupted'


@pytest.mark.parametrize('response,status', [
    ({'entities': [{'id': 'unknown', 'name': 'Invented', 'evidence': ['Invented'], 'type': '',
                    'description': '', 'aliases': [], 'identifiers': [], 'existing_id': ''}], 'edges': []}, 'complete_with_gaps'),
    ([], 'failed'),
])
def test_rejected_proposals_remain_visible_but_failed_processing_stops_comparison(monkeypatch, response, status):
    monkeypatch.setattr('graph_offline.fixture_proposal', lambda *a: response)
    recorder = UsageRecorder()
    answerer, extractor, embedder = providers(recorder)
    try:
        report = run(FIXTURE, answerer, embedder, recorder, extractor=extractor)
    finally:
        answerer.client.close()
        extractor.client.close()
    assert report['graph_build']['status'] == status
    if status == 'complete_with_gaps':
        assert report['graph_build']['counts']['rejected_entities'] == 1
        assert report['status'] == 'complete_pending_review' and len(report['answers']) == 2
        validate_bundle(report['evidence_bundle'], FIXTURE, 2048)
    else:
        assert report['status'] == 'failed' and not report['answers']
        assert report['usage_by_role']['answering']['calls'] == 0


def test_offline_cli_does_not_construct_live_providers(tmp_path, monkeypatch):
    import graph_answers
    monkeypatch.setattr(graph_answers, 'build_generation_provider', lambda *a, **k: pytest.fail('No live client'))
    monkeypatch.setattr(graph_answers, 'build_embedding_provider', lambda *a, **k: pytest.fail('No live client'))
    output = tmp_path / 'offline.json'
    monkeypatch.setattr('sys.argv', ['graph_answers', '--output', str(output)])
    graph_answers.main()
    result = json.loads(output.read_text(encoding='utf8'))
    assert result['execution'] == 'offline_transport' and result['actual_provider_spend_usd'] == 0
    assert result['graph_build']['status'] == 'complete' and len(result['answers']) == 24
    selected = next(row for row in result['answers'] if row['question_id'] == 'selected namesake' and row['variant'] == 'graph_assisted')
    assert selected['graph_status']['path_complete']
    ambiguous = next(row for row in result['answers'] if row['question_id'] == 'namesake ambiguity' and row['variant'] == 'graph_assisted')
    assert ambiguous['graph_status']['status'] == 'ambiguous'
