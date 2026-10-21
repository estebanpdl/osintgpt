'''Verify CLI model choices through native SDK fixtures, including opt-in and invalid pairs.'''

import json

import pytest

import graph_answers
from graph_offline import RecordedFixtureEmbedder
from test_graph_model_benchmarks import FIXTURE
from tests.llm.test_generation_contract import provider_fixture


@pytest.mark.parametrize('override,expected', [
    ([], [('openai', 'answer-A')]),
    (['--extraction-model', 'extract-B'], [('openai', 'answer-A'), ('openai', 'extract-B')]),
    (['--extraction-provider', 'anthropic', '--extraction-model', 'extract-B'],
     [('openai', 'answer-A'), ('anthropic', 'extract-B')]),
])
def test_cli_routes_independent_roles_and_inheritance_through_native_transports(tmp_path, monkeypatch, override, expected):
    created, requests, recorders = [], {}, []

    def generation(provider, settings, *, model, recorder):
        instance, calls = provider_fixture('responses' if provider == 'openai' else provider)
        instance.model, instance.recorder = model, recorder
        created.append((provider, model))
        requests[model] = calls
        recorders.append(recorder)
        return instance

    def embedding(provider, settings, *, model, recorder):
        recorders.append(recorder)
        return RecordedFixtureEmbedder(recorder)

    monkeypatch.setattr(graph_answers, 'build_generation_provider', generation)
    monkeypatch.setattr(graph_answers, 'build_embedding_provider', embedding)
    monkeypatch.setattr('osintgpt.llm.usage.estimate_cost', lambda *a, **k: 0.00001)
    fixture, output = tmp_path / 'fixture.json', tmp_path / 'report.json'
    fixture.write_text(json.dumps(FIXTURE), encoding='utf8')
    monkeypatch.setattr('sys.argv', ['graph_answers', '--live', '--answer-provider', 'openai', '--answer-model', 'answer-A',
        '--embedding-provider', 'openai', '--embedding-model', 'embed-C', '--max-cost-usd', '1',
        '--fixture', str(fixture), '--output', str(output), *override])
    graph_answers.main()
    report = json.loads(output.read_text(encoding='utf8'))
    assert report['status'] == 'complete_pending_review'
    assert created == expected and all(recorder is recorders[0] for recorder in recorders)
    for model, calls in requests.items():
        assert calls and all(call['model'] == model for call in calls)
    extraction = expected[-1]
    assert report['models']['extraction']['provider'] == extraction[0]
    assert report['models']['extraction']['model'] == extraction[1]
    assert report['models']['extraction_selection'] == ('inherit_answering' if not override else 'independent')
    assert {row['model'] for row in report['usage_by_role']['extraction']['records']} == {extraction[1]}
    assert {row['model'] for row in report['usage_by_role']['answering']['records']} == {'answer-A'}
    assert report['usage_by_role']['extraction']['calls'] == 1
    assert report['usage_by_role']['answering']['calls'] == 2


@pytest.mark.parametrize('args', [
    ['--answer-provider', 'openai', '--answer-model', 'unused'],
    ['--live', '--answer-provider', 'openai', '--answer-model', 'unused', '--max-cost-usd', '1',
     '--embedding-provider', 'openai', '--embedding-model', 'unused', '--extraction-provider', 'anthropic'],
])
def test_live_opt_in_and_complete_extraction_pair_required_before_clients(tmp_path, monkeypatch, args):
    monkeypatch.setattr(graph_answers, 'build_generation_provider', lambda *a, **k: pytest.fail('No provider expected'))
    monkeypatch.setattr('sys.argv', ['graph_answers', '--output', str(tmp_path / 'report.json'), *args])
    with pytest.raises(SystemExit) as error:
        graph_answers.main()
    assert error.value.code == 2
