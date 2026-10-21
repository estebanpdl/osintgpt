'''Failure archives, legacy work recovery, adaptive source coverage and operator inspection.'''

import json
from dataclasses import asdict

import pytest

from osintgpt.graph import build_graph, graph_for, extract_document, ExtractionOptions, saved_graph_status
from osintgpt.graph.accounting import BuildCounts, recorded_usage, usage_messages
from osintgpt.graph.diagnostics import generation_diagnostics
from osintgpt.graph.events import GraphCancelled
from osintgpt.graph.recipes import fingerprint
from osintgpt.graph.run_settings import GraphRunSettings
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder
from osintgpt.llm.generation import GenerationResponse
from test_build_status import project, Generator
from test_continuation_capacity import scripted, search


@pytest.mark.parametrize('kind', ['responses', 'gemini', 'anthropic'])
@pytest.mark.parametrize('bad', ['', '{broken', {'wrong': []}])
def test_structural_correction_keeps_original_and_diagnostics_after_publication(project, kind, bad):
    provider, requests = scripted(kind, [bad, {'entities': [], 'edges': []}])
    try:
        report = build_graph(project, provider)
        assert report.status == 'complete', report.failed
        assert len(requests) == 2 and report.accounting.structured_corrections == 1
        before = (project.paths.root / 'graph.sqlite').read_bytes()
        summary = generation_diagnostics(project)
        assert len(summary['results']) == 2
        assert [r['state'] for r in summary['results']] == ['invalid', 'validated']
        assert all('response' not in row for row in summary['results'])
        raw = generation_diagnostics(project, raw=True)
        assert raw['results'][0]['response']
        assert (project.paths.root / 'graph.sqlite').read_bytes() == before
        assert saved_graph_status(project)['checkpoints']['responses_saved'] == 0
    finally:
        provider.client.close()


def test_legacy_tool_response_is_recovered_without_resubmitting_the_old_request(project):
    provider, requests = scripted('responses', [[search('Example')], {'entities': [], 'edges': []}])
    provider.recorder.cost_ceiling_usd = 0
    try:
        with pytest.raises(CostLimitReached):
            build_graph(project, provider)
        with graph_for(project) as graph:
            row = graph.connection.execute('SELECT * FROM graph_jobs').fetchone()
            recipe = json.loads(row['recipe_data'])
            recipe['version'] = 4
            recipe['options']['context_units'] = 4096
            graph.connection.execute('UPDATE graph_jobs SET recipe=?,recipe_data=?', (fingerprint(recipe), json.dumps(recipe)))
            response = graph.connection.execute("SELECT data FROM graph_responses WHERE purpose='extraction'").fetchone()
            data = json.loads(response[0]); data.pop('_journal')
            graph.connection.execute("UPDATE graph_responses SET request_hash='original-v4-request',data=? WHERE purpose='extraction'", (json.dumps(data),))
            graph.connection.commit()
        provider.recorder = UsageRecorder()
        report = build_graph(project, provider)
        assert report.status == 'complete', report.failed
        assert report.accounting.responses_recovered == 1
        assert report.calls == 1 and len(requests) == 2
        assert report.accounting.checkpoints_discarded == 0
    finally:
        provider.client.close()


def test_continuation_checkpoint_resumes_without_repeating_paid_turn(project):
    provider, requests = scripted('responses', [[search('Example')], {'entities': [], 'edges': []}], continuation_size=44000)
    cancelled = False
    def cancel(event):
        nonlocal cancelled
        if event.kind == 'progress' and event.data['counts'].get('entity_lookups'):
            cancelled = True
    try:
        with pytest.raises(GraphCancelled):
            build_graph(project, provider, on_event=cancel, should_cancel=lambda: cancelled)
        report = build_graph(project, provider)
        assert report.status == 'complete', report.failed
        assert report.accounting.responses_reused == 1
        assert len(requests) == 2
    finally:
        provider.client.close()


def test_actual_provider_capacity_rejection_replans_source_and_keeps_complete_coverage():
    class ContextError(ValueError):
        code = 'context_length_exceeded'
    class Model:
        def __init__(self): self.accepted = []
        def generate(self, system, user):
            source = json.loads(user)['source']
            if len(source['text']) > 100:
                raise ContextError('Provider context cannot fit this source.')
            self.accepted.append(source)
            return '{"entities":[],"edges":[]}'
    model = Model()
    text = 'Texto sin modificaciones. ' * 40
    result = extract_document(model, 'source.md', text)
    assert result.ok, result.problem
    covered = 0
    for source in model.accepted:
        assert source['start'] <= covered
        assert source['text'] == text[source['start']:source['end']]
        covered = source['end']
    assert covered == len(text)
    assert result.accounting.windows_completed == len(model.accepted)


def test_saved_retired_settings_are_ignored_and_removed_on_save(project):
    path = project.paths.root / 'graph-build.toml'
    path.write_text('context_budget=24000\ncontext_units=0\nmax_calls=1\noutput_tokens=4096\nrpm=12\n')
    settings = GraphRunSettings.load(project)
    assert settings.context_budget == 24000 and settings.rpm == 12
    assert not {'context_units', 'max_calls', 'output_tokens'} & asdict(settings).keys()
    settings.save(project)
    assert 'context_units' not in path.read_text() and 'max_calls' not in path.read_text()


def test_usage_messages_distinguish_pending_missing_and_unpriced_calls():
    class Model: recorder = UsageRecorder()
    counts = BuildCounts(requests_attempted=1)
    pending = recorded_usage(Model(), counts)
    assert 'Awaiting usage' in usage_messages(pending)[0]
    counts.requests_failed = 1
    failed = recorded_usage(Model(), counts)
    assert 'Usage unavailable' in usage_messages(failed)[0]
    counts.requests_failed, counts.requests_succeeded = 0, 1
    Model.recorder.record(Usage('openai', 'not-in-local-catalogue', 20, 10))
    priced = recorded_usage(Model(), counts)
    assert priced['tokens_complete'] and not priced['cost_complete']
    assert usage_messages(priced) == ['Model pricing unavailable for 1 calls; recorded token usage is shown separately.']


def test_more_than_twelve_eligible_corrections_are_attempted():
    from test_extraction_quality import Script, proposal
    bad = proposal('Invented.')
    bad['edges'] *= 25
    model = Script(bad, {'corrections': []})
    result = extract_document(model, 'source.md', 'Alpha funds Beta.')
    assert result.ok
    correction = model.calls[1][1]
    assert len(correction['candidates']) == 25


def test_adapted_plan_resumes_and_is_removed_when_source_changes(project):
    from osintgpt.graph.checkpoints import Checkpoints
    from osintgpt.graph.windowing import Window
    from osintgpt.graph.recipes import extraction_recipe
    model = Generator()
    with graph_for(project) as graph:
        journal = Checkpoints(graph)
        recipe = extraction_recipe(model, ExtractionOptions())
        journal.prepare('source.md', 'v1', fingerprint(recipe), recipe)
        record = journal.record('source.md', 'source.md', [], 1)
        record.replan([Window(0, 80), Window(60, 140)], 1)
        assert record.plan([Window(0, 140)]) == [Window(0, 80), Window(60, 140)]
        journal.prepare('source.md', 'v2', fingerprint(recipe), recipe)
        fresh = journal.record('source.md', 'source.md', [], 1)
        assert fresh.plan([Window(0, 200)]) == [Window(0, 200)]


@pytest.mark.parametrize('response', [
    GenerationResponse('{"entities":[],"edges":[]}', status='completed', refused=True),
    GenerationResponse('{"entities":[],"edges":[]}', status='incomplete', stop_reason='max_output_tokens'),
])
def test_refusal_and_truncation_are_archived_and_not_reissued_on_resume(project, response):
    class Model:
        calls = 0
        def generate_response(self, system, user, **kwargs):
            self.calls += 1
            return response
    model = Model()
    first = build_graph(project, model)
    second = build_graph(project, model)
    assert first.status == second.status == 'failed'
    assert model.calls == 1
    audit = generation_diagnostics(project, raw=True)['results']
    assert len(audit) == 1 and audit[0]['state'] == 'invalid'
    assert audit[0]['response']['text'] == response.text
    assert not audit[0]['reusable']
