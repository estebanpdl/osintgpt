'''Deterministic chain semantics, configuration and run-local accounting.'''

import importlib
import math

import pytest

from osintgpt import Project, ProjectSettings, snowball
from osintgpt.agentic import ToolContext, snowball_search
from osintgpt.agentic.snowball_output import snowball_data, snowball_rows, grouped_rows
from osintgpt.config import Settings
from osintgpt.index_events import observe_index, emit_index
from osintgpt.llm.base import EmbeddingPurpose
from osintgpt.llm.usage import Usage, UsageRecorder
from osintgpt.projects import save_user_defaults
from tests.snowball_fakes import GraphEmbedder, GraphStore

operation = importlib.import_module('osintgpt.agentic.snowball')


def walk(**options):
    return snowball(None, 'start', GraphEmbedder(), store=GraphStore(), **options)


def test_chain_order_scores_and_drift_use_the_initial_query():
    embedder = GraphEmbedder()
    result = snowball(None, 'start', embedder, store=GraphStore(), depth=4)
    assert [hop.text for hop in result.hops] == [f'passage {i}' for i in range(4)]
    assert [hop.query for hop in result.hops] == ['start', 'passage 0', 'passage 1', 'passage 2']
    assert [hop.drift for hop in result.hops] == pytest.approx([math.cos(.1 + i * .03) for i in range(4)])
    assert [hop.result.score for hop in result.hops] == pytest.approx([math.cos(.1), *[math.cos(.03)] * 3])
    assert [p for _, p in embedder.calls] == [EmbeddingPurpose.QUERY, EmbeddingPurpose.DOCUMENT] * 4
    assert result.stopped == 'reached depth 4'


def test_default_text_dedupe_grows_past_all_copies_and_retains_sources():
    store = GraphStore(count=8, copies=18)
    result = snowball(None, 'start', GraphEmbedder(), store=store, depth=8)
    assert [hop.text for hop in result.hops] == [f'passage {i}' for i in range(8)]
    assert len(result.hops[0].identical_text_refs) == 19
    assert 'copies.jsonl#17' in result.hops[0].identical_text_refs
    assert any(request[2] > 18 for request in store.requests)
    assert not store.closed


def test_chunk_dedupe_retains_copied_records_without_revisiting_chunks():
    result = snowball(None, 'start', GraphEmbedder(), store=GraphStore(2, 6), depth=20, dedupe='chunk')
    assert len(result) == 8
    assert len({(h.ref, h.result.chunk.sequence) for h in result.hops}) == 8
    assert result.stopped == 'nothing new to walk to'


def test_default_depth_is_25_but_agentic_default_remains_five_and_has_no_clamp():
    assert len(snowball(None, 'start', GraphEmbedder(), store=GraphStore(30))) == 25
    context = ToolContext(None, GraphEmbedder(), store=GraphStore(30))
    assert snowball_search(context, 'start').count == 5
    assert snowball_search(context, 'start', depth=12).count == 12


def test_threshold_is_inclusive_and_stops_before_accepting_a_low_score():
    score = math.cos(.1)
    result = walk(threshold=score, depth=2)
    assert len(result) == 2
    result = walk(threshold=score + .0001)
    assert len(result) == 0 and 'fell below' in result.stopped
    assert result.embedding_calls == 1


def test_empty_store_and_ref_scope_stop_without_fabricating_hops():
    assert not walk(refs=['absent'])
    assert not snowball(None, 'start', GraphEmbedder(), store=GraphStore(0))
    assert len(walk(refs=['posts.jsonl'], depth=3)) == 3


def test_no_drift_avoids_document_embeddings():
    result = walk(depth=3, score_against_initial=False)
    assert all(hop.drift is None for hop in result.hops)
    assert result.embedding_calls == 3


@pytest.mark.parametrize('options', [
    {'depth': 0}, {'depth': -1}, {'depth': 1.5}, {'depth': True},
    {'threshold': float('nan')}, {'threshold': float('inf')}, {'threshold': -2},
    {'dedupe': 'unknown'},
])
def test_invalid_parameters_fail_before_embedding(options):
    embedder = GraphEmbedder()
    with pytest.raises(ValueError):
        snowball(None, 'start', embedder, store=GraphStore(), **options)
    assert not embedder.calls


def test_empty_query_fails_before_configuration(monkeypatch):
    monkeypatch.setattr(operation, '_configuration', lambda *a: pytest.fail('configuration read'))
    with pytest.raises(ValueError, match='nonempty'):
        snowball(None, '  ')


def test_dates_default_to_timestamp_and_grouping_preserves_hop_numbers():
    result = walk(depth=4)
    rows = snowball_rows(result, 'month')
    assert [row['date_status'] for row in rows] == ['parsed', 'parsed', 'invalid', 'missing']
    assert [row['hop'] for row in rows] == [1, 2, 3, 4]
    groups = grouped_rows(rows, 'month')
    assert [name for name, _ in groups] == ['2026-01', '2026-02', 'Unknown date']
    assert [row['hop'] for row in groups[-1][1]] == [3, 4]
    for grouping, expected in [('day', '2026-02-03'), ('week', '2026-W06'), ('year', '2026')]:
        assert snowball_rows(result, grouping)[0]['period'] == expected


def test_explicit_metadata_date_format_and_ambiguous_dates():
    result = walk(depth=1, date_field='published', date_format='%d/%m/%Y')
    assert snowball_rows(result, 'month')[0]['period'] == '2026-10'
    result = walk(depth=1, date_field='published')
    assert snowball_rows(result, 'month')[0]['period'] == 'Unknown date'
    result = walk(depth=1, date_field='missing')
    assert snowball_rows(result)[0]['date_status'] == 'missing'


def test_usage_is_per_run_and_does_not_duplicate_or_replace_parent_recorder():
    parent = UsageRecorder()
    parent.record(Usage('fake', GraphEmbedder.model, input_tokens=100))
    embedder = GraphEmbedder(recorder=parent)
    result = snowball(None, 'start', embedder, store=GraphStore(), depth=3)
    assert result.usage.calls == 6 and result.usage.total_tokens == 42
    assert parent.calls == 7 and parent.total_tokens == 142
    assert embedder.recorder is parent
    usage = snowball_data(result)['usage']
    assert usage['complete'] and usage['estimated_cost_usd'] > 0
    assert result.usage.cost_ceiling_usd is None


def test_missing_usage_and_unknown_pricing_never_look_free():
    result = snowball(None, 'start', GraphEmbedder(report=False), store=GraphStore(), depth=1)
    assert snowball_data(result)['usage']['estimated_cost_usd'] is None
    assert result.unrecorded_calls == 2
    result.usage.record(Usage('fake', 'unknown-model', input_tokens=4))
    assert not snowball_data(result)['usage']['complete']


def test_observer_is_restored_after_failure_and_injected_store_stays_open():
    events = []
    embedder = GraphEmbedder()
    embedder.vectors['start'] = [float('nan')]
    store = GraphStore()
    with observe_index(events.append):
        with pytest.raises(ValueError, match='nonfinite'):
            snowball(None, 'start', embedder, store=store)
        emit_index('restored')
    assert [event.kind for event in events] == ['usage', 'restored']
    assert not store.closed


def test_each_usage_event_reaches_an_enclosing_observer_once():
    events = []
    with observe_index(events.append):
        result = walk(depth=3)
    assert len(events) == result.usage.calls == 6
    assert all(event.kind == 'usage' for event in events)


def test_dimension_changes_are_rejected_even_without_drift():
    embedder = GraphEmbedder()
    embedder.vectors['passage 0'] = [1, 0, 0]
    with pytest.raises(ValueError, match='dimensions changed'):
        snowball(None, 'start', embedder, store=GraphStore(), score_against_initial=False)


def test_owned_runtime_uses_project_model_defaults_and_remote_configuration(tmp_path, monkeypatch):
    project = Project.create('Configured', home=tmp_path).with_settings(
        embedding_provider='openai', embedding_model=GraphEmbedder.model,
        storage_backend='postgres', cost_ceiling_usd=0,
    )
    save_user_defaults(tmp_path, ProjectSettings(embedding_model='other-model'))
    config = Settings(postgres_dsn='postgresql://synthetic-only')
    seen = {}
    store = GraphStore()

    def factory(provider, effective_config, model=None):
        seen.update(provider=provider, config=effective_config, model=model)
        return GraphEmbedder()

    def open_store(selected, effective_config):
        assert selected.settings.storage_backend == 'postgres'
        assert effective_config.postgres_dsn == config.postgres_dsn
        assert effective_config is seen['config']
        return store

    monkeypatch.setattr(operation, 'build_embedding_provider', factory)
    monkeypatch.setattr(operation, 'store_for', open_store)
    result = snowball(project, 'start', home=tmp_path, config=config, depth=2)
    assert seen['model'] == GraphEmbedder.model and seen['provider'] == 'openai'
    assert len(result) == 2 and store.closed
    assert result.usage.cost_ceiling_usd is None


def test_owned_store_is_closed_when_embedding_fails(tmp_path, monkeypatch):
    project = Project.create('Failure', home=tmp_path)
    store = GraphStore()
    monkeypatch.setattr(operation, 'store_for', lambda *a: store)
    with pytest.raises(KeyError):
        snowball(project, 'unknown', GraphEmbedder(), home=tmp_path, config=Settings())
    assert store.closed
