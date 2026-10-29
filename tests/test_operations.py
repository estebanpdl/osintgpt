'''Behavioral contracts for one invocation's policy and owned dependencies.'''

from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from threading import Barrier

import pytest

from osintgpt import Project, ProjectSettings, snowball
from osintgpt.config import Settings
from osintgpt.index_events import emit_index, observe_index
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder
from osintgpt.operations import OperationContext, lower_ceiling
from osintgpt.projects import save_user_defaults
from tests.snowball_fakes import GraphEmbedder, GraphStore


def test_settings_are_resolved_once_with_inheritance_and_invocation_choices(tmp_path, monkeypatch):
    project = Project.create('Policy', home=tmp_path).with_settings(
        embedding_provider='', embedding_model='', storage_backend='',
        ingestion_model='', cost_ceiling_usd=None,
    )
    save_user_defaults(tmp_path, ProjectSettings(
        embedding_provider='voyage', embedding_model='inherited',
        ingestion_model='vision', storage_backend='postgres', cost_ceiling_usd=3,
    ))
    calls = []
    original = Project.effective_settings

    def effective(selected, defaults):
        calls.append(selected.id)
        return original(selected, defaults)

    monkeypatch.setattr(Project, 'effective_settings', effective)
    context = OperationContext(project, home=tmp_path, config=Settings(), overrides={
        'embedding_model': 'invocation', 'suggest_followups': False,
        'conversation_window': 0, 'cost_ceiling_usd': 9,
    })
    save_user_defaults(tmp_path, ProjectSettings(embedding_model='later', cost_ceiling_usd=0))
    assert calls == [project.id]
    assert context.settings.embedding_provider == 'voyage'
    assert context.settings.embedding_model == context.config.openai_embedding_model == 'invocation'
    assert context.settings.ingestion_model == 'vision'
    assert context.settings.conversation_window == 0 and not context.settings.suggest_followups
    assert context.settings.cost_ceiling_usd == 3
    assert context.project.id == project.id and context.project.settings.storage_backend == 'postgres'
    assert project.settings.storage_backend == '' and project.settings.embedding_model == ''


def test_explicit_connection_model_precedence_can_be_overridden(tmp_path):
    project = Project.create('Model', home=tmp_path).with_settings(embedding_model='project')
    config = Settings(openai_embedding_model='explicit')
    first = OperationContext(project, home=tmp_path, config=config)
    second = OperationContext(project, home=tmp_path, config=config, overrides={'embedding_model': 'invocation'})
    assert first.settings.embedding_model == first.config.openai_embedding_model == 'explicit'
    assert second.settings.embedding_model == second.config.openai_embedding_model == 'invocation'


@pytest.mark.parametrize('overrides,expected', [
    ({'graph_model_mode': 'inherit'}, ('inherit', '', '')),
    ({'graph_generation_provider': 'anthropic'}, ('independent', 'anthropic', '')),
])
def test_graph_overrides_do_not_retain_a_model_from_another_selection(overrides, expected):
    defaults = ProjectSettings(generation_model='chat', ingestion_model='vision',
                               graph_model_mode='independent', graph_generation_provider='openai',
                               graph_generation_model='graph')
    context = OperationContext(None, defaults=defaults, overrides=overrides)
    settings = context.settings
    assert (settings.graph_model_mode, settings.graph_generation_provider,
            settings.graph_generation_model) == expected
    assert settings.generation_model == 'chat' and settings.ingestion_model == 'vision'


@pytest.mark.parametrize('project,run,expected', [
    (None, None, None), (None, 2, 2), (3, None, 3), (3, 2, 2), (2, 3, 2), (0, 3, 0),
])
def test_only_nonempty_ceilings_constrain_calls(project, run, expected):
    assert lower_ceiling(project, run) == expected


@pytest.mark.parametrize('ceiling', [-1, float('nan'), float('inf'), True])
def test_invalid_ceiling_fails_before_dependencies(ceiling):
    with pytest.raises(ValueError, match='finite'):
        OperationContext(None, cost_ceiling_usd=ceiling)


def test_no_work_is_keyless_and_builds_nothing(tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail('dependency built without work')

    project = Project.create('Lazy', home=tmp_path)
    with OperationContext(project, home=tmp_path, config=Settings(),
                          embedding_factory=unexpected, store_factory=unexpected) as context:
        assert context.settings.embedding_provider == 'openai'
    assert context.outcome.status == 'completed'
    assert context.outcome.usage['calls'] == 0
    assert context.usage.cost_ceiling_usd is None


def test_empty_embedding_input_does_not_build_or_account_for_a_provider():
    def unexpected(*args, **kwargs):
        pytest.fail('provider built for empty input')

    with OperationContext(None, embedding_factory=unexpected, cost_ceiling_usd=0) as context:
        assert context.embed([]) == []
    assert context.outcome.usage['calls'] == 0 and context.outcome.status == 'completed'


def test_inherited_remote_store_gets_configuration_and_closes_once(tmp_path):
    project = Project.create('Remote', home=tmp_path).with_settings(storage_backend='')
    config = Settings(postgres_dsn='postgresql://synthetic-only', qdrant_api_key='synthetic-secret')
    calls = []
    store = GraphStore()
    store.close = lambda: calls.append('closed')

    def open_store(selected, resolved):
        assert selected.settings.storage_backend == 'postgres'
        assert resolved.postgres_dsn == config.postgres_dsn
        assert selected.id == project.id
        calls.append('opened')
        return store

    with OperationContext(project, config=config, defaults=ProjectSettings(storage_backend='postgres'),
                          store_factory=open_store) as context:
        assert context.store is context.store is store
    assert calls == ['opened', 'closed']
    assert 'synthetic' not in json.dumps(asdict(context.outcome))


def test_injected_resources_and_parent_recorder_are_preserved():
    parent = UsageRecorder()
    embedder, store = GraphEmbedder(recorder=parent), GraphStore()
    events = []
    with observe_index(events.append):
        with OperationContext(None, embedder=embedder, store=store) as context:
            assert context.store is store
            context.embed(['start'])
        emit_index('restored')
    assert context.usage.calls == parent.calls == 1
    assert [event.kind for event in events] == ['usage', 'restored']
    assert embedder.recorder is parent and not store.closed


def test_owned_factory_does_not_receive_recorder_and_only_builds_once():
    providers = []

    def build(provider, config, model=None):
        embedder = GraphEmbedder()
        embedder.closed = 0
        embedder.close = lambda: setattr(embedder, 'closed', embedder.closed + 1)
        providers.append(embedder)
        return embedder

    context = OperationContext(None, embedding_factory=build)
    with context:
        context.embed(['start'])
        context.embed(['start'])
    assert len(providers) == 1 and providers[0].closed == 1
    assert context.usage.calls == 2
    with pytest.raises(RuntimeError, match='fresh'):
        with context:
            pass
    with pytest.raises(RuntimeError, match='active'):
        context.embed(['start'])


def test_unknown_usage_stays_unknown_without_a_default_stop():
    with OperationContext(None, embedder=GraphEmbedder(report=False)) as context:
        context.embed(['start'])
        context.embed(['start'])
    assert context.unrecorded_calls == context.usage.calls == 2
    assert context.outcome.usage['estimated_cost_usd'] is None
    assert context.outcome.usage['uncounted_calls'] == 2
    assert not context.outcome.usage['complete']


def test_unpriced_usage_stays_unknown_without_a_default_stop():
    embedder = GraphEmbedder()
    embedder.model = 'unpriced-test-model'
    with OperationContext(None, embedder=embedder) as context:
        context.embed(['start'])
    assert context.usage.calls == 1 and not context.outcome.usage['complete']
    assert context.outcome.usage['estimated_cost_usd'] is None


@pytest.mark.parametrize('reported', [True, False])
def test_explicit_ceiling_records_overshoot_or_unknown_before_stopping(reported):
    embedder = GraphEmbedder(report=reported)
    context = OperationContext(None, embedder=embedder, cost_ceiling_usd=0)
    events = []
    with observe_index(events.append), pytest.raises(CostLimitReached) as raised:
        with context:
            context.embed(['start'])
            pytest.fail('continued after hard stop')
    assert len(embedder.calls) == context.usage.calls == len(events) == 1
    assert context.outcome.status == 'cost_limit'
    assert raised.value.operation_outcome is context.outcome
    assert context.outcome.usage['estimated_cost_usd'] == (pytest.approx(.00000014) if reported else None)


@pytest.mark.parametrize('failure', [KeyboardInterrupt(), CancelledError(), RuntimeError('secret DSN')],
                         ids=['interrupt', 'cancel', 'failure'])
def test_cleanup_and_safe_outcomes_preserve_original_failures(tmp_path, failure):
    project = Project.create('Failure', home=tmp_path)
    store = GraphStore()
    events = []
    context = OperationContext(project, home=tmp_path, config=Settings(),
                               store_factory=lambda *a: store, on_event=events.append)
    with pytest.raises(type(failure)) as raised:
        with context:
            assert context.store is store
            raise failure
    assert raised.value is failure and store.closed
    assert 'secret' not in json.dumps([event.data for event in events])
    assert context.outcome.status == ('failed' if isinstance(failure, RuntimeError) else 'cancelled')


def test_cleanup_failure_does_not_mask_a_hard_stop(tmp_path):
    project = Project.create('Close', home=tmp_path)
    store = GraphStore()

    def broken_close():
        raise RuntimeError('secret connection failure')

    store.close = broken_close
    context = OperationContext(project, home=tmp_path, config=Settings(),
                               embedder=GraphEmbedder(), store_factory=lambda *a: store,
                               cost_ceiling_usd=0)
    with pytest.raises(CostLimitReached):
        with context:
            assert context.store is store
            context.embed(['start'])
    assert context.outcome.status == 'cost_limit'


def test_display_exception_cannot_break_accounting():
    def display(event):
        raise RuntimeError('secret display error')

    with OperationContext(None, embedder=GraphEmbedder(), on_event=display) as context:
        context.embed(['start'])
    assert context.usage.calls == 1 and context.outcome.status == 'completed'


def test_display_cancellation_propagates_but_does_not_mask_an_accounting_stop():
    def cancel(event):
        raise CancelledError()

    context = OperationContext(None, embedder=GraphEmbedder(), on_event=cancel)
    with pytest.raises(CancelledError), context:
        context.embed(['start'])
    assert context.outcome.status == 'cancelled' and context.usage.calls == 1
    context = OperationContext(None, embedder=GraphEmbedder(), on_event=cancel, cost_ceiling_usd=0)
    with pytest.raises(CostLimitReached), context:
        context.embed(['start'])
    assert context.outcome.status == 'cost_limit' and context.usage.calls == 1


def test_store_open_failure_closes_an_already_created_provider(tmp_path):
    project = Project.create('Open failure', home=tmp_path)
    provider = GraphEmbedder()
    closed = []
    provider.close = lambda: closed.append(True)

    def fail_store(*args):
        raise RuntimeError('secret backend response')

    context = OperationContext(project, config=Settings(), defaults=ProjectSettings(),
                               embedding_factory=lambda *a, **k: provider, store_factory=fail_store)
    with pytest.raises(RuntimeError), context:
        assert context.embedder is provider
        context.store
    assert closed == [True] and context.outcome.status == 'failed'


def test_nested_contexts_forward_once_and_keep_independent_totals():
    with OperationContext(None, embedder=GraphEmbedder()) as parent:
        parent.embed(['start'])
        with OperationContext(None, embedder=GraphEmbedder()) as child:
            child.embed(['start'])
        parent.embed(['start'])
    assert child.usage.calls == 1 and parent.usage.calls == 3


def test_two_concurrent_operations_do_not_share_clients_recorders_or_events(tmp_path):
    barrier = Barrier(2)
    projects = [Project.create(name, home=tmp_path) for name in ('One', 'Two')]

    def run(project, count):
        provider = GraphEmbedder()
        store = GraphStore()
        context = OperationContext(project, home=tmp_path, config=Settings(),
                                   embedding_factory=lambda *a, **k: provider,
                                   store_factory=lambda *a: store)
        with context:
            assert context.store is store
            barrier.wait(timeout=5)
            for _ in range(count):
                context.embed(['start'])
            barrier.wait(timeout=5)
        return context, provider, store

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, project, count) for project, count in zip(projects, (1, 3))]
        results = [future.result(timeout=10) for future in futures]
    assert [context.usage.calls for context, _, _ in results] == [1, 3]
    assert results[0][0].usage is not results[1][0].usage
    assert results[0][1] is not results[1][1]
    assert all(store.closed for _, _, store in results)


def test_snowball_rejects_configured_model_mismatch_before_work(tmp_path):
    project = Project.create('Mismatch', home=tmp_path).with_settings(embedding_model='other')
    embedder, store = GraphEmbedder(), GraphStore()
    with pytest.raises(ValueError, match='does not match'):
        snowball(project, 'start', embedder, store=store, config=Settings(), home=tmp_path)
    assert not embedder.calls and not store.requests and not store.closed


def test_snowball_returns_safe_shared_outcome_and_fresh_usage_each_time():
    first = snowball(None, 'start', GraphEmbedder(), store=GraphStore(), depth=1)
    second = snowball(None, 'start', GraphEmbedder(), store=GraphStore(), depth=1)
    assert first.outcome.status == second.outcome.status == 'completed'
    assert first.usage.calls == second.usage.calls == 2
    assert first.usage is not second.usage
