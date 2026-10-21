'''Recovery reuses committed windows while accounting only for new provider calls.'''

import json

import pytest

from osintgpt.graph import ExtractionOptions, build_graph, graph_for, saved_graph_status
from osintgpt.graph.checkpoints import Checkpoints
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.llm.usage import UsageRecorder
from test_build_status import EMPTY, Generator, project
from test_extraction_quality import proposal


def long_source(project):
    path = project.paths.root / 'material' / 'a.md'
    path.write_text('Alpha funds Beta. ' + 'text ' * 20000, encoding='utf-8')
    return path


@pytest.mark.parametrize('failure', [RuntimeError('Unavailable'), KeyboardInterrupt()])
def test_resume_reuses_windows_after_restart_and_accounts_only_new_calls(project, failure):
    long_source(project)
    first = Generator(EMPTY, EMPTY, failure, recorder=UsageRecorder())
    if isinstance(failure, Exception):
        assert build_graph(project, first).status == 'failed'
    else:
        with pytest.raises(KeyboardInterrupt):
            build_graph(project, first)
    before = saved_graph_status(project)
    assert before['checkpoints']['windows_saved'] == 2
    assert before['checkpoints']['windows_pending'] > 0
    with graph_for(project) as graph:
        assert graph.state.documents() == {} and graph.entity_count == 0
    second = Generator(recorder=UsageRecorder())
    result = build_graph(project, second)
    assert result.status == 'complete'
    assert result.accounting.windows_reused == 2
    assert second.calls + 2 == result.accounting.windows_total == result.accounting.windows_completed
    assert result.calls == result.usage['calls'] == second.calls
    assert result.usage['total_tokens'] == second.calls * 15
    after = saved_graph_status(project)
    assert after['checkpoints']['documents'] == 0
    assert after['previous_runs'][0]['counts']['requests_attempted'] == 3
    assert after['previous_runs'][0]['usage'] == before['usage']
    assert build_graph(project, Generator()).calls == 0


def test_resume_restores_entity_ids_and_does_not_duplicate_audit(project):
    long_source(project)
    first = Generator(json.dumps(proposal('Invented.')), KeyboardInterrupt())
    options = ExtractionOptions(repair_quotes=False)
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, first, options=options)
    with graph_for(project) as graph:
        checkpoint = Checkpoints(graph).record('material/a.md', 'material/a.md', (), 1)
        saved = checkpoint.windows()[0][1]
        ids = {item['key'] for item in saved['entities']}
        assert ids and len(graph.rejections()) == 1
        assert not graph.entities()

    second = Generator()
    generate = second.generate
    contexts = []
    def capture(system, user):
        contexts.append(json.loads(user)['known_entities'])
        return generate(system, user)
    second.generate = capture
    result = build_graph(project, second, options=options)
    assert result.status == 'complete_with_gaps'
    assert ids <= {item['id'] for item in contexts[0]}
    assert result.accounting.rejected_claims == 0
    with graph_for(project) as graph:
        assert len(graph.rejections()) == 1
        assert ids <= {item.key for item in graph.entities()}


@pytest.mark.parametrize('change', ['source', 'model', 'prompt', 'window', 'recipe_version', 'repair'])
def test_incompatible_saved_work_is_replaced_with_visible_reason(project, monkeypatch, change):
    path = long_source(project)
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, KeyboardInterrupt()))
    generator, options = Generator(), ExtractionOptions()
    if change == 'source':
        path.write_text('Changed. ' + path.read_text())
    elif change == 'model':
        generator.model = 'different-model'
    elif change == 'prompt':
        from osintgpt.graph import recipes
        original = recipes.prompt
        monkeypatch.setattr(recipes, 'prompt', lambda name: original(name) + '\nRevised.')
    elif change == 'window':
        options = ExtractionOptions(context_budget=16000)
    elif change == 'recipe_version':
        from osintgpt.graph import recipes
        monkeypatch.setattr(recipes, 'RECIPE_VERSION', recipes.RECIPE_VERSION + 1)
    else:
        options = ExtractionOptions(repair_quotes=False)
    result = build_graph(project, generator, options=options)
    assert result.status == 'complete'
    assert result.accounting.windows_reused == 0
    assert result.accounting.checkpoints_discarded == 1
    assert 'changed' in result.details[0]['reason'].lower()


def test_retired_record_cap_does_not_restart_or_restrict_compatible_saved_work(project):
    long_source(project)
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, EMPTY, KeyboardInterrupt()))
    with pytest.warns(DeprecationWarning, match='retired'):
        options = ExtractionOptions(max_calls=2)
    result = build_graph(project, Generator(), options=options)
    assert result.status == 'complete' and result.accounting.windows_reused == 2
    assert result.accounting.checkpoints_discarded == 0


def test_rebuild_discards_pending_work_even_when_source_unchanged(project):
    long_source(project)
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, KeyboardInterrupt()))
    result = build_graph(project, Generator(), rebuild=True)
    assert result.status == 'complete' and result.accounting.windows_reused == 0
    assert result.accounting.checkpoints_discarded == 1


def test_completed_structured_records_resume_without_repeating_calls(project):
    path = project.paths.root / 'rows.csv'
    path.write_text('id,body\none,First record\ntwo,Second record\nthree,Third record\n')
    corpus = Corpus.load(project.paths.sources)
    corpus.unregister('material')
    corpus.register(path, FieldMapping(content=('body',), identity='id'))
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, KeyboardInterrupt()))
    result = build_graph(project, Generator())
    assert result.status == 'complete' and result.calls == 2
    assert result.accounting.windows_reused == 1
    assert result.accounting.records_loaded == 3


def test_broken_progress_observer_does_not_lose_saved_work(project):
    def broken(*args):
        raise RuntimeError('UI detached')
    assert build_graph(project, Generator(), on_progress=broken).status == 'complete'
