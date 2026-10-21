'''Cached source mappings retain precedence and invalidate on relevant filesystem changes.'''

import pytest

from osintgpt.graph import graph_for
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.verification_cache import EvidenceCache, VerificationChanged
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.documents import FieldMapping
from test_progressive_exploration import finish


def test_mapping_cache_preserves_direct_file_and_first_folder_precedence(graph_case):
    root = graph_case.project.paths.root
    folder = root / 'records'
    (folder / 'nested').mkdir(parents=True)
    direct, nested = folder / 'direct.csv', folder / 'nested' / 'nested.csv'
    for path in (direct, nested):
        path.write_text('first,second\na,b\n', encoding='utf8')
    corpus = Corpus.load(graph_case.project.paths.sources)
    first, second = FieldMapping(content=('first',)), FieldMapping(content=('second',))
    corpus.register('records', first)
    corpus.register('records/nested', second)
    corpus.register('records/direct.csv', second)
    cache = EvidenceCache(graph_case.project)
    assert cache.mapping_for(direct) == second
    assert cache.mapping_for(nested) == first
    assert cache.mapping_for(nested) == corpus.mapping_for(nested, root)
    assert not cache.problem()
    corpus.register('records/direct.csv', first)
    with pytest.raises(VerificationChanged, match='mappings changed'):
        cache.check()
    fresh = EvidenceCache(graph_case.project)
    assert fresh.mapping_for(direct) == first


@pytest.mark.parametrize('change', ['nested_add', 'nested_remove', 'restore_missing', 'canon_appears'])
def test_folder_membership_invalidates_cached_mappings(graph_case, change):
    root = graph_case.project.paths.root
    graph_case.publish('records/nested/one.md', 'Alpha links Beta.')
    corpus = Corpus.load(graph_case.project.paths.sources)
    corpus.unregister('records/nested/one.md')
    corpus.register('records')
    corpus.register('absent')
    from osintgpt.projects.paths import CANON_DIR
    canon = root / CANON_DIR
    # Project.create has an empty canon; remove it only inside this disposable fixture.
    if change == 'canon_appears' and canon.exists():
        canon.rmdir()
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    prepared = query.prepare_export()
    if change == 'nested_add':
        (root / 'records/nested/two.md').write_text('Additional synthetic source.', encoding='utf8')
    elif change == 'nested_remove':
        (root / 'records/nested/one.md').unlink()
    elif change == 'restore_missing':
        (root / 'absent').mkdir()
    else:
        canon.mkdir()
    assert not query.check_current() and query.state == 'invalidated'
    assert prepared['edges'][0]['verification']['current'] == 'found'
    with pytest.raises(VerificationChanged):
        query.prepare_export()


def test_directory_change_during_mapping_walk_is_rejected(graph_case, monkeypatch):
    graph_case.publish('records/one.md', 'Alpha links Beta.')
    corpus = Corpus.load(graph_case.project.paths.sources)
    corpus.unregister('records/one.md')
    corpus.register('records')
    original = Corpus.mappings
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (graph_case.project.paths.root / 'records/added.md').write_text('New membership.', encoding='utf8')
        return result
    monkeypatch.setattr(Corpus, 'mappings', changed)
    query = OperatorExploration(graph_case.project, 'alpha')
    query.advance()
    assert query.state == 'invalidated' and not query.rows


def test_mapping_change_during_export_rejects_preparation(graph_case, monkeypatch):
    graph_case.publish('notes.md', 'Alpha links Beta.')
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    from osintgpt.graph import operator_exports
    original = operator_exports.query_export_context
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        Corpus.load(graph_case.project.paths.sources).register('notes.md', FieldMapping(content=('changed',)))
        return result
    monkeypatch.setattr(operator_exports, 'query_export_context', changed)
    with pytest.raises(VerificationChanged):
        query.prepare_export()
    assert query.state == 'invalidated'


def test_cached_mapping_is_not_shared_across_projects(graph_case, tmp_path):
    from osintgpt import Project
    first = graph_case.project
    second = Project.create('Other', home=tmp_path / 'other')
    for project, column in ((first, 'first'), (second, 'second')):
        path = project.paths.root / 'record.csv'
        path.write_text('first,second\na,b\n', encoding='utf8')
        Corpus.load(project.paths.sources).register('record.csv', FieldMapping(content=(column,)))
    a, b = EvidenceCache(first), EvidenceCache(second)
    assert a.mapping_for(first.paths.root / 'record.csv').content == ('first',)
    assert b.mapping_for(second.paths.root / 'record.csv').content == ('second',)
