'''Refresh and removal preserve published claims until safe replacement.'''

import json
import sqlite3

import pytest

from osintgpt.graph import build_graph, graph_for, saved_graph_status
from osintgpt.graph.checkpoints import Checkpoints
from osintgpt.ingestion import Corpus, FieldMapping
from test_build_status import EMPTY, Generator, project
from test_extraction_quality import proposal

CLAIM = json.dumps(proposal('Alpha funds Beta.'))


def publish(project):
    path = project.paths.root / 'material' / 'a.md'
    path.write_text('Alpha funds Beta.')
    assert build_graph(project, Generator(CLAIM)).status == 'complete'
    return path


def stored(project):
    with graph_for(project) as graph:
        return graph.edges(), graph.entities(), graph.state.entries()


def test_default_build_refreshes_changed_sources_and_skips_unchanged(project):
    path = publish(project)
    new = path.with_name('b.md')
    new.write_text('Another source')
    first = build_graph(project, Generator())
    assert first.calls == 1 and first.skipped == 1
    path.write_text('Alpha no longer funds Beta.')
    result = build_graph(project, Generator())
    assert result.calls == 1 and result.skipped == 1
    assert result.accounting.documents_changed == 1
    assert result.removed == 1 and stored(project)[0] == []
    assert stored(project)[1] == []


def test_completed_sources_refresh_when_recipe_changes(project):
    publish(project)
    generator = Generator()
    generator.model = 'new-model'
    report = build_graph(project, generator, incremental=True)
    assert report.calls == 1 and report.accounting.documents_changed == 1
    assert 'settings changed' in report.details[0]['reason']


def test_failed_replacement_keeps_old_graph_and_saved_windows(project):
    path = publish(project)
    previous = stored(project)
    path.write_text('New text ' * 10000)
    result = build_graph(project, Generator(EMPTY, RuntimeError('Unavailable')))
    assert result.status == 'complete_with_gaps'
    assert stored(project) == previous
    assert saved_graph_status(project)['checkpoints']['windows_saved'] == 1
    assert saved_graph_status(project)['stale']
    result = build_graph(project, Generator())
    assert result.status == 'complete' and result.accounting.windows_reused == 1
    assert stored(project)[0] == []


def test_publication_failure_rolls_back_claims_entities_and_completion(project, monkeypatch):
    path = publish(project)
    previous = stored(project)
    path.write_text('Updated source')
    from osintgpt.graph.state import GraphState
    complete = GraphState.completed
    def fail(self, *args, **kwargs):
        complete(self, *args, **kwargs)
        raise sqlite3.OperationalError('publication failure')
    monkeypatch.setattr(GraphState, 'completed', fail)
    result = build_graph(project, Generator())
    assert result.failed and stored(project) == previous
    assert saved_graph_status(project)['checkpoints']['windows_saved'] == 1
    monkeypatch.setattr(GraphState, 'completed', complete)
    result = build_graph(project, Generator())
    assert result.calls == 0 and result.accounting.windows_reused == 1
    assert result.status == 'complete'


def test_readers_see_previous_version_until_publication(project):
    path = publish(project)
    previous = stored(project)
    path.write_text('Updated source ' * 5000)
    observations = []
    result = build_graph(project, Generator(observe=lambda: observations.append(stored(project))))
    assert result.status == 'complete' and observations
    assert all(item == previous for item in observations)
    assert stored(project)[0] == []


@pytest.mark.parametrize('missing', ['file', 'folder'])
def test_missing_registered_sources_preserve_claims(project, missing):
    path = publish(project)
    previous = stored(project)
    if missing == 'file':
        path.unlink()
    else:
        path.parent.rename(path.parent.with_name('temporarily-offline'))
    result = build_graph(project, Generator())
    assert result.failed and result.calls == 0
    assert result.accounting.documents_removed == 0
    assert stored(project) == previous
    assert 'unavailable' in result.failed[0].problem


def test_unreadable_source_preserves_claims(project, monkeypatch):
    publish(project)
    previous = stored(project)
    def fail(*args, **kwargs):
        raise PermissionError('unreadable')
    monkeypatch.setattr('osintgpt.graph.build.load_snapshot', fail)
    result = build_graph(project, Generator())
    assert result.failed and stored(project) == previous


def test_unregister_removes_only_that_source_and_prunes_orphans(project):
    publish(project)
    other = project.paths.root / 'other.md'
    other.write_text('Alpha funds Beta.')
    corpus = Corpus.load(project.paths.sources)
    corpus.register(other)
    build_graph(project, Generator(CLAIM))
    corpus.unregister('material')
    result = build_graph(project, Generator(), incremental=True)
    assert result.calls == 0 and result.accounting.documents_removed == 1
    edges, entities, documents = stored(project)
    assert len(edges) == 1 and edges[0].ref == 'other.md'
    assert len(entities) == 2 and set(documents) == {'other.md'}


def test_removed_pending_source_discards_checkpoints_but_keeps_audit(project):
    path = project.paths.root / 'material' / 'a.md'
    path.write_text('Alpha funds Beta. ' + 'text ' * 20000)
    from osintgpt.graph import ExtractionOptions
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(json.dumps(proposal('Invented.')), KeyboardInterrupt()),
                    options=ExtractionOptions(repair_quotes=False))
    Corpus.load(project.paths.sources).unregister('material')
    result = build_graph(project, Generator())
    assert result.calls == 0 and result.accounting.documents_removed == 1
    with graph_for(project) as graph:
        assert Checkpoints(graph).summary()['documents'] == 0
        assert len(graph.rejections()) == 1


def test_missing_source_registry_does_not_remove_graph(project):
    publish(project)
    previous = stored(project)
    project.paths.sources.unlink()
    with pytest.raises(ValueError, match='registrations are unavailable'):
        build_graph(project, Generator())
    assert stored(project) == previous


def test_source_modified_during_calls_does_not_publish(project):
    path = publish(project)
    previous = stored(project)
    path.write_text('First revision')
    result = build_graph(project, Generator(observe=lambda: path.write_text('Second revision')))
    assert result.failed and 'changed during' in result.failed[0].problem
    assert stored(project) == previous
    assert build_graph(project, Generator()).accounting.checkpoints_discarded == 1


def test_mapping_change_reprocesses_source(project):
    path = project.paths.root / 'records.csv'
    path.write_text('id,first,second\none,First,Second\n')
    corpus = Corpus.load(project.paths.sources)
    corpus.unregister('material')
    corpus.register(path, FieldMapping(content=('first',), identity='id'))
    assert build_graph(project, Generator()).calls == 1
    corpus.register(path, FieldMapping(content=('second',), identity='id'))
    result = build_graph(project, Generator())
    assert result.calls == 1 and result.accounting.documents_changed == 1


def test_new_scan_with_missing_text_does_not_replace_existing_claims(project, monkeypatch):
    from osintgpt.ingestion.documents import Document
    from osintgpt.ingestion.provenance import TextSpan
    from osintgpt.ingestion.snapshots import TextSnapshot
    path = publish(project)
    previous = stored(project)
    snapshot = TextSnapshot('material/a.md', 'new', 'new',
                            (Document('material/a.md', '', spans=(TextSpan(0, 0, page=1, missing=True),)),))
    monkeypatch.setattr('osintgpt.graph.build.load_snapshot', lambda *args, **kwargs: snapshot)
    monkeypatch.setattr('osintgpt.graph.build.validate_current', lambda *args: None)
    result = build_graph(project, Generator())
    assert result.failed and result.calls == 0 and stored(project) == previous


def test_removal_keeps_shared_entity_with_surviving_name_and_evidence(project):
    corpus = Corpus.load(project.paths.sources)
    corpus.unregister('material')
    for name in ('Alpha', 'Beta'):
        path = project.paths.root / f'{name}.md'
        path.write_text(f'{name} REG 7.')
        corpus.register(path)
        response = {'entities': [{'name': name, 'type': 'org', 'evidence': f'{name} REG 7.',
                                 'identifiers': [{'issuer': 'REG', 'value': '7', 'evidence': 'REG 7'}]}],
                    'edges': []}
        assert build_graph(project, Generator(json.dumps(response))).status == 'complete'
    before = stored(project)[1]
    assert len(before) == 1
    corpus.unregister('Alpha.md')
    result = build_graph(project, Generator())
    entities = stored(project)[1]
    assert result.calls == 0 and len(entities) == 1
    assert entities[0].key == before[0].key and entities[0].name == 'Beta'
    assert entities[0].provenance['source_ref'] == 'Beta.md'
    assert all(item['source_ref'] == 'Beta.md' for item in entities[0].provenance['identity_sources'])


def test_failed_recipe_update_is_flagged_outdated_even_when_source_is_unchanged(project):
    publish(project)
    generator = Generator(RuntimeError('Unavailable'))
    generator.model = 'new-model'
    assert build_graph(project, generator).failed
    assert saved_graph_status(project)['stale']
