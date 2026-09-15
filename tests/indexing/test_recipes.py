'''Configuration invalidation, destination reconciliation and upgrade behavior.'''

import shutil
import sqlite3
from dataclasses import replace

import pytest

from test_checkpoints import Endpoint, project, provider, stored_vectors
from osintgpt import Project, index_project
from osintgpt.ingestion import Corpus, FieldMapping, IndexState
from osintgpt.ingestion import chunking, recipes
from osintgpt.ingestion.checkpoints import fingerprint
from osintgpt.llm.usage import UsageRecorder
from osintgpt.vector_store import SQLiteVectorStore


def openai(endpoint=None):
    embedder = provider('openai', endpoint or Endpoint())
    embedder.client.base_url = 'https://api.openai.com/v1/'
    return embedder


def make_legacy(project):
    state = IndexState.load(project.paths.index_state)
    state.documents = {
        ref: replace(entry, recipe='', destination='', embedding_model='')
        for ref, entry in state.documents.items()
    }
    state.save()
    with sqlite3.connect(project.paths.store) as connection:
        connection.execute("UPDATE chunks SET document_ref = ''")


@pytest.mark.parametrize('change', [
    'model', 'provider', 'endpoint', 'floor', 'identity', 'metadata',
    'chunk_version', 'chunk_size', 'ingestion_version'
])
def test_completed_file_is_invalidated_without_force(project, monkeypatch, change):
    assert index_project(project, openai()).chunks == 250
    initial = IndexState.load(project.paths.index_state).documents['records.csv']
    embedder = openai()
    if change == 'model':
        embedder.model = 'text-embedding-3-large'
    elif change == 'provider':
        embedder.provider = 'voyage'
    elif change == 'endpoint':
        embedder.client.base_url = 'https://different.invalid/v1'
    elif change in ('floor', 'identity', 'metadata'):
        mapping = FieldMapping(content=('message',), identity='id')
        mapping = replace(mapping, **{
            'floor': {'min_chars': 10},
            'identity': {'identity': 'message'},
            'metadata': {'metadata': ('id',)}
        }[change])
        Corpus.load(project.paths.sources).register('records.csv', mapping)
    elif change == 'chunk_version':
        monkeypatch.setattr(chunking, 'CHUNKING_VERSION', 2)
    elif change == 'chunk_size':
        monkeypatch.setattr(chunking, 'MAX_CHARS', 2000)
    else:
        monkeypatch.setattr(recipes, 'INGESTION_VERSION', 2)
    report = index_project(Project.load(project.paths.root), embedder)
    assert not report.failed and report.chunks == 250
    assert len(embedder.client.embeddings.calls) == 3
    current = IndexState.load(project.paths.index_state).documents['records.csv']
    assert current.hash == initial.hash and current.recipe != initial.recipe
    assert index_project(project, embedder).unchanged == 1


def test_higher_floor_removes_excluded_records_without_embedding(project):
    index_project(project, openai())
    Corpus.load(project.paths.sources).register(
        'records.csv', FieldMapping(content=('message',), identity='id', min_chars=120)
    )
    embedder = openai()
    report = index_project(project, embedder)
    assert not report.failed and len(report.indexed) == 1 and report.chunks == 0
    assert embedder.client.embeddings.calls == [] and stored_vectors(project) == []
    assert index_project(project, openai()).unchanged == 1


def test_operating_controls_do_not_invalidate_completed_work(project):
    index_project(project, openai())
    embedder = openai()
    embedder.batch_size = 37
    embedder.recorder = UsageRecorder(cost_ceiling_usd=0.01)
    embedder.client.api_key = 'new-credential'
    report = index_project(project, embedder)
    assert report.unchanged == 1 and embedder.client.embeddings.calls == []
    assert embedder.recorder.calls == 0
    assert 'new-credential' not in project.paths.index_state.read_text()
    assert 'https://' not in project.paths.index_state.read_text()


@pytest.mark.parametrize('damage', ['delete_file', 'missing_chunk', 'wrong_model', 'sequence', 'vector'])
def test_completion_is_reconciled_with_actual_destination(project, damage):
    index_project(project, openai())
    if damage == 'delete_file':
        project.paths.store.unlink()
    else:
        commands = {
            'missing_chunk': 'DELETE FROM chunks WHERE sequence = 10',
            'wrong_model': "UPDATE chunks SET embedding_model = 'another-model'",
            'sequence': 'UPDATE chunks SET sequence = 999 WHERE sequence = 10',
            'vector': "UPDATE chunks SET vector = X'00' WHERE sequence = 10"
        }
        with sqlite3.connect(project.paths.store) as connection:
            connection.execute(commands[damage])
    embedder = openai()
    report = index_project(project, embedder)
    assert not report.failed and report.chunks == 250 and report.unchanged == 0
    assert len(stored_vectors(project)) == 250


def test_changing_destination_does_not_reuse_completion_or_pending_job(project, tmp_path):
    index_project(project, openai())
    original = stored_vectors(project)
    with SQLiteVectorStore(tmp_path / 'other.sqlite') as other:
        embedder = openai()
        assert index_project(project, embedder, store=other).chunks == 250
        assert len(embedder.client.embeddings.calls) == 3
    assert stored_vectors(project) == original
    # A pending replacement belongs to its selected destination, too.
    index_project(project, openai(Endpoint(fail_at=2)))
    with SQLiteVectorStore(tmp_path / 'third.sqlite') as third:
        embedder = openai()
        assert index_project(project, embedder, store=third).chunks == 250
        assert embedder.client.embeddings.calls[0][0] == 'Record 0 contains evidence.'


def test_project_move_keeps_local_destination_and_record_refs_stable(project, tmp_path):
    # The fixture initially registered an absolute source: make it portable.
    corpus = Corpus.load(project.paths.sources)
    corpus.unregister(str(project.paths.root / 'records.csv'))
    corpus.register('records.csv', FieldMapping(content=('message',), identity='id'))
    index_project(project, openai())
    moved = tmp_path / 'moved-case'
    shutil.copytree(project.paths.root, moved)
    embedder = openai()
    assert index_project(Project.load(moved), embedder).unchanged == 1
    assert embedder.client.embeddings.calls == []
    with SQLiteVectorStore(moved / 'store.sqlite') as store:
        assert store.chunks_for('records.csv')[99].document_ref == 'records.csv#99'


def test_failed_automatic_model_change_keeps_old_index_then_resumes(project):
    index_project(project, openai())
    original = stored_vectors(project)
    before = IndexState.load(project.paths.index_state).documents['records.csv']
    embedder = openai(Endpoint(fail_at=2))
    embedder.model = 'text-embedding-3-large'
    assert index_project(project, embedder).failed
    assert stored_vectors(project) == original
    assert IndexState.load(project.paths.index_state).documents['records.csv'] == before
    resumed = openai()
    resumed.model = embedder.model
    assert index_project(project, resumed).chunks == 250
    assert resumed.client.embeddings.calls[0][0] == 'Record 100 contains evidence.'


def test_matching_legacy_openai_is_adopted_without_embedding(project):
    index_project(project, openai())
    before = stored_vectors(project)
    make_legacy(project)
    embedder = openai()
    report = index_project(project, embedder)
    assert report.adopted == 1 and report.unchanged == 1 and not report.failed
    assert report.notices and embedder.client.embeddings.calls == []
    assert stored_vectors(project) == before
    assert IndexState.load(project.paths.index_state).documents['records.csv'].recipe
    assert index_project(project, embedder).adopted == 0


@pytest.mark.parametrize('mismatch', ['model', 'endpoint', 'metadata', 'content', 'provider', 'missing'])
def test_unverifiable_legacy_preserves_data_and_requires_explicit_rebuild(project, mismatch):
    index_project(project, openai())
    make_legacy(project)
    embedder = openai()
    if mismatch == 'model':
        embedder.model = 'text-embedding-3-large'
    elif mismatch == 'endpoint':
        embedder.client.base_url = 'https://custom.invalid/v1'
    elif mismatch == 'provider':
        embedder.provider = 'voyage'
    else:
        with sqlite3.connect(project.paths.store) as connection:
            connection.execute({
                'metadata': "UPDATE chunks SET metadata = '{\"id\": \"unexpected\"}' WHERE sequence = 10",
                'content': "UPDATE chunks SET text = 'different' WHERE sequence = 10",
                'missing': 'DELETE FROM chunks WHERE sequence = 10'
            }[mismatch])
    before = stored_vectors(project)
    report = index_project(project, embedder)
    assert report.failed and '--force' in report.failed[0].problem
    assert report.unchanged == 0 and embedder.client.embeddings.calls == []
    assert stored_vectors(project) == before
    assert not IndexState.load(project.paths.index_state).documents['records.csv'].recipe
    assert index_project(project, embedder, force=True).chunks == 250


def test_first_journal_format_is_upgraded_without_repeating_saved_batches(project):
    embedder = openai(Endpoint(fail_at=2))
    index_project(project, embedder)
    path = project.paths.root / 'records.csv'
    mapping = Corpus.load(project.paths.sources).mapping_for(path, project.paths.root)
    chunks, texts, refs = recipes.prepare_chunks(path, 'records.csv', mapping, embedder)
    signature = fingerprint(recipes.content_hash(path.read_bytes()), mapping, embedder, chunks, texts, refs)
    with sqlite3.connect(project.paths.index_journal) as connection:
        connection.execute('UPDATE jobs SET fingerprint = ?', (signature,))
    resumed = openai()
    assert index_project(project, resumed).chunks == 250
    assert resumed.client.embeddings.calls[0][0] == 'Record 100 contains evidence.'


def test_gemini_resolved_task_and_template_affect_recipe(monkeypatch):
    from osintgpt.llm.gemini import INSTRUCTIONS, TASK_TYPES
    from osintgpt.llm.base import EmbeddingPurpose
    embedder = provider('gemini', Endpoint())
    mapping = FieldMapping(content=('message',))
    initial = recipes.recipe_for(mapping, embedder)
    assert recipes.supports_legacy_checkpoint(embedder)
    monkeypatch.setitem(TASK_TYPES, EmbeddingPurpose.DOCUMENT, 'SEMANTIC_SIMILARITY')
    assert recipes.recipe_for(mapping, embedder) != initial
    assert not recipes.supports_legacy_checkpoint(embedder)
    embedder.task_style = 'instruction'
    instructed = recipes.recipe_for(mapping, embedder)
    assert instructed != initial
    assert recipes.supports_legacy_checkpoint(embedder)
    monkeypatch.setitem(INSTRUCTIONS, EmbeddingPurpose.DOCUMENT, 'passage: {text}')
    assert recipes.recipe_for(mapping, embedder) != instructed
    assert not recipes.supports_legacy_checkpoint(embedder)


def test_custom_options_are_hashed_but_not_inferred_for_old_checkpoints():
    class Customized(type(openai())):
        def indexing_options(self):
            return {**super().indexing_options(), 'dimensions': self.dimensions}
    embedder = Customized.__new__(Customized)
    embedder.__dict__.update(openai().__dict__)
    embedder.dimensions = 1536
    mapping = FieldMapping(content=('message',))
    first = recipes.recipe_for(mapping, embedder)
    embedder.dimensions = 256
    assert recipes.recipe_for(mapping, embedder) != first
    assert not recipes.supports_legacy_checkpoint(embedder)
    assert not embedder.can_adopt_legacy_index()
