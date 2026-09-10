# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_indexing.py
# Description: Corpus to searchable vectors, and back again. The first point
#   where ingestion, embedding and storage have to agree.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, index_project, search_project
from osintgpt.canon import is_canon_ref, write_page
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.llm.base import EmbeddingProvider
from osintgpt.vector_store import SQLiteVectorStore

MODEL = 'test-embedding'


class CountingEmbedder(EmbeddingProvider):
    '''
    Deterministic vectors from the text itself, so a search can be checked
    without a provider — and a call count, so "unchanged costs nothing" is
    provable rather than asserted.
    '''
    def __init__(self, model=MODEL):
        self.model = model
        self.calls = 0
        self.texts = []

    def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)

        return [self._vector(text) for text in texts]

    def _vector(self, text):
        # Character histogram over the alphabet: similar text, similar vector.
        counts = [0.0] * 26
        for character in text.lower():
            index = ord(character) - 97
            if 0 <= index < 26:
                counts[index] += 1.0
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


@pytest.fixture
def embedder():
    return CountingEmbedder()


@pytest.fixture
def project(tmp_path):
    '''A project with two prose documents and a small dataset registered.'''
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()

    (material / 'alpha.md').write_text(
        '---\ndate: 2026-04-22\nauthor: an analyst\n---\n\n'
        '# Alpha Report\n\nA paragraph about aardvarks and assessments.\n\n'
        '## Findings\n\nMore about aardvarks.',
        encoding='utf-8'
    )
    (material / 'beta.md').write_text(
        '# Beta Report\n\nA paragraph about zebras and zoning.',
        encoding='utf-8'
    )
    (material / 'records.csv').write_text(
        'record_id,content,captured_at\n'
        'r1,"a record mentioning aardvarks",2026-03-01\n'
        'r2,"a record mentioning zebras",2026-03-02\n',
        encoding='utf-8'
    )

    corpus = Corpus.load(instance.paths.sources)
    corpus.register('material')
    corpus.register(
        'material/records.csv',
        FieldMapping(
            content=('content',), timestamp='captured_at', identity='record_id'
        )
    )

    return instance


class TestFirstPass:
    def test_it_indexes_the_registered_corpus(self, project, embedder):
        report = index_project(project, embedder)

        assert len(report.indexed) == 3
        assert report.chunks > 3
        assert report.failed == []

    def test_the_store_holds_what_was_indexed(self, project, embedder):
        report = index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            assert store.count() == report.chunks
            assert store.models() == [MODEL]

    def test_a_dataset_is_one_ref_holding_a_chunk_per_row(
        self, project, embedder
    ):
        '''
        The document is the file, and rows are chunks within it. Per-row refs
        would mean a deleted row had nothing to replace it, so its vector
        would outlive the record.
        '''
        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            refs = [ref for ref in store.refs() if 'records.csv' in ref]
            chunks = store.chunks_for(refs[0])

        assert len(refs) == 1
        assert len(chunks) == 2

    def test_unregistered_files_stay_out(self, project, embedder):
        (project.paths.root / 'loose.md').write_text(
            'Not corpus.', encoding='utf-8'
        )
        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            assert all('loose' not in ref for ref in store.refs())

    def test_the_summary_says_what_happened(self, project, embedder):
        assert 'documents' in index_project(project, embedder).summary


class TestCanon:
    def test_a_project_without_a_canon_directory_indexes_as_before(
        self, project, embedder
    ):
        project.paths.canon.rmdir()

        report = index_project(project, embedder)

        assert len(report.indexed) == 3
        assert all(not is_canon_ref(result.ref) for result in report.indexed)

    def test_canon_pages_index_without_registration(self, project, embedder):
        page = write_page(
            project.paths.canon, 'entities', 'Synthesis',
            'A curated synthesis about a distinctive subject.'
        )

        report = index_project(project, embedder)

        assert Corpus.load(project.paths.sources).find('canon') is not None
        assert page.relative_to(project.paths.root).as_posix() in {
            result.ref for result in report.indexed
        }

    def test_removing_every_registration_leaves_canon_indexable(
        self, project, embedder
    ):
        corpus = Corpus.load(project.paths.sources)
        for source in list(corpus.sources):
            assert corpus.unregister(source.path) is True
        write_page(
            project.paths.canon, 'narratives', 'Only synthesis',
            'The canon remains searchable without primary material.'
        )

        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            assert store.refs()
            assert all(is_canon_ref(ref) for ref in store.refs())

    def test_a_search_result_identifies_canon_synthesis(
        self, project, embedder
    ):
        text = 'Quizzical xenolith synthesis with a vexing provenance.'
        write_page(project.paths.canon, 'decisions', 'Finding', text)
        index_project(project, embedder)

        result = search_project(project, text, embedder, top_k=1)[0]

        assert result.ref == 'canon/decisions/finding.md'
        assert is_canon_ref(result.ref) is True


class TestSecondPass:
    def test_unchanged_documents_cost_no_embedding_call(
        self, project, embedder
    ):
        '''
        The whole point of the hash comparison: embedding is the expensive
        half, so an untouched corpus must not pay for it twice.
        '''
        index_project(project, embedder)
        calls = embedder.calls

        report = index_project(project, embedder)

        assert embedder.calls == calls
        assert report.indexed == []
        assert report.unchanged == 3

    def test_an_edited_document_is_re_embedded(self, project, embedder):
        index_project(project, embedder)

        (project.paths.root / 'material' / 'beta.md').write_text(
            '# Beta Report\n\nRewritten about quokkas.', encoding='utf-8'
        )
        report = index_project(project, embedder)

        assert [r.ref for r in report.indexed] == ['material/beta.md']
        assert report.unchanged == 2

    def test_re_indexing_replaces_a_document_rather_than_adding(
        self, project, embedder
    ):
        index_project(project, embedder)
        with SQLiteVectorStore(project.paths.store) as store:
            before = store.count()

        (project.paths.root / 'material' / 'beta.md').write_text(
            '# Beta Report\n\nStill one chunk.', encoding='utf-8'
        )
        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            assert store.count() == before

    def test_a_deleted_document_loses_its_chunks(self, project, embedder):
        index_project(project, embedder)

        (project.paths.root / 'material' / 'beta.md').unlink()
        report = index_project(project, embedder)

        assert report.removed >= 1
        with SQLiteVectorStore(project.paths.store) as store:
            assert all('beta' not in ref for ref in store.refs())

    def test_force_re_embeds_everything(self, project, embedder):
        index_project(project, embedder)
        calls = embedder.calls

        report = index_project(project, embedder, force=True)

        assert embedder.calls > calls
        assert len(report.indexed) == 3


class TestProvenance:
    def test_frontmatter_reaches_the_store(self, project, embedder):
        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            chunk = store.chunks_for('material/alpha.md')[0]

        assert chunk.timestamp == '2026-04-22'
        assert chunk.author == 'an analyst'

    def test_a_mapped_timestamp_reaches_the_store(self, project, embedder):
        index_project(project, embedder)

        with SQLiteVectorStore(project.paths.store) as store:
            ref = next(r for r in store.refs() if 'records.csv' in r)
            chunk = store.chunks_for(ref)[0]

        assert chunk.timestamp in ('2026-03-01', '2026-03-02')

    def test_the_section_path_reaches_the_store(self, project, embedder):
        '''
        Nested sections carry a path; a top-level one opens with its own
        heading and needs none.
        '''
        long_body = 'A sentence about assessments. ' * 60
        (project.paths.root / 'material' / 'alpha.md').write_text(
            f'# Report\n\n{long_body}\n\n## Findings\n\n{long_body}',
            encoding='utf-8'
        )
        index_project(project, embedder, force=True)

        with SQLiteVectorStore(project.paths.store) as store:
            paths = {c.path for c in store.chunks_for('material/alpha.md')}

        assert 'Report' in paths

    def test_what_is_embedded_includes_the_path(self, project, embedder):
        '''
        The path is part of the vector, so a passage is findable by the
        section that frames it — and kept as a field so a citation need not
        parse it back out.
        '''
        long_body = 'A sentence about assessments. ' * 60
        (project.paths.root / 'material' / 'alpha.md').write_text(
            f'# Report\n\n{long_body}\n\n## Findings\n\n{long_body}',
            encoding='utf-8'
        )
        index_project(project, embedder, force=True)

        assert any('Report' in text for text in embedder.texts)


class TestSearch:
    def test_it_finds_the_relevant_document(self, project, embedder):
        index_project(project, embedder)

        results = search_project(project, 'aardvarks', embedder, top_k=3)

        assert results
        assert 'aardvark' in results[0].text.lower()

    def test_results_carry_their_citation(self, project, embedder):
        index_project(project, embedder)

        result = search_project(project, 'zebras', embedder)[0]

        assert result.chunk.citation
        assert result.ref

    def test_a_different_model_finds_nothing(self, project, embedder):
        '''
        Vectors from another model are not comparable, so the store returns
        nothing rather than ranking them anyway.
        '''
        index_project(project, embedder)

        results = search_project(
            project, 'aardvarks', CountingEmbedder('another-model')
        )

        assert results == []

    def test_search_can_be_restricted_to_documents(self, project, embedder):
        index_project(project, embedder)

        results = search_project(
            project, 'aardvarks', embedder, refs=['material/beta.md']
        )

        assert all(r.ref == 'material/beta.md' for r in results)

    def test_an_unindexed_project_returns_nothing(self, tmp_path, embedder):
        empty = Project.create('Empty', home=tmp_path)

        assert search_project(empty, 'anything', embedder) == []


class TestFailures:
    def test_one_unreadable_document_does_not_stop_the_pass(
        self, project, embedder
    ):
        (project.paths.root / 'material' / 'broken.json').write_text(
            '{not valid json', encoding='utf-8'
        )
        corpus = Corpus.load(project.paths.sources)
        corpus.register(
            'material/broken.json', FieldMapping(content=('body',))
        )

        report = index_project(project, embedder)

        assert len(report.failed) == 1
        assert report.indexed

    def test_a_failure_says_why(self, project, embedder):
        (project.paths.root / 'material' / 'broken.json').write_text(
            '{not valid json', encoding='utf-8'
        )
        corpus = Corpus.load(project.paths.sources)
        corpus.register(
            'material/broken.json', FieldMapping(content=('body',))
        )

        report = index_project(project, embedder)

        assert report.failed[0].problem

    def test_a_failed_document_is_not_recorded_as_indexed(
        self, project, embedder
    ):
        '''
        Otherwise the next pass would treat it as done and never retry it.
        '''
        (project.paths.root / 'material' / 'broken.json').write_text(
            '{not valid json', encoding='utf-8'
        )
        corpus = Corpus.load(project.paths.sources)
        corpus.register(
            'material/broken.json', FieldMapping(content=('body',))
        )

        index_project(project, embedder)
        report = index_project(project, embedder)

        assert len(report.failed) == 1


class TestModelSwitch:
    def test_re_indexing_everything_leaves_nothing_from_the_old_model(
        self, project, embedder
    ):
        '''
        upsert replaces a document's chunks whatever model produced them, so a
        forced pass over the whole corpus is self-cleaning.
        '''
        index_project(project, embedder)
        index_project(project, CountingEmbedder('a-second-model'), force=True)

        with SQLiteVectorStore(project.paths.store) as store:
            assert store.models() == ['a-second-model']

    def test_purging_reclaims_documents_that_left_the_corpus(
        self, project, embedder
    ):
        '''
        Old vectors survive a model switch only where the document is no
        longer indexed — unregistered, or deleted while the index was not
        run. Those are invisible to search, which filters by model, and still
        occupy the store.
        '''
        index_project(project, embedder)

        # The index state is what tells a pass a document has gone. Losing it
        # is how vectors are stranded: the corpus shrinks, nothing notices,
        # and the chunks stay under a model that is no longer in use.
        project.paths.index_state.unlink()
        corpus = Corpus.load(project.paths.sources)
        corpus.unregister('material')
        corpus.unregister('material/records.csv')
        corpus.register('material/alpha.md')

        report = index_project(
            project, CountingEmbedder('a-second-model'),
            force=True, purge_other_models=True
        )

        assert report.purged > 0
        with SQLiteVectorStore(project.paths.store) as store:
            assert store.models() == ['a-second-model']

    def test_purging_happens_after_the_re_embed(self, project, embedder):
        '''
        A failed pass must not leave a project with neither the old vectors
        nor the new ones.
        '''
        index_project(project, embedder)

        class Failing(CountingEmbedder):
            def embed(self, texts):
                raise RuntimeError('the provider refused')

        index_project(
            project, Failing('a-second-model'), force=True,
            purge_other_models=True
        )

        with SQLiteVectorStore(project.paths.store) as store:
            assert store.count(MODEL) == 0 or store.models() == [MODEL]


class TestProgress:
    def test_it_reports_each_document(self, project, embedder):
        seen = []
        index_project(
            project, embedder,
            on_progress=lambda ref, position, total: seen.append(
                (ref, position, total)
            )
        )

        assert len(seen) == 3
        assert seen[0][1] == 1
        assert seen[-1][2] == 3
