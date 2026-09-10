# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_sources.py
# Description: Registered sources, tool-owned canon, and the hash comparison
#   that keeps a re-index cheap.
# =================================================================================

# import modules
import shutil
from pathlib import Path

import pytest

# import osintgpt ingestion
from osintgpt.ingestion import (
    Corpus,
    FieldMapping,
    IndexState,
    Source,
    content_hash
)
from osintgpt.ingestion.sources import MAX_FOLDER_FILES

MAPPING = FieldMapping(content=('content',), identity='record_id')


@pytest.fixture
def project(tmp_path):
    '''A project directory with material inside and outside the corpus.'''
    (tmp_path / 'collected').mkdir()
    (tmp_path / 'collected' / 'first.md').write_text(
        '# First\n\nAssessed material.', encoding='utf-8'
    )
    (tmp_path / 'collected' / 'second.md').write_text(
        '# Second\n\nMore material.', encoding='utf-8'
    )
    (tmp_path / 'collected' / 'notes.zip').write_bytes(b'not a document')
    (tmp_path / 'unregistered.md').write_text('Not corpus.', encoding='utf-8')
    (tmp_path / 'records.csv').write_text(
        'record_id,content\nr1,first record\nr2,second record\n',
        encoding='utf-8'
    )

    return tmp_path


@pytest.fixture
def corpus(project):
    return Corpus.load(project / 'sources.toml')


class TestTheGate:
    def test_nothing_is_corpus_until_registered(self, corpus, project):
        '''
        A vector index is only as good as its signal, so nothing is indexed
        because it happened to be on disk.
        '''
        assert corpus.files(project) == []

    def test_a_registered_folder_covers_what_is_under_it(
        self, corpus, project
    ):
        corpus.register('collected')

        names = {p.name for p in corpus.files(project)}

        assert names == {'first.md', 'second.md'}

    def test_an_unregistered_sibling_stays_out(self, corpus, project):
        corpus.register('collected')

        assert all(
            p.name != 'unregistered.md' for p in corpus.files(project)
        )

    def test_unreadable_extensions_are_skipped_quietly(
        self, corpus, project
    ):
        '''A folder of mixed material is normal, not an error.'''
        corpus.register('collected')

        assert all(p.suffix != '.zip' for p in corpus.files(project))

    def test_a_folder_tracks_files_arriving_later(self, corpus, project):
        corpus.register('collected')
        (project / 'collected' / 'third.md').write_text(
            'Arrived after registration.', encoding='utf-8'
        )

        assert len(corpus.files(project)) == 3

    def test_the_folder_ceiling_is_bounded(self):
        '''
        One registration must not be able to swallow a home directory.
        '''
        assert 0 < MAX_FOLDER_FILES <= 100_000


class TestManagedCanon:
    def test_canon_is_an_effective_source_without_registration(
        self, corpus, project
    ):
        canon = project / 'canon'
        canon.mkdir()
        page = canon / 'index.md'
        page.write_text('Curated synthesis.', encoding='utf-8')

        assert [source.path for source in corpus] == ['canon']
        assert corpus.find('canon').note
        assert corpus.files(project) == [page]
        assert corpus.sources == []

    def test_canon_cannot_be_unregistered(self, corpus, project):
        canon = project / 'canon'
        canon.mkdir()
        page = canon / 'index.md'
        page.write_text('Curated synthesis.', encoding='utf-8')

        assert corpus.unregister('canon') is False
        assert corpus.files(project) == [page]

    def test_no_canon_directory_preserves_the_registered_only_behavior(
        self, corpus, project
    ):
        corpus.register('collected')

        assert [source.path for source in corpus] == ['collected']


class TestRegistration:
    def test_it_persists(self, corpus, project):
        corpus.register('records.csv', MAPPING, note='collected 2026-03')

        reloaded = Corpus.load(project / 'sources.toml')
        source = reloaded.find('records.csv')

        assert source is not None
        assert source.mapping.content == ('content',)
        assert source.note == 'collected 2026-03'

    def test_registering_again_replaces_rather_than_duplicates(
        self, corpus, project
    ):
        corpus.register('records.csv', FieldMapping(content=('content',)))
        corpus.register('records.csv', FieldMapping(content=('other',)))

        assert len(corpus) == 1
        assert corpus.find('records.csv').mapping.content == ('other',)

    def test_unregistering_leaves_the_files_alone(self, corpus, project):
        corpus.register('collected')

        assert corpus.unregister('collected') is True
        assert (project / 'collected' / 'first.md').exists()
        assert corpus.files(project) == []

    def test_unregistering_something_absent_reports_a_miss(self, corpus):
        assert corpus.unregister('never-registered') is False

    def test_the_file_says_what_it_is_for(self, corpus, project):
        corpus.register('collected')
        text = (project / 'sources.toml').read_text(encoding='utf-8')

        assert text.lstrip().startswith('#')
        assert 'happened to be on disk' in text


class TestPortableSourcePaths:
    def test_a_project_local_absolute_path_is_stored_relative(
        self, corpus, project
    ):
        source = corpus.register(project / 'collected')
        reloaded = Corpus.load(project / 'sources.toml')

        assert source.path == 'collected'
        assert reloaded.sources[0].path == 'collected'

    def test_an_external_path_stays_absolute_and_resolves(
        self, corpus, project
    ):
        external = project.parent / f'{project.name}-external.md'
        external.write_text('External material.', encoding='utf-8')

        source = corpus.register(external)

        assert Path(source.path).is_absolute()
        assert corpus.files(project) == [external.resolve()]

    def test_a_moved_project_keeps_its_registered_files(
        self, corpus, project
    ):
        corpus.register(project / 'collected')
        moved = project.parent / f'{project.name}-moved'

        shutil.move(project, moved)
        reloaded = Corpus.load(moved / 'sources.toml')

        assert {path.name for path in reloaded.files(moved)} == {
            'first.md', 'second.md'
        }

    def test_an_existing_absolute_source_still_loads(
        self, project
    ):
        absolute = (project / 'collected').resolve().as_posix()
        Corpus(
            path=project / 'sources.toml',
            sources=[Source(path=absolute)]
        ).save()

        reloaded = Corpus.load(project / 'sources.toml')

        assert reloaded.sources[0].path == absolute
        assert {path.name for path in reloaded.files(project)} == {
            'first.md', 'second.md'
        }

    def test_relative_and_absolute_registration_do_not_duplicate(
        self, corpus, project
    ):
        corpus.register('collected')
        corpus.register(project / 'collected')

        assert len(corpus) == 1
        assert corpus.sources[0].path == 'collected'

    def test_absolute_lookup_and_removal_match_relative_storage(
        self, corpus, project
    ):
        source = corpus.register(project / 'collected')

        assert corpus.find(project / 'collected') == source
        assert corpus.unregister(project / 'collected') is True

    def test_parent_segments_are_resolved_before_comparison(
        self, corpus, project
    ):
        nested = project / 'nested'
        nested.mkdir()

        source = corpus.register(nested / '..' / 'collected')

        assert source.path == 'collected'

    def test_mapping_survives_a_project_move(self, corpus, project):
        corpus.register(project / 'records.csv', MAPPING)
        moved = project.parent / f'{project.name}-moved'

        shutil.move(project, moved)
        reloaded = Corpus.load(moved / 'sources.toml')

        mapping = reloaded.mapping_for(moved / 'records.csv', moved)

        assert mapping == MAPPING


class TestOverlap:
    def test_a_file_covered_twice_is_read_once(self, corpus, project):
        corpus.register('collected')
        corpus.register('collected/first.md')

        paths = corpus.files(project)

        assert len(paths) == len({p.resolve() for p in paths})

    def test_a_direct_registration_beats_the_folder(self, corpus, project):
        '''
        A spreadsheet inside a registered folder can name its own fields.
        '''
        (project / 'collected' / 'rows.csv').write_text(
            'record_id,content\nr1,text\n', encoding='utf-8'
        )
        corpus.register('collected')
        corpus.register('collected/rows.csv', MAPPING)

        mapping = corpus.mapping_for(
            project / 'collected' / 'rows.csv', project
        )

        assert mapping.content == ('content',)

    def test_a_file_with_no_mapping_gets_an_empty_one(self, corpus, project):
        corpus.register('collected')

        mapping = corpus.mapping_for(
            project / 'collected' / 'first.md', project
        )

        assert mapping.content == ()


class TestIndexState:
    @pytest.fixture
    def state(self, project):
        return IndexState.load(project / 'index.toml')

    def test_everything_is_new_the_first_time(self, state, corpus, project):
        corpus.register('collected')

        plan = state.plan(corpus.files(project), project)

        assert len(plan.added) == 2
        assert plan.changed == []
        assert plan.removed == []

    def test_an_unchanged_document_costs_nothing(
        self, state, corpus, project
    ):
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        plan = state.plan(corpus.files(project), project)

        assert plan.work == []
        assert len(plan.unchanged) == 2
        assert plan.is_empty

    def test_an_edited_document_is_re_indexed(self, state, corpus, project):
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        (project / 'collected' / 'first.md').write_text(
            '# First\n\nRevised material.', encoding='utf-8'
        )
        plan = state.plan(corpus.files(project), project)

        assert [p.name for p in plan.changed] == ['first.md']
        assert len(plan.unchanged) == 1

    def test_a_touched_file_with_the_same_bytes_is_unchanged(
        self, state, corpus, project
    ):
        '''
        The comparison is on content, not modification time: re-saving a file
        without editing it should cost nothing.
        '''
        corpus.register('collected')
        path = project / 'collected' / 'first.md'
        state.record(path, project, chunks=1)

        original = path.read_text(encoding='utf-8')
        path.write_text(original, encoding='utf-8')

        assert state.plan([path], project).unchanged == [path]

    def test_a_deleted_document_is_reported_for_removal(
        self, state, corpus, project
    ):
        '''
        Vectors nobody covers are invisible to a corpus walk and still
        returned by a search, which is the worst of both.
        '''
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        (project / 'collected' / 'second.md').unlink()
        plan = state.plan(corpus.files(project), project)

        assert plan.removed == ['collected/second.md']

    def test_unregistering_marks_its_documents_removed(
        self, state, corpus, project
    ):
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        corpus.unregister('collected')
        plan = state.plan(corpus.files(project), project)

        assert len(plan.removed) == 2

    def test_force_re_indexes_everything(self, state, corpus, project):
        '''
        The escape hatch for a chunker change, which alters output without
        altering any document.
        '''
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        plan = state.plan(corpus.files(project), project, force=True)

        assert len(plan.changed) == 2
        assert plan.unchanged == []

    def test_state_survives_a_reload(self, state, corpus, project):
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=3)
        state.save()

        reloaded = IndexState.load(project / 'index.toml')

        assert len(reloaded) == 2
        assert reloaded.chunks == 6
        assert reloaded.plan(corpus.files(project), project).is_empty

    def test_refs_are_relative_to_the_project(
        self, state, corpus, project
    ):
        '''
        Relative refs let a project be moved or handed on without every
        document looking new.
        '''
        corpus.register('collected')
        entry = state.record(
            project / 'collected' / 'first.md', project, chunks=1
        )

        assert entry.ref == 'collected/first.md'

    def test_forgetting_drops_entries(self, state, corpus, project):
        corpus.register('collected')
        for path in corpus.files(project):
            state.record(path, project, chunks=1)

        assert state.forget(['collected/first.md']) == 1
        assert len(state) == 1

    def test_the_summary_says_what_would_happen(
        self, state, corpus, project
    ):
        corpus.register('collected')

        assert '2 new' in state.plan(corpus.files(project), project).summary
        assert 'nothing to index' in IndexState(
            path=project / 'index.toml'
        ).plan([], project).summary


class TestContentHash:
    def test_the_same_content_hashes_the_same(self):
        assert content_hash('abc') == content_hash(b'abc')

    def test_different_content_hashes_differently(self):
        assert content_hash('abc') != content_hash('abd')

    def test_it_is_sha256(self):
        assert len(content_hash('abc')) == 64
