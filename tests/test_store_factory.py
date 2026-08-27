# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_store_factory.py
# Description: Choosing a backend from settings. The claim "scaling up is
#   configuration" is only true if the choice is actually read.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.vector_store import SQLiteVectorStore, store_for


@pytest.fixture
def project(tmp_path):
    return Project.create('Case', home=tmp_path)


class TestChoosingABackend:
    def test_sqlite_is_the_default(self, project):
        with store_for(project) as store:
            assert isinstance(store, SQLiteVectorStore)

    def test_it_reads_the_project_setting(self, project):
        project.with_settings(storage_backend='sqlite').save()

        with store_for(Project.load(project.paths.root)) as store:
            assert isinstance(store, SQLiteVectorStore)

    def test_an_unknown_backend_is_refused(self, project):
        '''
        Failing here beats falling back to the default: a project configured
        for a server that quietly indexed to a local file would look fine and
        be wrong.
        '''
        project.with_settings(storage_backend='not-a-backend').save()

        with pytest.raises(ValueError, match='unknown storage backend'):
            store_for(Project.load(project.paths.root))

    def test_the_name_is_case_and_space_tolerant(self, project):
        project.with_settings(storage_backend='  SQLite ').save()

        with store_for(Project.load(project.paths.root)) as store:
            assert isinstance(store, SQLiteVectorStore)

    def test_qdrant_is_a_known_backend(self):
        from osintgpt.vector_store import BACKENDS

        assert 'qdrant' in BACKENDS
