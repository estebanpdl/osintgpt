# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_cross_project.py
# Description: Searching several projects at once — which ones may be merged,
#   which are dropped, and whether the caller is told.
# =================================================================================

# import modules
import pytest

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL

# import osintgpt projects
from osintgpt.projects import Project, ProjectSettings
from osintgpt.projects.cross_project import (
    MISMATCH,
    embedding_model_of,
    search_projects,
    select_projects
)

LARGE = 'text-embedding-3-large'
SMALL = 'text-embedding-3-small'


@pytest.fixture
def make_project(tmp_path):
    def factory(name, embedding_model=''):
        return Project.create(
            name, home=tmp_path,
            settings=ProjectSettings(embedding_model=embedding_model)
        )

    return factory


class TestEmbeddingModelOf:
    def test_uses_the_project_choice(self, make_project):
        assert embedding_model_of(make_project('A', LARGE)) == LARGE

    def test_falls_back_to_the_library_default(self, make_project):
        assert embedding_model_of(make_project('A')) == DEFAULT_EMBEDDING_MODEL

    def test_user_defaults_fill_an_unset_project(self, make_project):
        model = embedding_model_of(
            make_project('A'), ProjectSettings(embedding_model=LARGE)
        )

        assert model == LARGE


class TestSelect:
    def test_matching_projects_are_all_included(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', LARGE)]
        selection = select_projects(projects)

        assert selection.embedding_model == LARGE
        assert len(selection.included) == 2
        assert selection.excluded == []

    def test_mismatched_projects_are_excluded(self, make_project):
        projects = [
            make_project('A', LARGE),
            make_project('B', LARGE),
            make_project('C', SMALL)
        ]
        selection = select_projects(projects)

        assert selection.embedding_model == LARGE
        assert [p.slug for p in selection.included] == ['a', 'b']
        assert [e.slug for e in selection.excluded] == ['c']
        assert selection.excluded[0].reason == MISMATCH
        assert selection.excluded[0].detail == SMALL

    def test_the_majority_model_wins(self, make_project):
        projects = [
            make_project('A', SMALL),
            make_project('B', LARGE),
            make_project('C', LARGE)
        ]
        selection = select_projects(projects)

        assert selection.embedding_model == LARGE
        assert [e.slug for e in selection.excluded] == ['a']

    def test_a_tie_keeps_the_first_listed(self, make_project):
        projects = [make_project('A', SMALL), make_project('B', LARGE)]
        selection = select_projects(projects)

        assert selection.embedding_model == SMALL
        assert [e.slug for e in selection.excluded] == ['b']

    def test_an_explicit_model_overrides_the_majority(self, make_project):
        projects = [
            make_project('A', LARGE),
            make_project('B', LARGE),
            make_project('C', SMALL)
        ]
        selection = select_projects(projects, embedding_model=SMALL)

        assert [p.slug for p in selection.included] == ['c']
        assert len(selection.excluded) == 2

    def test_an_explicit_model_nobody_uses_includes_nothing(self, make_project):
        selection = select_projects(
            [make_project('A', LARGE)], embedding_model='nomic-embed-text'
        )

        assert selection.included == []
        assert len(selection.excluded) == 1

    def test_unset_projects_agree_on_the_library_default(self, make_project):
        selection = select_projects([make_project('A'), make_project('B')])

        assert selection.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert len(selection.included) == 2

    def test_duplicates_are_counted_once(self, make_project):
        project = make_project('A', LARGE)
        selection = select_projects([project, project])

        assert len(selection.included) == 1
        assert selection.total == 1

    def test_no_projects_selects_nothing(self):
        selection = select_projects([])

        assert selection.included == []
        assert selection.excluded == []
        assert selection.notice == ''


class TestNotice:
    def test_is_empty_when_nothing_was_dropped(self, make_project):
        selection = select_projects([make_project('A', LARGE)])

        assert selection.notice == ''

    def test_names_the_projects_and_the_reason(self, make_project):
        projects = [
            make_project('Alpha', LARGE),
            make_project('Beta', SMALL),
            make_project('Gamma', 'nomic-embed-text')
        ]
        notice = select_projects(projects).notice

        assert notice.startswith('2 of 3 projects skipped')
        assert MISMATCH in notice
        assert 'beta' in notice and 'gamma' in notice
        assert SMALL in notice and 'nomic-embed-text' in notice


class TestSearch:
    def test_merges_by_score_across_projects(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', LARGE)]
        scores = {'a': [(0.9, 'a-high'), (0.3, 'a-low')], 'b': [(0.7, 'b-mid')]}

        results = search_projects(projects, lambda p: scores[p.slug])

        assert [hit.payload for hit in results] == ['a-high', 'b-mid', 'a-low']
        assert [hit.project_slug for hit in results] == ['a', 'b', 'a']

    def test_never_queries_an_excluded_project(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', SMALL)]
        asked = []

        def query(project):
            asked.append(project.slug)
            return [(0.5, project.slug)]

        results = search_projects(projects, query)

        assert asked == ['a']
        assert len(results) == 1

    def test_carries_the_notice_with_the_results(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', SMALL)]

        results = search_projects(projects, lambda p: [(0.5, p.slug)])

        assert 'b' in results.notice
        assert results.selection.embedding_model == LARGE

    def test_limit_keeps_the_best(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', LARGE)]
        scores = {'a': [(0.9, 'a1'), (0.1, 'a2')], 'b': [(0.5, 'b1')]}

        results = search_projects(projects, lambda p: scores[p.slug], limit=2)

        assert [hit.payload for hit in results] == ['a1', 'b1']

    def test_everything_excluded_returns_nothing_but_says_why(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', SMALL)]

        results = search_projects(
            projects, lambda p: [(0.5, p.slug)],
            embedding_model='nomic-embed-text'
        )

        assert len(results) == 0
        assert results.notice.startswith('2 of 2 projects skipped')

    def test_a_project_with_no_hits_is_not_an_error(self, make_project):
        projects = [make_project('A', LARGE), make_project('B', LARGE)]

        results = search_projects(
            projects, lambda p: [] if p.slug == 'b' else [(0.5, 'only-a')]
        )

        assert [hit.payload for hit in results] == ['only-a']
        assert results.notice == ''
