# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app.py
# Description: The app's own logic, tested without Streamlit. The bug worth
#   preventing is a cached client answering from the wrong project.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.app import (
    HISTORY,
    PENDING,
    SELECTED,
    cache_key,
    list_projects,
    queue_question,
    remember,
    runtime_for,
    script_path,
    select_project,
    selected_project,
    take_pending
)
from osintgpt.app.views.chat import passages_of
from osintgpt.app.views.projects import describe


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def projects(home):
    first = Project.create('Case Alpha', home=home)
    second = Project.create('Case Beta', home=home)

    return first, second


class TestCachesAreKeyedOnTheProject:
    '''
    A cached client from another project is the worst bug available here: it
    answers, and the answer looks right.
    '''

    def test_two_projects_have_different_keys(self, projects):
        first, second = projects

        assert cache_key(first) != cache_key(second)

    def test_the_key_is_the_id_not_the_slug(self, projects):
        '''
        A slug can be reused after a delete; an id cannot.
        '''
        first, _ = projects

        assert cache_key(first) == first.id
        assert cache_key(first) != first.slug

    def test_the_key_survives_a_rename(self, projects, home):
        first, _ = projects
        before = cache_key(first)

        renamed = Project.load(first.paths.root)

        assert cache_key(renamed) == before


class TestSelection:
    '''
    Selection is the app's concept. Every library call takes a project
    explicitly, so nothing outside the session module can act on "current".
    '''

    def test_nothing_is_selected_to_begin_with(self, home):
        assert selected_project({}, home) is None

    def test_a_selected_project_comes_back(self, projects, home):
        first, _ = projects
        state = {}

        select_project(state, first.slug)

        assert selected_project(state, home).slug == first.slug

    def test_switching_clears_the_previous_conversation(self, projects):
        '''
        History is per project. Carrying it across a switch shows an analyst
        answers from a corpus they are no longer looking at.
        '''
        first, second = projects
        state = {}
        select_project(state, first.slug)
        remember(state, 'q', 'a')

        select_project(state, second.slug)

        assert state.get(HISTORY) is None

    def test_reselecting_the_same_project_keeps_the_conversation(
        self, projects
    ):
        first, _ = projects
        state = {}
        select_project(state, first.slug)
        remember(state, 'q', 'a')

        select_project(state, first.slug)

        assert len(state[HISTORY]) == 1

    def test_a_deleted_project_is_forgotten_rather_than_raising(
        self, projects, home
    ):
        import shutil

        first, _ = projects
        state = {}
        select_project(state, first.slug)
        shutil.rmtree(first.paths.root)

        assert selected_project(state, home) is None
        assert SELECTED not in state

    def test_clearing_the_selection_clears_the_conversation(self, projects):
        first, _ = projects
        state = {}
        select_project(state, first.slug)
        remember(state, 'q', 'a')

        select_project(state, None)

        assert SELECTED not in state
        assert HISTORY not in state


class TestFollowupButtons:
    '''
    A button queues a question and the next rerun consumes it. Clearing on
    read is what stops it being asked again on every subsequent rerun.
    '''

    def test_a_queued_question_is_taken_once(self):
        state = {}
        queue_question(state, 'What else about Alpha?')

        assert take_pending(state) == 'What else about Alpha?'
        assert take_pending(state) is None

    def test_nothing_queued_returns_nothing(self):
        assert take_pending({}) is None

    def test_a_blank_question_is_not_queued(self):
        state = {}
        queue_question(state, '   ')

        assert PENDING not in state

    def test_a_question_in_another_script_survives(self):
        state = {}
        queue_question(state, 'Что ещё об Альфе?')

        assert take_pending(state) == 'Что ещё об Альфе?'


class TestListing:
    def test_it_finds_projects_created_outside_the_app(self, projects, home):
        '''
        Rebuilt from disk, so a project made with the CLI appears without the
        operator doing anything.
        '''
        assert len(list_projects(home)) == 2

    def test_an_empty_home_lists_nothing(self, home):
        assert list_projects(home) == []


class TestDescribe:
    def test_it_reports_what_a_project_holds(self, projects):
        first, _ = projects

        facts = describe(first)

        assert facts['documents'] == 0
        assert facts['chunks'] == 0
        assert facts['backend'] == 'sqlite'

    def test_it_lists_the_legs_that_are_on(self, projects):
        first, _ = projects

        assert 'semantic' in describe(first)['legs']
        assert 'graph' not in describe(first)['legs']

    def test_an_unopenable_store_is_reported_not_raised(self, projects):
        '''
        A misconfigured backend should still let the operator see the project
        and fix it.
        '''
        first, _ = projects
        broken = first.with_settings(storage_backend='not-a-backend')
        broken.save()

        facts = describe(Project.load(first.paths.root))

        assert facts['problem']
        assert facts['documents'] is None


class TestSourceChips:
    def test_passages_come_from_the_trace(self):
        from osintgpt.agentic import Trace

        trace = Trace()
        trace.record(1, 'semantic_search', {}, count=1)
        object.__setattr__(
            trace.entries[0], 'payload',
            {'passages': [{'citation': 'a.md › Part', 'ref': 'a.md',
                           'text': 'The passage.'}]}
        )
        answer = type('A', (), {'trace': trace, 'sources': ['a.md']})()

        found = passages_of(answer)

        assert found[0]['citation'] == 'a.md › Part'
        assert found[0]['text'] == 'The passage.'

    def test_a_trace_with_no_passages_yields_none(self):
        from osintgpt.agentic import Trace

        answer = type('A', (), {'trace': Trace(), 'sources': []})()

        assert passages_of(answer) == []


class TestPackaging:
    def test_streamlit_is_an_extra_not_a_core_dependency(self):
        '''
        A library that pulls a web framework into every dependency tree is a
        library nobody embeds.
        '''
        from pathlib import Path

        from osintgpt.projects.toml_io import read_toml

        root = Path(__file__).resolve().parent.parent
        config = read_toml(root / 'pyproject.toml')

        core = ' '.join(config['project']['dependencies'])
        extras = config['project']['optional-dependencies']

        assert 'streamlit' not in core
        assert any('streamlit' in dep for dep in extras['app'])

    def test_the_entry_point_exists(self):
        from pathlib import Path

        from osintgpt.projects.toml_io import read_toml

        root = Path(__file__).resolve().parent.parent
        config = read_toml(root / 'pyproject.toml')

        assert 'osintgpt-app' in config['project']['scripts']

    def test_the_script_ships_with_the_package(self):
        '''
        Streamlit runs a file rather than importing one, so the path has to
        resolve from an installed wheel as well as a source tree.
        '''
        assert script_path().name == 'main.py'
        assert script_path().is_file()

    def test_launching_without_streamlit_says_what_to_install(
        self, monkeypatch, capsys
    ):
        import builtins

        from osintgpt.app import launch

        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name.startswith('streamlit'):
                raise ImportError('no streamlit')

            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        assert launch.main([]) == 1
        assert 'osintgpt[app]' in capsys.readouterr().err
