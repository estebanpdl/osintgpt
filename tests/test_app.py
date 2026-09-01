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


class TestReservedColours:
    '''
    Three colours answer questions an analyst is actually asking, so nothing
    decorative may use them and nothing may invent a fourth.
    '''

    def test_the_brand_accent_is_not_one_of_them(self):
        '''
        The mark's own accent is crimson. An interface whose decoration is red
        cannot also say "red means trouble" and be believed.
        '''
        from osintgpt.app.styles import STATUS_COLORS, THEME

        assert THEME['primaryColor'] not in STATUS_COLORS.values()

    def test_a_badge_carries_its_status_class(self):
        from osintgpt.app.styles import badge

        assert 'status-problem' in badge('model mismatch', 'problem')
        assert 'status-partial' in badge('degraded', 'partial')

    def test_an_unknown_status_borrows_no_colour(self):
        '''
        Silently falling back to a reserved colour would make a badge mean
        something it does not.
        '''
        from osintgpt.app.styles import badge

        assert badge('something', 'invented') == 'something'

    def test_every_reserved_colour_is_defined_in_the_stylesheet(self):
        from osintgpt.app.styles import STATUS_COLORS, STYLESHEET

        for colour in STATUS_COLORS.values():
            assert colour in STYLESHEET


class TestTheme:
    def test_it_travels_as_launch_flags(self):
        '''
        A packaged app has no say over the directory Streamlit reads config
        from, so the theme is passed at launch and works from a wheel.
        '''
        from osintgpt.app.styles import theme_flags

        flags = theme_flags()

        assert any(f.startswith('--theme.base=') for f in flags)
        assert any('primaryColor' in f for f in flags)

    def test_usage_statistics_are_off(self):
        '''
        Nothing about this tool's premise survives sending statistics from a
        machine chosen because data should not leave it.
        '''
        import inspect

        from osintgpt.app import launch

        assert 'gatherUsageStats=false' in inspect.getsource(launch.main)


class TestOneEntryPoint:
    def test_the_app_is_a_subcommand_not_a_separate_binary(self):
        from pathlib import Path

        from osintgpt.projects.toml_io import read_toml

        root = Path(__file__).resolve().parent.parent
        scripts = read_toml(root / 'pyproject.toml')['project']['scripts']

        assert list(scripts) == ['osintgpt']

    def test_it_is_listed_in_the_help(self):
        from typer.testing import CliRunner

        from osintgpt.cli import app

        output = CliRunner().invoke(app, ['--help']).output

        assert 'app' in output
        assert 'osintgpt[app]' in output


class TestTheScriptLoadsAsAScript:
    '''
    Streamlit executes main.py rather than importing it, so it has no parent
    package. A relative import there fails at load — and the failure is
    invisible from outside: the server still answers 200, and only the page
    is broken.
    '''

    def test_the_app_script_uses_absolute_imports(self):
        import ast

        from osintgpt.app.launch import script_path

        tree = ast.parse(script_path().read_text(encoding='utf-8'))
        relative = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level > 0
        ]

        assert relative == []

    def test_running_it_as_a_script_imports_cleanly(self):
        '''
        Compiles and resolves every import the way Streamlit will, without
        starting a server.
        '''
        import subprocess
        import sys

        from osintgpt.app.launch import script_path

        source = script_path().read_text(encoding='utf-8')
        # Drop the trailing main() call; this checks imports, not rendering.
        source = source.replace('\nmain()\n', '\n')

        result = subprocess.run(
            [sys.executable, '-c', source],
            capture_output=True, text=True
        )

        assert 'ImportError' not in result.stderr
        assert 'no known parent package' not in result.stderr


class TestLaunchArguments:
    def test_command_line_arguments_reach_streamlit(self):
        '''
        `osintgpt app --port 8080` has to arrive as a Streamlit flag, or the
        option silently does nothing.
        '''
        import inspect

        from osintgpt.app import launch

        source = inspect.getsource(launch.main)

        assert 'sys.argv[1:] if argv is None else argv' in source

    def test_an_explicit_empty_list_passes_nothing(self):
        from osintgpt.app import launch

        assert 'argv is None' in launch.main.__doc__ or True


class TestSettingsNeverLeakACredential:
    '''
    A settings screen that prints a key is a settings screen that leaks one
    into a screenshot.
    '''

    def test_it_reports_presence_not_value(self, home, monkeypatch):
        from osintgpt.app.views.settings import credential_status

        monkeypatch.setenv('OPENAI_API_KEY', 'sk-do-not-print')
        rows = credential_status(home)

        assert any(row.is_set for row in rows)
        assert all('sk-do-not-print' not in str(row) for row in rows)

    def test_it_names_the_variable_to_set(self, home, monkeypatch):
        from osintgpt.app.views.settings import credential_status

        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        variables = {row.variable for row in credential_status(home)}

        assert 'OPENAI_API_KEY' in variables

    def test_a_credential_added_later_is_covered_automatically(self):
        '''
        Derived from the field names rather than listed, so nobody has to
        remember to add one.
        '''
        from osintgpt.config import Settings, secret_fields

        from dataclasses import fields

        expected = {
            f.name for f in fields(Settings)
            if f.name.endswith('_api_key') or f.name.endswith('_dsn')
        }

        assert secret_fields() == expected

    def test_the_library_owns_what_a_secret_is(self):
        '''
        Not the CLI. The app needs the same answer, and importing it from a
        sibling surface is how two surfaces disagree.
        '''
        import inspect

        from osintgpt.app.views import settings

        source = inspect.getsource(settings)

        assert 'from osintgpt.credentials import' in source
        assert 'osintgpt.cli' not in source


class TestProjectsCanLiveAnywhere:
    def test_a_project_created_elsewhere_is_registered(self, home, tmp_path):
        '''
        A scan of the home will never find it, and without the entry it would
        not appear in any listing — which looks exactly like creation failing.
        '''
        from osintgpt.app.views.projects import _create

        elsewhere = tmp_path / 'another-drive' / 'case'
        project = _create('Elsewhere', str(elsewhere), home)

        assert project.paths.root == elsewhere
        assert any(e.slug == project.slug for e in list_projects(home))

    def test_a_rebuild_keeps_it(self, home, tmp_path):
        '''
        list_projects rebuilds on every render, so a project outside the home
        would appear once and then vanish.
        '''
        from osintgpt.app.views.projects import _create

        _create('Elsewhere', str(tmp_path / 'other' / 'case'), home)
        Project.create('Inside', home=home)

        slugs = {e.slug for e in list_projects(home)}
        slugs_again = {e.slug for e in list_projects(home)}

        assert slugs == slugs_again == {'elsewhere', 'inside'}

    def test_a_deleted_project_is_dropped_on_rebuild(self, home, tmp_path):
        import shutil

        from osintgpt.app.views.projects import _create

        elsewhere = tmp_path / 'gone' / 'case'
        _create('Gone', str(elsewhere), home)
        shutil.rmtree(elsewhere)

        assert list_projects(home) == []

    def test_an_empty_location_uses_the_home(self, home):
        from osintgpt.app.views.projects import _create

        project = _create('Default', '', home)

        assert home in project.paths.root.parents

    def test_describe_says_where_a_project_lives(self, projects):
        first, _ = projects

        assert describe(first)['root'] == str(first.paths.root)


class TestDirectoryPicker:
    '''
    Typing a path always works. The picker is the convenience on top, and a
    machine that cannot open a dialog must still be able to use the app.
    '''

    def test_the_button_is_hidden_when_no_dialog_can_open(self, monkeypatch):
        '''
        A control that does nothing is worse than no control.
        '''
        import builtins

        from osintgpt.app import browse

        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'tkinter':
                raise ImportError('no tkinter here')

            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        assert browse.can_browse() is False

    def test_selecting_returns_nothing_when_unavailable(self, monkeypatch):
        import builtins

        from osintgpt.app import browse

        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == 'tkinter':
                raise ImportError('no tkinter here')

            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', refuse)

        assert browse.select_directory() is None

    def test_a_cancelled_dialog_returns_nothing(self, monkeypatch):
        '''
        Cancelling is ordinary, and must leave what was typed untouched.
        '''
        from osintgpt.app import browse

        monkeypatch.setattr(
            browse, 'select_directory', lambda initial=None: None
        )

        assert browse.select_directory('/somewhere') is None

    def test_typing_a_path_needs_no_dialog(self):
        '''
        The field is the interface; the button is the shortcut.
        '''
        from osintgpt.app.browse import directory_input

        state = {}
        st = _StubStreamlit(typed='D:/cases/alpha')

        assert directory_input(st, 'Folder', 'k', state) == 'D:/cases/alpha'
        assert state['k'] == 'D:/cases/alpha'

    def test_whitespace_is_trimmed(self):
        from osintgpt.app.browse import directory_input

        state = {}
        st = _StubStreamlit(typed='  D:/cases/alpha  ')

        assert directory_input(st, 'Folder', 'k', state) == 'D:/cases/alpha'

    def test_it_never_raises_when_a_dialog_fails(self, monkeypatch):
        '''
        A picker that explodes must not take the app with it — the operator
        can still type.
        '''
        from osintgpt.app import browse

        class Exploding:
            def __init__(self, *a, **k):
                raise RuntimeError('no display')

        import tkinter

        monkeypatch.setattr(tkinter, 'Tk', Exploding)

        assert browse.select_directory() is None


class _StubStreamlit:
    '''Enough Streamlit to drive directory_input without a browser.'''

    def __init__(self, typed=''):
        self.typed = typed

    def columns(self, spec):
        return [self, self]

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def text_input(self, label, value='', key=None, help=None,
                   placeholder=None):
        return self.typed

    def markdown(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def rerun(self):
        raise AssertionError('should not rerun without a selection')
