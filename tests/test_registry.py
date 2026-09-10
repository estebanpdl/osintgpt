# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_registry.py
# Description: The project index and the user defaults beside it, including the
#   rule that a project outranks whatever the index remembers about it.
# =================================================================================

# import modules
import shutil
import pytest

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt projects
from osintgpt.projects import (
    Project,
    ProjectSettings,
    Registry,
    RegistryEntry,
    load_user_defaults,
    save_user_defaults
)
from osintgpt.projects.home import config_file

from conftest import FAKE_KEY


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def registered(home):
    '''Two projects, both indexed.'''
    registry = Registry.load(home)
    projects = [
        Project.create('Case A', home=home),
        Project.create('Case B', home=home)
    ]
    for project in projects:
        registry.register(project)

    return registry, projects


class TestEmptyRegistry:
    def test_loads_without_a_file(self, home):
        registry = Registry.load(home)

        assert len(registry) == 0
        assert registry.find('anything') is None

    def test_reading_creates_nothing_on_disk(self, home):
        Registry.load(home)

        assert not home.exists()


class TestRegister:
    def test_indexes_a_project(self, home):
        registry = Registry.load(home)
        project = Project.create('Case A', home=home)
        registry.register(project)

        entry = registry.find('case-a')

        assert entry is not None
        assert entry.id == project.id
        assert entry.path == str(project.paths.root)

    def test_persists_across_loads(self, registered, home):
        _, projects = registered
        reloaded = Registry.load(home)

        assert len(reloaded) == 2
        assert {e.slug for e in reloaded} == {'case-a', 'case-b'}
        assert reloaded.find('case-a').id == projects[0].id

    def test_finds_by_id_as_well_as_slug(self, registered):
        registry, projects = registered

        assert registry.find(projects[0].id).slug == 'case-a'

    def test_re_registering_refreshes_rather_than_duplicates(self, registered):
        registry, projects = registered
        updated = projects[0].with_settings(
            embedding_model='text-embedding-3-large'
        )
        updated.save()
        registry.register(updated)

        assert len(registry) == 2
        assert registry.find('case-a').embedding_model == (
            'text-embedding-3-large'
        )

    def test_unregister_leaves_the_project_on_disk(self, registered):
        registry, projects = registered

        assert registry.unregister('case-a') is True
        assert registry.find('case-a') is None
        assert projects[0].paths.config.is_file()

    def test_unregister_reports_a_miss(self, registered):
        registry, _ = registered

        assert registry.unregister('not-a-project') is False


class TestOpen:
    def test_returns_the_project_itself(self, registered):
        registry, projects = registered

        assert registry.open('case-b').id == projects[1].id

    def test_unknown_key_raises(self, registered):
        registry, _ = registered

        with pytest.raises(KeyError):
            registry.open('case-z')

    def test_the_project_outranks_a_stale_entry(self, registered):
        '''
        The index is a cache. When it disagrees with project.toml, opening the
        project has to return what the project says.
        '''
        registry, projects = registered
        edited = projects[0].with_settings(embedding_model='changed-by-hand')
        edited.save()

        # the index still remembers the old value
        assert registry.find('case-a').embedding_model == ''
        assert registry.open('case-a').settings.embedding_model == (
            'changed-by-hand'
        )


class TestRebuild:
    def test_recovers_a_deleted_index(self, registered, home):
        registry, _ = registered
        Registry.file_for(home).unlink()

        rebuilt = Registry.rebuild(home)

        assert len(rebuilt) == 2
        assert {e.slug for e in rebuilt} == {'case-a', 'case-b'}

    def test_drops_projects_that_are_gone(self, registered, home):
        registry, projects = registered
        shutil.rmtree(projects[0].paths.root)

        rebuilt = Registry.rebuild(home)

        assert {e.slug for e in rebuilt} == {'case-b'}

    def test_picks_up_a_project_nobody_registered(self, home):
        Project.create('Case C', home=home)

        rebuilt = Registry.rebuild(home)

        assert {e.slug for e in rebuilt} == {'case-c'}

    def test_ignores_directories_that_are_not_projects(self, home):
        Project.create('Case A', home=home)
        (home / 'projects' / 'not-a-project').mkdir()

        rebuilt = Registry.rebuild(home)

        assert {e.slug for e in rebuilt} == {'case-a'}

    def test_does_not_find_projects_outside_the_home(self, home, tmp_path):
        Project.create('Elsewhere', home=home, path=tmp_path / 'elsewhere')

        rebuilt = Registry.rebuild(home)

        assert len(rebuilt) == 0

    def test_an_out_of_tree_project_can_still_be_registered(self, home, tmp_path):
        project = Project.create(
            'Elsewhere', home=home, path=tmp_path / 'elsewhere'
        )
        registry = Registry.rebuild(home)
        registry.register(project)

        assert Registry.load(home).find('elsewhere').path == str(
            project.paths.root
        )

    def test_writes_itself_to_disk(self, registered, home):
        Registry.file_for(home).unlink()
        Registry.rebuild(home)

        assert Registry.file_for(home).is_file()


class TestRegistryFile:
    def test_says_it_is_not_the_source_of_truth(self, registered, home):
        text = Registry.file_for(home).read_text(encoding='utf-8')

        assert text.lstrip().startswith('#')
        assert 'source of truth' in text


class TestUserDefaults:
    def test_absent_config_yields_unconfigured_settings(self, home):
        defaults = load_user_defaults(home)

        assert defaults == ProjectSettings()
        assert not home.exists()

    def test_round_trips(self, home):
        save_user_defaults(
            home, ProjectSettings(generation_model='gpt-4o', graph_enabled=True)
        )
        defaults = load_user_defaults(home)

        assert defaults.generation_model == 'gpt-4o'
        assert defaults.graph_enabled is True

    def test_warns_against_storing_secrets(self, home):
        save_user_defaults(home, ProjectSettings())
        text = config_file(home).read_text(encoding='utf-8')

        assert 'API keys' in text


class TestResolutionOrder:
    '''explicit argument -> project -> user defaults -> library default'''

    def test_user_defaults_fill_what_nobody_chose(self, home):
        save_user_defaults(home, ProjectSettings(generation_model='gpt-4o-mini'))
        project = Project.create('Case A', home=home)

        resolved = project.settings_for(
            Settings(openai_api_key=FAKE_KEY), load_user_defaults(home)
        )

        assert resolved.openai_gpt_model == 'gpt-4o-mini'

    def test_project_beats_user_defaults(self, home):
        save_user_defaults(home, ProjectSettings(generation_model='gpt-4o-mini'))
        project = Project.create(
            'Case A', home=home,
            settings=ProjectSettings(generation_model='gpt-4o')
        )

        resolved = project.settings_for(
            Settings(openai_api_key=FAKE_KEY), load_user_defaults(home)
        )

        assert resolved.openai_gpt_model == 'gpt-4o'

    def test_explicit_argument_beats_everything(self, home):
        save_user_defaults(home, ProjectSettings(generation_model='gpt-4o-mini'))
        project = Project.create(
            'Case A', home=home,
            settings=ProjectSettings(generation_model='gpt-4o')
        )

        resolved = project.settings_for(
            Settings(openai_api_key=FAKE_KEY, openai_gpt_model='o3'),
            load_user_defaults(home)
        )

        assert resolved.openai_gpt_model == 'o3'

    def test_defaults_are_optional(self, home):
        project = Project.create('Case A', home=home)
        resolved = project.settings_for(Settings(openai_api_key=FAKE_KEY))

        assert resolved.openai_gpt_model == ''
