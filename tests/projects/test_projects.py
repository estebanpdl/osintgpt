# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_projects.py
# Description: Project creation, round-tripping, and the isolation between two
#   projects that the whole design rests on.
# =================================================================================

# import modules
import shutil
import pytest

# import submodules
from pathlib import Path

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt projects
from osintgpt.projects import Project, ProjectPaths, ProjectSettings, slugify
from osintgpt.projects.conversations import (
    append_turn, list_conversations, open_conversation, start_conversation,
)
from osintgpt.projects.paths import default_home

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from tests.conftest import FAKE_KEY


class TestSlugify:
    @pytest.mark.parametrize('name, expected', [
        ('Operation Blackcore', 'operation-blackcore'),
        ('  Case   #42  ', 'case-42'),
        ('Elecciones/2026', 'elecciones-2026'),
        ('ALL CAPS', 'all-caps')
    ])
    def test_makes_a_directory_safe_name(self, name, expected):
        assert slugify(name) == expected

    def test_falls_back_when_nothing_survives(self):
        assert slugify('!!!') == 'project'


class TestPaths:
    def test_default_home_is_not_created_as_a_side_effect(self):
        home = default_home()

        assert home.name == '.osintgpt'
        assert home.parent == Path.home()

    def test_layout_derives_from_the_root(self, tmp_path):
        paths = ProjectPaths(tmp_path / 'case')

        assert paths.config.name == 'project.toml'
        assert paths.store.name == 'store.sqlite'
        assert paths.sources.name == 'sources.toml'
        assert paths.extracts.parent == paths.root

    def test_under_home_uses_the_conventional_location(self, tmp_path):
        paths = ProjectPaths.under_home(tmp_path, 'case-a')

        assert paths.root == tmp_path / 'projects' / 'case-a'


class TestCreate:
    def test_writes_the_documented_layout(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)

        assert project.paths.config.is_file()
        assert project.paths.extracts.is_dir()
        assert project.paths.canon.is_dir()
        assert project.paths.root == tmp_path / 'projects' / 'case-a'

    def test_identity_carries_a_uuid_and_a_slug(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)

        assert len(project.id) == 32
        int(project.id, 16)
        assert project.slug == 'case-a'
        assert project.name == 'Case A'
        assert project.created_at

    def test_two_projects_get_different_ids(self, tmp_path):
        a = Project.create('Case A', home=tmp_path)
        b = Project.create('Case B', home=tmp_path)

        assert a.id != b.id

    def test_an_explicit_path_wins_over_home(self, tmp_path):
        elsewhere = tmp_path / 'encrypted' / 'case'
        project = Project.create('Case A', home=tmp_path, path=elsewhere)

        assert project.paths.root == elsewhere
        assert project.paths.config.is_file()

    def test_refuses_to_overwrite_an_existing_project(self, tmp_path):
        Project.create('Case A', home=tmp_path)

        with pytest.raises(FileExistsError):
            Project.create('Case A', home=tmp_path)

    def test_graph_is_off_by_default(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)

        assert project.settings.graph_enabled is False
        assert project.settings.semantic_enabled is True
        assert project.settings.lexical_enabled is True


class TestConfigFile:
    def test_warns_against_storing_secrets(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        text = project.paths.config.read_text(encoding='utf-8')

        assert text.lstrip().startswith('#')
        assert 'API keys' in text

    def test_is_valid_toml_with_both_sections(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)

        with open(project.paths.config, 'rb') as handle:
            data = tomllib.load(handle)

        assert data['project']['slug'] == 'case-a'
        assert data['settings']['graph_enabled'] is False

    def test_survives_a_windows_style_path_in_a_value(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        project = project.with_settings(generation_model=r'model\with\slashes')
        project.save()

        loaded = Project.load(project.paths.root)

        assert loaded.settings.generation_model == r'model\with\slashes'


class TestLoad:
    def test_round_trips_identity_and_settings(self, tmp_path):
        created = Project.create(
            'Case A', home=tmp_path,
            settings=ProjectSettings(
                generation_model='gpt-4o', graph_enabled=True
            )
        )
        loaded = Project.load(created.paths.root)

        assert loaded.id == created.id
        assert loaded.slug == created.slug
        assert loaded.name == created.name
        assert loaded.created_at == created.created_at
        assert loaded.settings == created.settings

    def test_missing_project_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Project.load(tmp_path / 'nothing-here')

    def test_unknown_settings_keys_are_ignored(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        text = project.paths.config.read_text(encoding='utf-8')
        project.paths.config.write_text(
            text + '\nfrom_a_newer_version = true\n', encoding='utf-8'
        )

        assert Project.load(project.paths.root).slug == 'case-a'

    def test_the_project_file_is_the_source_of_truth_for_its_slug(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        renamed = project.paths.root.parent / 'renamed-on-disk'
        project.paths.root.rename(renamed)

        loaded = Project.load(renamed)

        assert loaded.id == project.id
        assert loaded.slug == 'case-a'


class TestSettingsComposition:
    def test_preserves_the_sql_path_setting_for_compatibility(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        resolved = project.settings_for(Settings(openai_api_key=FAKE_KEY))

        assert resolved.sql_db_file_path == str(project.paths.store)

    def test_project_fills_what_the_caller_left_unset(self, tmp_path):
        project = Project.create(
            'Case A', home=tmp_path,
            settings=ProjectSettings(generation_model='gpt-4o-mini')
        )
        resolved = project.settings_for(Settings(openai_api_key=FAKE_KEY))

        assert resolved.openai_gpt_model == 'gpt-4o-mini'

    def test_an_explicit_argument_beats_the_project(self, tmp_path):
        project = Project.create(
            'Case A', home=tmp_path,
            settings=ProjectSettings(generation_model='gpt-4o-mini')
        )
        resolved = project.settings_for(
            Settings(openai_api_key=FAKE_KEY, openai_gpt_model='gpt-4o')
        )

        assert resolved.openai_gpt_model == 'gpt-4o'

    def test_project_embedding_model_applies_over_the_library_default(
        self, tmp_path
    ):
        project = Project.create(
            'Case A', home=tmp_path,
            settings=ProjectSettings(embedding_model='text-embedding-3-large')
        )
        resolved = project.settings_for(Settings(openai_api_key=FAKE_KEY))

        assert resolved.openai_embedding_model == 'text-embedding-3-large'

    def test_an_explicit_embedding_model_beats_the_project(self, tmp_path):
        project = Project.create(
            'Case A', home=tmp_path,
            settings=ProjectSettings(embedding_model='text-embedding-3-large')
        )
        resolved = project.settings_for(Settings(
            openai_api_key=FAKE_KEY,
            openai_embedding_model='text-embedding-3-small'
        ))

        assert resolved.openai_embedding_model == 'text-embedding-3-small'

    def test_never_reads_secrets_from_the_project(self, tmp_path):
        project = Project.create('Case A', home=tmp_path)
        resolved = project.settings_for(Settings(openai_api_key=FAKE_KEY))

        assert resolved.openai_api_key == FAKE_KEY
        assert 'openai_api_key' not in project.settings.to_dict()


class TestIsolation:
    def test_two_projects_do_not_share_a_store(self, tmp_path):
        a = Project.create('Case A', home=tmp_path)
        b = Project.create('Case B', home=tmp_path)

        assert a.paths.store != b.paths.store
        assert a.paths.root.parent == b.paths.root.parent

    def test_conversation_threads_stay_separate(self, tmp_path):
        '''Persist and reopen both projects without sharing their turns.'''
        asked = {}
        projects = []
        threads = []
        for name, question in (('Case A', 'about alpha'),
                               ('Case B', 'about beta')):
            project = Project.create(name, home=tmp_path)
            thread = start_conversation(project, question)
            assert append_turn(thread, question, {'answer': name}) is not None
            projects.append(Project.load(project.paths.root))
            threads.append(thread)

        for project, thread in zip(projects, threads):
            reopened = open_conversation(project, thread.id)
            assert reopened is not None
            assert reopened.path.parent == project.paths.root / 'conversations'
            assert [item.id for item in list_conversations(project)] == [thread.id]
            assert [turn.answer for turn in reopened.turns] == [{'answer': project.name}]
            asked[project.name] = {turn.question for turn in reopened.turns}

        assert asked['Case A'] == {'about alpha'}
        assert asked['Case B'] == {'about beta'}
        assert open_conversation(projects[0], threads[1].id) is None
        assert open_conversation(projects[1], threads[0].id) is None

    def test_deleting_a_project_is_deleting_a_directory(self, tmp_path):
        a = Project.create('Case A', home=tmp_path)
        b = Project.create('Case B', home=tmp_path)
        shutil.rmtree(a.paths.root)

        assert not a.paths.root.exists()
        assert Project.load(b.paths.root).slug == 'case-b'
