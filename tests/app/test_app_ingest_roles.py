# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app_ingest_roles.py
# Description: Assigning field roles in the Material view. A mapping collected
#   here governs what a corpus can ever be asked, so these run the real script
#   rather than the functions behind it.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.app.session import MATERIAL, SELECTED, VIEW
from osintgpt.ingestion import Corpus

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest

SCRIPT = str(
    __import__('osintgpt.app.launch', fromlist=['script_path']).script_path()
)

ROLES = (
    'primary', 'extra', 'timestamp', 'author', 'identity', 'metadata',
    'min-chars'
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    place = tmp_path / 'home'
    place.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        'osintgpt.projects.default_home', lambda: place, raising=False
    )
    monkeypatch.setattr(
        'osintgpt.projects.paths.default_home', lambda: place, raising=False
    )

    return place


@pytest.fixture
def project(home):
    return Project.create('Case Alpha', home=home)


@pytest.fixture
def folder(tmp_path):
    '''Two files of one schema, one of another, and a file needing no mapping.'''
    place = tmp_path / 'material'
    place.mkdir()
    for name in ('one.csv', 'two.csv'):
        (place / name).write_text(
            'id,body,handle,at\n'
            'a1,First record content.,alpha,2026-03-01\n'
            'a2,,beta,2026-03-02\n',
            encoding='utf-8'
        )
    (place / 'other.csv').write_text(
        'sku,note\nx1,A different schema entirely.\n', encoding='utf-8'
    )
    (place / 'prose.md').write_text('# Heading\n\nSome prose.\n', encoding='utf-8')

    return place


def run(project, folder):
    app = AppTest.from_file(SCRIPT, default_timeout=60)
    app.session_state[SELECTED] = project.slug
    app.session_state[VIEW] = MATERIAL
    app.session_state['ingest-folder-text'] = str(folder)
    app.run()

    return app


def register(app):
    button = [b for b in app.button if b.label == 'Register this folder'][0]

    return button.click().run()


class TestSchemaGrouping:
    def test_files_sharing_a_schema_are_asked_about_once(self, project, folder):
        app = run(project, folder)

        assert not app.exception
        schemas = [
            panel for panel in app.expander
            if panel.label.startswith('Fields in')
        ]
        assert len(schemas) == 2

    def test_the_count_is_of_files_not_schemas(self, project, folder):
        app = run(project, folder)

        assert any('3 structured file(s)' in w.value for w in app.warning)


class TestRolesAreOffered:
    def test_every_role_has_a_widget(self, project, folder):
        app = run(project, folder)

        keys = {widget.key for widget in app.selectbox}
        keys |= {widget.key for widget in app.multiselect}
        keys |= {widget.key for widget in app.number_input}

        for role in ROLES:
            assert f'map-0-{role}' in keys, role

    def test_each_schema_gets_its_own_keys(self, project, folder):
        app = run(project, folder)

        keys = {widget.key for widget in app.selectbox}

        assert 'map-1-primary' in keys


class TestTheOutcomeIsShown:
    def test_nothing_is_read_until_content_is_named(self, project, folder):
        app = run(project, folder)

        assert not any(
            ' records' in caption.value for caption in app.caption
        )

    def test_the_counts_appear_once_it_is(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()

        assert any(
            'with no content' in caption.value for caption in app.caption
        )

    def test_every_file_of_the_shape_is_counted_not_just_the_first(
        self, project, folder
    ):
        '''one.csv and two.csv each hold 2 records, 1 of them with content.'''
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()

        assert any(
            '2 of 4 records' in caption.value for caption in app.caption
        )

    def test_the_files_are_read_once_per_question_asked_of_them(
        self, project, folder, monkeypatch
    ):
        '''
        Every widget interaction reruns the script, and a role that cannot
        move the counts must not pay to recompute them.
        '''
        from osintgpt.app.views import mapping

        mapping._COUNTS.clear()
        passes = []
        real = mapping.preview_files
        monkeypatch.setattr(
            mapping, 'preview_files',
            lambda paths, roles: (
                passes.append(tuple(paths)), real(paths, roles)
            )[1]
        )

        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        after_content = len(passes)
        app.selectbox(key='map-0-author').set_value('handle').run()
        app.selectbox(key='map-0-identity').set_value('id').run()

        assert after_content == 1
        assert len(passes) == after_content

    def test_the_floor_is_reflected_as_it_changes(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app.number_input(key='map-0-min-chars').set_value(200).run()

        assert any(
            'under the floor' in caption.value for caption in app.caption
        )

    def test_a_mapping_that_keeps_nothing_is_called_out(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app.number_input(key='map-0-min-chars').set_value(200).run()

        assert any(
            'Every record would be skipped' in error.value
            for error in app.error
        )


class TestRegistering:
    def test_a_mapping_is_recorded_for_each_file_of_the_schema(
        self, project, folder
    ):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app.selectbox(key='map-0-timestamp').set_value('at').run()
        app.selectbox(key='map-0-author').set_value('handle').run()
        app.selectbox(key='map-0-identity').set_value('id').run()
        app.number_input(key='map-0-min-chars').set_value(20).run()
        app = register(app)

        assert not app.exception
        mapped = [
            source.mapping
            for source in Corpus.load(project.paths.sources).sources
            if source.mapping.is_set
        ]

        assert len(mapped) == 2
        for mapping in mapped:
            assert mapping.content == ('body',)
            assert mapping.timestamp == 'at'
            assert mapping.author == 'handle'
            assert mapping.identity == 'id'
            assert mapping.min_chars == 20

    def test_the_folder_stays_registered_for_its_prose(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app = register(app)

        sources = Corpus.load(project.paths.sources).sources

        assert any(not source.mapping.is_set for source in sources)

    def test_the_mapping_governs_the_files_it_was_given(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app.number_input(key='map-0-min-chars').set_value(20).run()
        app = register(app)

        governing = Corpus.load(project.paths.sources).mapping_for(
            folder / 'two.csv', project.paths.root
        )

        assert governing.content == ('body',)
        assert governing.min_chars == 20

    def test_extra_content_fields_follow_the_primary(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app.multiselect(key='map-0-extra').set_value(['handle']).run()
        app = register(app)

        governing = Corpus.load(project.paths.sources).mapping_for(
            folder / 'one.csv', project.paths.root
        )

        assert governing.content == ('body', 'handle')

    def test_a_schema_left_without_content_is_reported(self, project, folder):
        app = run(project, folder)
        app.selectbox(key='map-0-primary').set_value('body').run()
        app = register(app)

        assert any(
            'left without a content field' in warning.value
            for warning in app.warning
        )
