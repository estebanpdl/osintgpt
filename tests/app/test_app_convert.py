# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app_convert.py
# Description: Converting one document from the Material view, and the promise
#   it rests on — nothing registered, nothing embedded, no model contacted.
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
def note(tmp_path):
    path = tmp_path / 'note.md'
    path.write_text(
        '---\nauthor: Ana\n---\n\n# Heading\n\nUn párrafo del expediente.\n',
        encoding='utf-8'
    )

    return path


def run(project, path):
    app = AppTest.from_file(SCRIPT, default_timeout=60)
    app.session_state[SELECTED] = project.slug
    app.session_state[VIEW] = MATERIAL
    app.session_state['convert-path'] = str(path)
    app.run()

    return app


def convert(app):
    button = [b for b in app.button if b.label == 'Convert'][0]

    return button.click().run()


class TestPanel:
    def test_the_panel_is_offered_before_anything_is_registered(
        self, project, note
    ):
        app = run(project, note)

        assert not app.exception
        assert any(
            panel.label == 'Convert a document' for panel in app.expander
        )

    def test_nothing_is_read_until_the_button_is_pressed(self, project, note):
        app = run(project, note)

        assert not [element for element in app.code]


class TestConverting:
    def test_the_markdown_is_shown_with_the_reader_that_made_it(
        self, project, note
    ):
        app = convert(run(project, note))

        assert not app.exception
        assert 'Un párrafo del expediente.' in app.code[0].value
        assert any('Read by text' in caption.value for caption in app.caption)

    def test_a_metadata_block_is_separated_the_way_indexing_separates_it(
        self, project, note
    ):
        app = convert(run(project, note))

        assert 'author: Ana' not in app.code[0].value

    def test_converting_registers_nothing(self, project, note):
        convert(run(project, note))

        assert Corpus.load(project.paths.sources).sources == []

    def test_a_path_that_is_not_a_file_is_reported_in_place(
        self, project, tmp_path
    ):
        app = convert(run(project, tmp_path / 'absent.md'))

        assert not app.exception
        assert any('path to a file' in error.value for error in app.error)

    def test_an_image_says_why_it_has_no_markdown(self, project, tmp_path):
        path = tmp_path / 'scan.png'
        path.write_bytes(b'\x89PNG\r\n')

        app = convert(run(project, path))

        assert not app.exception
        assert any('images are embedded' in error.value for error in app.error)


class TestStructured:
    def test_a_structured_file_asks_for_its_roles_first(
        self, project, tmp_path
    ):
        path = tmp_path / 'registros.csv'
        path.write_text(
            'record_id,body\nr1,primer registro\n', encoding='utf-8'
        )

        app = run(project, path)

        assert any(
            panel.label.startswith('Fields in') for panel in app.expander
        )
