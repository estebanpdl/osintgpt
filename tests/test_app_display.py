'''What the app shows an operator, and what it does with what they pick.'''

import pytest

from osintgpt.app.browse import directory_input
from osintgpt.app.styles import escape

BACKSLASH = chr(92)
WINDOWS_PATH = (
    'C:' + BACKSLASH + 'Users' + BACKSLASH + 'analyst' + BACKSLASH +
    '.osintgpt' + BACKSLASH + 'projects' + BACKSLASH + 'testing'
)


class TestEscape:
    def test_a_windows_path_keeps_every_separator(self):
        '''
        Markdown treats a backslash before punctuation as an escape, so
        `\\.osintgpt` renders as `.osintgpt` and the path shown to an operator
        is one they cannot paste anywhere.
        '''
        rendered = escape(WINDOWS_PATH)

        assert rendered.count(BACKSLASH * 2) == 5
        assert BACKSLASH + '.' not in rendered.replace(BACKSLASH * 2, '')

    def test_underscores_in_a_filename_do_not_become_italics(self):
        assert escape('case_notes_2026.md') == (
            'case' + BACKSLASH + '_notes' + BACKSLASH + '_2026.md'
        )

    def test_markup_in_data_is_shown_rather_than_rendered(self):
        '''
        Citations and model text reach a span written with
        unsafe_allow_html. Anything that arrives as data is displayed, never
        interpreted.
        '''
        rendered = escape('<script>alert(1)</script>')

        assert '<script>' not in rendered
        assert '&lt;script&gt;' in rendered

    def test_an_entity_is_not_broken_by_the_markdown_pass(self):
        # Escaping `&` after HTML-escaping would print `&amp;` literally.
        assert escape('a & b') == 'a &amp; b'

    def test_brackets_do_not_become_a_link(self):
        assert escape('[1](evil)') == (
            BACKSLASH + '[1' + BACKSLASH + '](evil)'
        )


class Column:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class St:
    '''
    Enough Streamlit to exercise `directory_input`: a text field that reads
    its value from session state the way a keyed widget really does, and a
    button that can be told to have been clicked.
    '''

    def __init__(self, state, clicked=False):
        self.state = state
        self.clicked = clicked
        self.reran = False

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)

        return [Column() for _ in range(count)]

    def text_input(self, _label, key=None, **_kwargs):
        return self.state.get(key, '')

    def button(self, *_args, **_kwargs):
        return self.clicked

    def markdown(self, *_args, **_kwargs):
        return None

    def rerun(self):
        self.reran = True


def run(state, clicked=False, chosen=None, monkeypatch=None):
    st = St(state, clicked=clicked)
    if monkeypatch is not None:
        monkeypatch.setattr(
            'osintgpt.app.browse.can_browse', lambda: True
        )
        monkeypatch.setattr(
            'osintgpt.app.browse.select_directory', lambda initial=None: chosen
        )

    value = directory_input(st, 'Location', 'loc', state)

    return st, value


class TestDirectoryInputKeepsWhatWasPicked:
    def test_a_chosen_directory_survives_the_rerun(self, monkeypatch):
        '''
        The bug this exists for: a keyed widget reads from session state and
        ignores anything written to a different key, so the chosen path was
        discarded and the field stayed empty.
        '''
        state = {}
        st, _ = run(state, clicked=True, chosen='D:/cases', monkeypatch=monkeypatch)

        assert st.reran is True

        # The rerun Streamlit would do next.
        _, value = run(state, monkeypatch=monkeypatch)

        assert value == 'D:/cases'
        assert state['loc'] == 'D:/cases'

    def test_cancelling_leaves_what_was_typed(self, monkeypatch):
        state = {'loc-text': 'D:/typed'}
        st, value = run(state, clicked=True, chosen=None, monkeypatch=monkeypatch)

        assert st.reran is False
        assert value == 'D:/typed'

    def test_typing_works_with_no_dialog_at_all(self, monkeypatch):
        monkeypatch.setattr('osintgpt.app.browse.can_browse', lambda: False)
        state = {'loc-text': 'D:/typed'}

        _, value = run(state)

        assert value == 'D:/typed'

    def test_whitespace_around_a_pasted_path_is_dropped(self):
        state = {'loc-text': '  D:/cases  '}

        _, value = run(state)

        assert value == 'D:/cases'

    def test_an_empty_field_is_an_empty_string(self):
        _, value = run({})

        assert value == ''


class TestTheCreateFormClearsBothKeys:
    def test_clearing_the_value_alone_would_leave_the_field_filled(self):
        '''
        The value and the widget behind it are two keys. Popping only the
        first leaves the previous project's path sitting in the box.
        '''
        import inspect

        from osintgpt.app.views import projects

        source = inspect.getsource(projects.render)

        assert "state.pop('new-project-location', None)" in source
        assert "state.pop('new-project-location-text', None)" in source
