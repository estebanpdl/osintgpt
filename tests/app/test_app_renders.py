# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app_renders.py
# Description: The Streamlit script actually running. The mistakes this catches
#   are the ones only a real render can — a container used the wrong way, an
#   API that moved, a widget key that collides.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.app.session import ASK, CONVERSATION, SELECTED, VIEW
from osintgpt.app.views.conversations import OPEN_KEY, THREAD_KEY
from osintgpt.projects import append_turn, start_conversation

# The whole point of this module is to run the app, so skip rather than fail
# where the test harness is unavailable.
AppTest = pytest.importorskip('streamlit.testing.v1').AppTest

SCRIPT = str(
    __import__('osintgpt.app.launch', fromlist=['script_path']).script_path()
)

ANSWER = {
    'question': 'What is alpha?',
    'answer': 'Aardvarks.',
    'sources': ['material/alpha.md'],
    'followups': [],
    'degraded': '',
    'trace': {'calls': [{
        'round': 1, 'tool': 'semantic_search', 'arguments': {'query': 'a'},
        'results': 1, 'unit': 'passage', 'documents': ['material/alpha.md'],
        'seconds': 0.1, 'error': ''
    }], 'narration': []}
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    '''
    The app reads the conventional home. Point it somewhere disposable, or
    this runs against whatever the machine happens to have.
    '''
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


def conversation_with(project, question: str, turns: int = 1):
    conversation = start_conversation(project, question)
    for _ in range(turns):
        append_turn(conversation, question, ANSWER)

    return conversation


def run(project=None, **state):
    app = AppTest.from_file(SCRIPT, default_timeout=60)
    if project is not None:
        app.session_state[SELECTED] = project.slug
    for key, value in state.items():
        app.session_state[key] = value
    app.run()

    return app


def buttons(app):
    return [b.label for b in app.sidebar.button]


class TestTheScriptRuns:
    def test_it_renders_without_raising(self, home):
        app = run()

        assert not app.exception

    def test_the_views_are_offered(self, home):
        app = run()

        assert app.sidebar.radio[0].options == [
            'Projects', 'Material', 'Ask', 'Settings'
        ]


class TestTheConversationSidebar:
    def test_a_project_with_none_offers_a_new_one(self, project):
        app = run(project)

        assert not app.exception
        assert 'New conversation' in buttons(app)

    def test_each_conversation_is_a_row(self, project):
        conversation_with(project, 'What is alpha?')
        conversation_with(project, 'What is beta?')

        app = run(project)

        assert not app.exception
        labels = buttons(app)
        assert 'What is alpha?' in labels
        assert 'What is beta?' in labels

    def test_the_open_one_is_marked_by_its_key(self, project):
        '''
        The stylesheet finds the open thread through this key prefix, so a
        change to it is a silent change to how the sidebar looks.
        '''
        first = conversation_with(project, 'What is alpha?')
        conversation_with(project, 'What is beta?')

        app = run(project, **{CONVERSATION: first.id, VIEW: ASK})
        keys = [b.key for b in app.sidebar.button if b.key]

        assert f'{OPEN_KEY}{first.id}' in keys
        assert sum(
            1 for key in keys if key.startswith(OPEN_KEY)
        ) == 1
        assert any(
            key.startswith(THREAD_KEY) and not key.startswith(OPEN_KEY)
            for key in keys
        )

    def test_opening_one_shows_its_turns(self, project):
        conversation = conversation_with(project, 'What is alpha?', turns=2)

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})

        assert not app.exception
        assert len(app.chat_message) == 4

    def test_a_replayed_answer_carries_its_trace(self, project):
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})
        rendered = ' '.join(m.value for m in app.markdown)

        assert 'trace-tool' in rendered
        assert 'semantic_search' in rendered

    def test_a_replayed_turn_offers_no_stale_followups(self, project):
        '''
        Suggestions belong to the newest turn. Replaying an old one must not
        re-offer questions the analyst has already moved past.
        '''
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})
        keys = [b.key for b in app.button if b.key]

        assert not [key for key in keys if key.startswith('followup-')]

    def test_a_conversation_from_another_project_is_not_listed(self, home):
        first = Project.create('Case Alpha', home=home)
        second = Project.create('Case Beta', home=home)
        conversation_with(first, 'What is alpha?')

        app = run(second)

        assert 'What is alpha?' not in buttons(app)

    def test_clicking_one_opens_it_in_the_ask_view(self, project):
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project)
        app.sidebar.button(key=f'{THREAD_KEY}{conversation.id}').click().run()

        assert not app.exception
        assert app.session_state[CONVERSATION] == conversation.id
        assert app.session_state[VIEW] == ASK

    def test_new_conversation_clears_the_open_one(self, project):
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})
        app.sidebar.button(key='conversation-new').click().run()

        assert not app.exception
        assert CONVERSATION not in app.session_state
        # The thread is still on disk; only the selection went.
        assert conversation.path.is_file()

    def test_renaming_shows_the_new_title(self, project):
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})
        app.sidebar.text_input(
            key=f'conversation-title-{conversation.id}'
        ).set_value('Aardvark sightings').run()
        app.sidebar.button(key='conversation-rename').click().run()

        assert not app.exception
        assert 'Aardvark sightings' in buttons(app)

    def test_delete_needs_confirming_first(self, project):
        '''
        A transcript is the record of what was asked and answered, and losing
        one is not recoverable.
        '''
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})

        assert app.sidebar.button(key='conversation-delete').disabled
        assert conversation.path.is_file()

    def test_confirming_then_deleting_removes_it(self, project):
        conversation = conversation_with(project, 'What is alpha?')

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})
        app.sidebar.checkbox(key='conversation-confirm').check().run()
        app.sidebar.button(key='conversation-delete').click().run()

        assert not app.exception
        assert not conversation.path.exists()
        assert CONVERSATION not in app.session_state

    def test_a_deleted_conversation_does_not_break_the_render(self, project):
        conversation = conversation_with(project, 'What is alpha?')
        conversation.path.unlink()

        app = run(project, **{CONVERSATION: conversation.id, VIEW: ASK})

        assert not app.exception
