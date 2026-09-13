# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app_conversation_sidebar.py
# Description: The sidebar's own logic. Listing reads a file per conversation,
#   so what invalidates the cache is the thing worth pinning down.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.app.session import ASK, CONVERSATION, PENDING, VIEW
from osintgpt.app.views.conversations import (
    MAX_LISTED,
    OPEN_KEY,
    THREAD_KEY,
    _filtered,
    _open,
    _start,
    fingerprint
)
from osintgpt.projects import append_turn, start_conversation
from osintgpt.projects.conversations import conversations_dir

ANSWER = {'answer': 'Aardvarks.'}


@pytest.fixture
def project(tmp_path):
    return Project.create('Case', home=tmp_path / 'home')


class FakeSidebar:
    '''Records what the section would render, and answers the filter.'''

    def __init__(self, typed: str = ''):
        self.typed = typed
        self.captions = []

    def text_input(self, label, **kwargs):
        return self.typed

    def caption(self, body):
        self.captions.append(body)


def threads(count: int):
    return [(f'id-{i}', f'Conversation {i}') for i in range(count)]


class TestTheListingCacheKey:
    '''
    The key has to change whenever the list would look different, or the
    sidebar shows a thread that is gone and hides one that is not.
    '''

    def test_an_absent_directory_signs_as_empty(self, project):
        assert fingerprint(conversations_dir(project)) == ()

    def test_a_new_conversation_changes_it(self, project):
        directory = conversations_dir(project)
        before = fingerprint(directory)

        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)

        assert fingerprint(directory) != before

    def test_a_new_turn_changes_it(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)
        directory = conversations_dir(project)
        before = fingerprint(directory)

        append_turn(conversation, 'second?', ANSWER)

        assert fingerprint(directory) != before

    def test_a_rename_changes_it(self, project):
        '''
        A rename appends, so it moves size and time like any other write.
        '''
        from osintgpt.projects import rename_conversation

        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)
        directory = conversations_dir(project)
        before = fingerprint(directory)

        rename_conversation(conversation, 'Renamed')

        assert fingerprint(directory) != before

    def test_a_delete_changes_it(self, project):
        from osintgpt.projects import delete_conversation

        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)
        directory = conversations_dir(project)
        before = fingerprint(directory)

        delete_conversation(conversation)

        assert fingerprint(directory) != before

    def test_reading_it_does_not_change_it(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)
        directory = conversations_dir(project)

        assert fingerprint(directory) == fingerprint(directory)


class TestTheFilter:
    def test_a_short_list_is_shown_whole(self):
        sidebar = FakeSidebar()
        listed = threads(MAX_LISTED)

        assert _filtered(sidebar, listed) == listed
        assert sidebar.captions == []

    def test_a_long_list_is_truncated_and_says_so(self):
        sidebar = FakeSidebar()
        listed = threads(MAX_LISTED + 5)

        shown = _filtered(sidebar, listed)

        assert len(shown) == MAX_LISTED
        assert 'filter to reach the rest' in sidebar.captions[0]

    def test_filtering_reaches_past_the_truncation(self):
        '''
        The cut is what makes the filter necessary, so the filter must not
        also be cut — a thread past the fold has to be findable.
        '''
        sidebar = FakeSidebar(typed='Conversation 2')
        listed = threads(MAX_LISTED + 10)

        shown = _filtered(sidebar, listed)

        assert ('id-24', 'Conversation 24') in shown

    def test_filtering_ignores_case(self):
        sidebar = FakeSidebar(typed='CONVERSATION 21')
        shown = _filtered(sidebar, threads(MAX_LISTED + 10))

        assert shown == [('id-21', 'Conversation 21')]

    def test_a_filter_matching_nothing_shows_nothing(self):
        sidebar = FakeSidebar(typed='zebra')

        assert _filtered(sidebar, threads(MAX_LISTED + 5)) == []


class TestOpeningAThread:
    def test_it_selects_and_moves_to_the_ask_view(self):
        state = {}

        _open(state, 'some-conversation')

        assert state[CONVERSATION] == 'some-conversation'
        assert state[VIEW] == ASK

    def test_starting_one_clears_the_selection(self):
        state = {CONVERSATION: 'old', PENDING: 'a queued question'}

        _start(state)

        assert CONVERSATION not in state
        assert state[VIEW] == ASK

    def test_a_queued_question_does_not_cross_into_a_new_thread(self):
        '''
        A follow-up button queues its question. Carrying it into a thread the
        operator just opened would ask it against the wrong conversation.
        '''
        state = {CONVERSATION: 'old', PENDING: 'a queued question'}

        _open(state, 'another')

        assert PENDING not in state


class TestTheStyleHooks:
    '''
    These prefixes reach the DOM as `st-key-<key>` and are what the stylesheet
    matches. A change here is a silent change to how the sidebar looks.
    '''

    def test_the_open_thread_is_matched_by_the_base_rule_too(self):
        assert THREAD_KEY in f'st-key-{OPEN_KEY}an-id'

    def test_the_other_buttons_are_not_caught(self):
        for key in ('conversation-new', 'conversation-rename',
                    'conversation-delete', 'conversation-filter'):
            assert f'st-key-{THREAD_KEY}' not in f'st-key-{key}'
