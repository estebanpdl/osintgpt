# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_app_conversations.py
# Description: Chat that outlives the browser tab. Session state dies with the
#   tab, so the test for this is a second, empty session dict.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.agentic import AgenticAnswer
from osintgpt.agentic.trace import Trace
from osintgpt.app import (
    CONVERSATION,
    current_conversation,
    remember,
    select_conversation
)
from osintgpt.projects import list_conversations


@pytest.fixture
def project(tmp_path):
    return Project.create('Case', home=tmp_path / 'home')


def answered(question: str, text: str) -> AgenticAnswer:
    trace = Trace()
    trace.record(
        1, 'semantic_search', {'query': question},
        count=1, unit='passage', refs=('material/alpha.md',), seconds=0.1
    )

    return AgenticAnswer(
        question=question, text=text, trace=trace,
        sources=['material/alpha.md'], followups=['And beta?']
    )


class TestAConversationOutlivesTheSession:
    def test_a_new_session_still_has_it(self, project):
        '''
        A browser refresh is a new session dict, and a transcript held only
        in the old one is gone with it.
        '''
        state = {}
        remember(state, project, 'What is alpha?', answered(
            'What is alpha?', 'Aardvarks.'
        ))

        refreshed = {CONVERSATION: state[CONVERSATION]}
        conversation = current_conversation(refreshed, project)

        assert [t.question for t in conversation.turns] == ['What is alpha?']
        assert conversation.turns[0].answer['answer'] == 'Aardvarks.'

    def test_the_trace_comes_back_with_it(self, project):
        from osintgpt.agentic import answer_from_dict

        state = {}
        remember(state, project, 'q', answered('q', 'Aardvarks.'))

        conversation = current_conversation(
            {CONVERSATION: state[CONVERSATION]}, project
        )
        restored = answer_from_dict(conversation.turns[0].answer)

        assert restored.trace.entries[0].tool == 'semantic_search'
        assert restored.trace.entries[0].refs == ('material/alpha.md',)
        assert restored.followups == ['And beta?']

    def test_turns_accumulate_in_one_conversation(self, project):
        state = {}
        remember(state, project, 'first?', answered('first?', 'one'))
        remember(state, project, 'second?', answered('second?', 'two'))

        assert len(list_conversations(project)) == 1
        assert len(current_conversation(state, project)) == 2

    def test_a_new_conversation_starts_a_second_file(self, project):
        state = {}
        remember(state, project, 'first?', answered('first?', 'one'))

        select_conversation(state, None)
        remember(state, project, 'second?', answered('second?', 'two'))

        assert len(list_conversations(project)) == 2
        assert len(current_conversation(state, project)) == 1

    def test_selecting_an_older_one_loads_its_turns(self, project):
        state = {}
        remember(state, project, 'first?', answered('first?', 'one'))
        older = state[CONVERSATION]
        select_conversation(state, None)
        remember(state, project, 'second?', answered('second?', 'two'))

        select_conversation(state, older)

        assert [t.question for t in
                current_conversation(state, project).turns] == ['first?']

    def test_no_selection_means_no_conversation(self, project):
        assert current_conversation({}, project) is None

    def test_a_deleted_one_is_forgotten_rather_than_raising(self, project):
        from osintgpt.projects import delete_conversation, open_conversation

        state = {}
        remember(state, project, 'q', answered('q', 'one'))
        delete_conversation(open_conversation(project, state[CONVERSATION]))

        assert current_conversation(state, project) is None
        assert CONVERSATION not in state


class TestConversationsAreScopedToTheirProject:
    def test_one_project_does_not_see_another_s(self, tmp_path):
        home = tmp_path / 'home'
        first = Project.create('Case Alpha', home=home)
        second = Project.create('Case Beta', home=home)

        state = {}
        remember(state, first, 'q', answered('q', 'one'))

        assert current_conversation(state, second) is None
        assert list_conversations(second) == []
