# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_conversations.py
# Description: Transcripts that survive a refresh. The guarantee worth keeping
#   is that nothing ever rewrites a file holding history.
# =================================================================================

# import modules
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.projects.conversations import (
    SLUG_CHARS,
    append_turn,
    conversations_dir,
    delete_conversation,
    list_conversations,
    load_conversation,
    open_conversation,
    recent_pairs,
    rename_conversation,
    slug_for,
    start_conversation,
    title_for
)

ANSWER = {'answer': 'Aardvarks.', 'sources': ['material/alpha.md']}


@pytest.fixture
def project(tmp_path):
    return Project.create('Case', home=tmp_path / 'home')


class TestASlugIsAFilename:
    def test_it_reads_as_the_question(self):
        assert slug_for('What is alpha?') == 'what-is-alpha'

    def test_it_cuts_at_a_word_boundary(self):
        slug = slug_for(
            'How are Blackcore and Clock Tower X related to each other in '
            'the assessments from April'
        )

        assert len(slug) <= SLUG_CHARS
        assert not slug.endswith('-')
        # Cut mid-word this would end '...relat'.
        assert slug.split('-')[-1] in {'related', 'clock', 'tower', 'x', 'to'}

    def test_a_script_it_cannot_spell_leaves_nothing(self):
        '''
        The filename is then the timestamp alone. Identity is the timestamp
        and the title inside the file carries the text, so this is lossy
        rather than broken.
        '''
        assert slug_for('Что говорит корпус?') == ''

    def test_one_very_long_word_is_still_bounded(self):
        assert len(slug_for('a' * 200)) <= SLUG_CHARS


class TestATitleIsForReading:
    def test_a_short_question_is_kept_whole(self):
        assert title_for('What is alpha?') == 'What is alpha?'

    def test_a_long_one_is_cut_at_a_word(self):
        title = title_for('How are Blackcore and Clock Tower X related to '
                          'each other across the April assessments')

        assert title.endswith('…')
        assert '  ' not in title

    def test_whitespace_is_collapsed(self):
        assert title_for('  what\n  is   alpha ') == 'what is alpha'


class TestNothingIsWrittenUntilThereIsSomethingToWrite:
    def test_starting_one_creates_no_file(self, project):
        start_conversation(project, 'What is alpha?')

        assert not conversations_dir(project).exists()

    def test_the_first_turn_creates_it(self, project):
        conversation = start_conversation(project, 'What is alpha?')

        append_turn(conversation, 'What is alpha?', ANSWER)

        assert conversation.path.is_file()
        assert len(list_conversations(project)) == 1

    def test_an_abandoned_thread_leaves_nothing_behind(self, project):
        start_conversation(project, 'What is alpha?')

        assert list_conversations(project) == []


class TestATranscriptSurvives:
    def test_turns_read_back_in_order(self, project):
        conversation = start_conversation(project, 'first?')
        append_turn(conversation, 'first?', {'answer': 'one'})
        append_turn(conversation, 'second?', {'answer': 'two'})

        reopened = load_conversation(conversation.path)

        assert [t.question for t in reopened.turns] == ['first?', 'second?']
        assert [t.answer['answer'] for t in reopened.turns] == ['one', 'two']

    def test_the_answer_is_kept_whole(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)

        reopened = load_conversation(conversation.path)

        assert reopened.turns[0].answer == ANSWER

    def test_the_title_comes_from_the_first_question(self, project):
        conversation = start_conversation(project, 'What is alpha?')
        append_turn(conversation, 'What is alpha?', ANSWER)

        assert load_conversation(conversation.path).title == 'What is alpha?'

    def test_it_opens_by_id(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)

        assert open_conversation(project, conversation.id) is not None

    def test_an_id_that_climbs_out_is_refused(self, project):
        assert open_conversation(project, '../../secrets') is None

    def test_a_truncated_last_line_does_not_hide_the_rest(self, project):
        '''
        Append-only is what makes this recoverable: a partial write costs the
        turn being written, never the conversation above it.
        '''
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'first?', {'answer': 'one'})
        with conversation.path.open('a', encoding='utf-8') as handle:
            handle.write('{"type": "turn", "question": "trunc')

        reopened = load_conversation(conversation.path)

        assert [t.question for t in reopened.turns] == ['first?']

    def test_a_record_from_a_newer_version_is_skipped(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'first?', {'answer': 'one'})
        with conversation.path.open('a', encoding='utf-8') as handle:
            handle.write('{"type": "something-new", "payload": 1}\n')

        assert len(load_conversation(conversation.path).turns) == 1


class TestRenamingAppends:
    def test_the_new_title_wins(self, project):
        conversation = start_conversation(project, 'What is alpha?')
        append_turn(conversation, 'What is alpha?', ANSWER)

        rename_conversation(conversation, 'Aardvark sightings')

        assert load_conversation(conversation.path).title == (
            'Aardvark sightings'
        )

    def test_no_turn_is_lost_to_a_rename(self, project):
        '''
        Rewriting the file to change one line would put every answer in it at
        risk. The rename is another append.
        '''
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'first?', {'answer': 'one'})
        append_turn(conversation, 'second?', {'answer': 'two'})

        rename_conversation(conversation, 'Renamed')
        reopened = load_conversation(conversation.path)

        assert len(reopened.turns) == 2
        assert reopened.title == 'Renamed'

    def test_renaming_before_the_first_turn_needs_no_file(self, project):
        conversation = start_conversation(project, 'q')

        rename_conversation(conversation, 'Renamed')
        append_turn(conversation, 'q', ANSWER)

        assert load_conversation(conversation.path).title == 'Renamed'


class TestListing:
    def test_it_is_most_recent_first(self, project):
        import os
        import time

        older = start_conversation(project, 'older question')
        append_turn(older, 'older question', ANSWER)
        time.sleep(0.01)
        newer = start_conversation(project, 'newer question')
        append_turn(newer, 'newer question', ANSWER)
        # Explicit rather than trusting the clock's resolution.
        os.utime(older.path, (1_600_000_000, 1_600_000_000))

        assert [c.id for c in list_conversations(project)] == [
            newer.id, older.id
        ]

    def test_listing_does_not_read_the_turns(self, project):
        '''
        A long history costs a header per file to list. The turns stay on disk
        until one is opened.
        '''
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)

        listed = list_conversations(project)[0]

        assert listed.title == 'q'
        assert listed.turns == []

    def test_a_project_with_none_lists_none(self, project):
        assert list_conversations(project) == []

    def test_a_rename_shows_in_the_listing(self, project):
        '''
        Listing parses only lines that could be a header, and a rename is a
        header appended after every turn written so far.
        '''
        conversation = start_conversation(project, 'q')
        for index in range(20):
            append_turn(conversation, f'question {index}', ANSWER)
        rename_conversation(conversation, 'Renamed')

        assert list_conversations(project)[0].name == 'Renamed'

    def test_a_turn_that_looks_like_a_header_is_not_one(self, project):
        '''
        The listing's fast path filters on a substring before parsing. A
        question quoting that substring must fail the parse, not rename the
        conversation.
        '''
        conversation = start_conversation(project, 'q')
        append_turn(
            conversation,
            'What does \'"type": "meta"\' mean in the transcript format?',
            ANSWER
        )

        listed = list_conversations(project)[0]

        assert listed.name == 'q'
        assert len(load_conversation(conversation.path).turns) == 1

    def test_deleting_removes_it(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', ANSWER)

        assert delete_conversation(conversation)
        assert list_conversations(project) == []


class TestTheSlidingWindow:
    '''
    W1=[p1,p2,p3], W2=[p2,p3,p4], W3=[p3,p4,p5] — the last k pairs, or fewer
    when fewer exist. Whole pairs only.
    '''

    def built(self, project, count: int):
        conversation = start_conversation(project, 'q1')
        for index in range(1, count + 1):
            append_turn(
                conversation, f'q{index}', {'answer': f'a{index}'}
            )

        return conversation

    def test_nothing_to_carry_on_the_first_question(self, project):
        assert recent_pairs(None, 3) == []

    def test_fewer_than_the_window(self, project):
        conversation = self.built(project, 2)

        assert recent_pairs(conversation, 3) == [
            ('q1', 'a1'), ('q2', 'a2')
        ]

    def test_exactly_the_window(self, project):
        conversation = self.built(project, 3)

        assert recent_pairs(conversation, 3) == [
            ('q1', 'a1'), ('q2', 'a2'), ('q3', 'a3')
        ]

    def test_it_slides(self, project):
        conversation = self.built(project, 5)

        assert recent_pairs(conversation, 3) == [
            ('q3', 'a3'), ('q4', 'a4'), ('q5', 'a5')
        ]

    def test_oldest_first(self, project):
        conversation = self.built(project, 4)
        pairs = recent_pairs(conversation, 3)

        assert pairs[0][0] == 'q2'
        assert pairs[-1][0] == 'q4'

    def test_zero_turns_it_off(self, project):
        assert recent_pairs(self.built(project, 5), 0) == []

    def test_a_negative_window_carries_nothing(self, project):
        assert recent_pairs(self.built(project, 5), -1) == []

    def test_answers_are_never_truncated(self, project):
        '''
        A half answer is something a model reasons from as though it were
        complete, so a turn is carried entire or dropped entire.
        '''
        long_answer = 'x' * 20_000
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', {'answer': long_answer})

        assert recent_pairs(conversation, 3) == [('q', long_answer)]

    def test_no_passages_travel_with_it(self, project):
        '''
        The tools retrieve those again, and re-sending them would crowd out
        the corpus material the next answer has to be grounded in.
        '''
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q', {
            'answer': 'Aardvarks.',
            'trace': {'calls': [{'documents': ['material/alpha.md']}]},
            'sources': ['material/alpha.md']
        })

        assert recent_pairs(conversation, 3) == [('q', 'Aardvarks.')]

    def test_a_turn_with_no_answer_is_skipped(self, project):
        conversation = start_conversation(project, 'q')
        append_turn(conversation, 'q1', {'answer': ''})
        append_turn(conversation, 'q2', {'answer': 'a2'})

        assert recent_pairs(conversation, 3) == [('q2', 'a2')]


class TestConversationsAreProjectState:
    def test_two_projects_do_not_share_them(self, tmp_path):
        home = tmp_path / 'home'
        first = Project.create('Case Alpha', home=home)
        second = Project.create('Case Beta', home=home)

        conversation = start_conversation(first, 'q')
        append_turn(conversation, 'q', ANSWER)

        assert len(list_conversations(first)) == 1
        assert list_conversations(second) == []

    def test_two_started_in_the_same_second_do_not_collide(self, project):
        first = start_conversation(project, 'same question')
        append_turn(first, 'same question', ANSWER)
        second = start_conversation(project, 'same question')
        append_turn(second, 'same question', ANSWER)

        assert first.path != second.path
        assert len(list_conversations(project)) == 2
