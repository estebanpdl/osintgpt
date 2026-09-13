# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_agentic_record.py
# Description: An answer through JSON and back. A replayed turn has to render
#   the way it did when it was new, trace included.
# =================================================================================

# import modules
import json

# import osintgpt
from osintgpt.agentic import (
    AgenticAnswer,
    answer_from_dict,
    answer_to_dict
)
from osintgpt.agentic.trace import Trace


def built() -> AgenticAnswer:
    trace = Trace()
    trace.say(1, 'Let me survey first.')
    trace.record(
        1, 'semantic_search', {'query': 'aardvark', 'limit': 8},
        count=3, unit='passage', refs=('material/alpha.md',), seconds=0.42
    )
    trace.record(
        2, 'fetch_source', {'ref': 'material/alpha.md'},
        error='gone', seconds=0.01
    )

    return AgenticAnswer(
        question='What is alpha?',
        text='Aardvarks.',
        trace=trace,
        sources=['material/alpha.md'],
        followups=['What about beta?']
    )


class TestRoundTrip:
    def test_the_answer_survives(self):
        restored = answer_from_dict(answer_to_dict(built()))

        assert restored.question == 'What is alpha?'
        assert restored.text == 'Aardvarks.'
        assert restored.sources == ['material/alpha.md']
        assert restored.followups == ['What about beta?']

    def test_the_trace_survives(self):
        restored = answer_from_dict(answer_to_dict(built()))
        first = restored.trace.entries[0]

        assert first.tool == 'semantic_search'
        assert first.arguments == {'query': 'aardvark', 'limit': 8}
        assert first.count == 3
        assert first.unit == 'passage'
        assert first.refs == ('material/alpha.md',)

    def test_the_rendered_lines_are_identical(self):
        '''
        The trace is what an analyst audits an old answer with, so a replayed
        one has to read exactly as it did when it was new.
        '''
        original = built()
        restored = answer_from_dict(answer_to_dict(original))

        assert restored.trace.lines() == original.trace.lines()

    def test_elapsed_time_reads_the_same_after_a_round_trip(self):
        '''
        Rounding on the way out and again on the way to the screen moves the
        number: 1.9548 stored as 1.955 displays as 1.96 rather than 1.95.
        '''
        import random

        random.seed(1)
        for _ in range(2000):
            trace = Trace()
            trace.record(
                1, 'semantic_search', {}, count=1, unit='passage',
                seconds=random.uniform(0, 3)
            )
            answer = AgenticAnswer(question='q', text='t', trace=trace)

            restored = answer_from_dict(answer_to_dict(answer))

            assert restored.trace.lines() == trace.lines()

    def test_a_failure_survives(self):
        restored = answer_from_dict(answer_to_dict(built()))
        failed = restored.trace.entries[1]

        assert not failed.ok
        assert failed.error == 'gone'

    def test_narration_keeps_its_round(self):
        restored = answer_from_dict(answer_to_dict(built()))

        assert restored.trace.narration[0].round == 1
        assert restored.trace.narration[0].text == 'Let me survey first.'

    def test_it_is_json(self):
        '''
        A transcript is a line in a file, so nothing in here may be an object
        json cannot write.
        '''
        data = answer_to_dict(built())

        assert answer_from_dict(json.loads(json.dumps(data))).text == (
            'Aardvarks.'
        )


class TestReadingIsDefensive:
    '''
    A transcript is a file a person can edit and an older osintgpt may have
    written. A turn missing part of itself should still show its answer.
    '''

    def test_an_empty_mapping_gives_an_empty_answer(self):
        assert answer_from_dict({}).text == ''

    def test_a_turn_without_a_trace_still_reads(self):
        restored = answer_from_dict({'answer': 'Aardvarks.'})

        assert restored.text == 'Aardvarks.'
        assert restored.trace.entries == []

    def test_a_malformed_call_does_not_take_the_turn_with_it(self):
        restored = answer_from_dict({
            'answer': 'Aardvarks.',
            'trace': {'calls': [
                'not a call',
                {'tool': 'semantic_search', 'results': 'many'}
            ]}
        })

        assert restored.text == 'Aardvarks.'
        assert len(restored.trace.entries) == 1
        assert restored.trace.entries[0].count == 0

    def test_a_degraded_answer_stays_degraded(self):
        restored = answer_from_dict({
            'answer': 'x', 'degraded': 'the model does not support tools'
        })

        assert restored.degraded
