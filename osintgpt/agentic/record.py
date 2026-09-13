# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: record.py
# Description: An answer as plain data, and back. One shape serves the stored
#   transcript and the CLI's JSON, so the two cannot drift apart.
# =================================================================================

# type hints
from typing import Any, Dict

from .loop import AgenticAnswer
from .trace import Narration, Trace, TraceEntry


# an answer as plain data
def answer_to_dict(answer: AgenticAnswer) -> Dict[str, Any]:
    '''
    Everything an answer carries, in JSON-safe types.

    The trace travels with it always, never behind a flag: an answer nobody
    can audit later is the thing the trace exists to prevent. `rounds` and
    `reading` are derived rather than stored — they are written out because a
    reader of the JSON wants them, and ignored on the way back in.

    Args:
        answer (AgenticAnswer): The answer.

    Returns:
        Dict[str, Any]: The answer as data.
    '''
    trace = answer.trace

    return {
        'question': answer.question,
        'answer': answer.text,
        'sources': list(answer.sources),
        # Each is a complete question, so an interface can send one as
        # written — a numbered line in a terminal, a button in an app.
        'followups': list(answer.followups),
        'degraded': trace.degraded,
        'trace': {
            'rounds': trace.rounds,
            'calls': [
                {
                    'round': entry.round,
                    'tool': entry.tool,
                    'arguments': entry.arguments,
                    'results': entry.count,
                    'unit': entry.unit,
                    'documents': list(entry.refs),
                    # Full precision. The trace rounds once for display,
                    # and rounding again here moves the number: 1.9548 stored
                    # as 1.955 displays as 1.96 rather than 1.95.
                    'seconds': entry.seconds,
                    'error': entry.error
                }
                for entry in trace.entries
            ],
            'narration': [
                {'round': said.round, 'text': said.text}
                for said in trace.narration
            ],
            'reading': trace.reading
        }
    }


# an answer rebuilt from plain data
def answer_from_dict(data: Dict[str, Any]) -> AgenticAnswer:
    '''
    Rebuild an answer read back from a transcript.

    Every field is read defensively. A transcript is a file on disk that an
    older osintgpt may have written and a person may have edited, and a replay
    that raises takes the whole conversation with it — a turn missing its
    trace should still show its answer.

    Args:
        data (Dict[str, Any]): An answer as `answer_to_dict` wrote it.

    Returns:
        AgenticAnswer: The answer, with its trace.
    '''
    stored = data.get('trace') or {}

    trace = Trace(
        entries=[
            _entry(call) for call in stored.get('calls') or []
            if isinstance(call, dict)
        ],
        narration=[
            Narration(
                round=_whole(said.get('round')),
                text=str(said.get('text', ''))
            )
            for said in stored.get('narration') or []
            if isinstance(said, dict) and said.get('text')
        ],
        degraded=str(data.get('degraded') or '')
    )

    return AgenticAnswer(
        question=str(data.get('question', '')),
        text=str(data.get('answer', '')),
        trace=trace,
        sources=[str(ref) for ref in data.get('sources') or []],
        followups=[str(text) for text in data.get('followups') or []]
    )


def _entry(call: Dict[str, Any]) -> TraceEntry:
    arguments = call.get('arguments')

    return TraceEntry(
        round=_whole(call.get('round')),
        tool=str(call.get('tool', '')),
        arguments=arguments if isinstance(arguments, dict) else {},
        count=_whole(call.get('results')),
        unit=str(call.get('unit') or 'result'),
        refs=tuple(str(ref) for ref in call.get('documents') or []),
        seconds=_number(call.get('seconds')),
        error=str(call.get('error') or '')
    )


def _whole(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
