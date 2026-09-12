# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: trace.py
# Description: What the model did to reach an answer. Reading traces is how
#   retrieval gets tuned, so the trace is a product surface, not a debug log.
# =================================================================================

# import submodules
from dataclasses import dataclass, field
from pathlib import PurePosixPath

# type hints
from typing import Any, Dict, List


# TraceEntry class
@dataclass(frozen=True)
class TraceEntry:
    '''
    One tool call: what was asked, what came back, and how long it took.
    '''
    round: int
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    count: int = 0
    # What `count` counts, singular, as the tool named it. Lines, passages
    # and documents are not comparable quantities, and a trace that calls
    # them all "results" reads as though they were.
    unit: str = 'result'
    seconds: float = 0.0
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def counted(self) -> str:
        '''
        Returns:
            str: The count with its unit, pluralized.
        '''
        unit = self.unit or 'result'

        return f'{self.count} {unit}' + ('' if self.count == 1 else 's')

    @property
    def arguments_line(self) -> str:
        '''
        Returns:
            str: The arguments the call was made with, one line, empty when \
                it took none. Shared with the app so a trace on screen and a \
                trace in a terminal say the same thing.
        '''
        return ', '.join(
            _argument(key, value) for key, value in self.arguments.items()
            if value not in (None, '', [], {})
        )

    @property
    def label(self) -> str:
        '''
        Returns:
            str: The call in one line, argument values included — a trace \
                that says only which tool ran cannot be read against another.
        '''
        outcome = f'error: {self.error}' if self.error else self.counted

        return (
            f'{self.tool}({self.arguments_line}) — {outcome}, '
            f'{self.seconds:.2f}s'
        )


# Narration class
@dataclass(frozen=True)
class Narration:
    '''
    Something the model said on its way to an answer, and when it said it.
    '''
    round: int
    text: str


# Trace class
@dataclass
class Trace:
    '''
    Everything a run did, in order.
    '''
    entries: List[TraceEntry] = field(default_factory=list)
    # What the model said on its way to the answer. Its reasoning is what
    # makes a trace explain a bad answer rather than merely display one.
    # Carries the round it belongs to, because narration read apart from the
    # calls it introduced explains nothing.
    narration: List[Narration] = field(default_factory=list)
    # Set when the tool loop could not run and the static pipeline answered.
    degraded: str = ''

    def record(
        self,
        round_number: int,
        tool: str,
        arguments: Dict[str, Any],
        count: int = 0,
        seconds: float = 0.0,
        error: str = '',
        unit: str = 'result'
    ) -> TraceEntry:
        entry = TraceEntry(
            round=round_number, tool=tool, arguments=dict(arguments),
            count=count, unit=unit, seconds=seconds, error=error
        )
        self.entries.append(entry)

        return entry

    def say(self, round_number: int, text: str) -> None:
        cleaned = (text or '').strip()
        if cleaned:
            self.narration.append(
                Narration(round=round_number, text=cleaned)
            )

    # every round that produced something, in order
    @property
    def round_numbers(self) -> List[int]:
        '''
        Returns:
            List[int]: Rounds that called a tool or spoke, ascending. A round \
                can do either without the other.
        '''
        numbers = {e.round for e in self.entries}
        numbers |= {n.round for n in self.narration}

        return sorted(numbers)

    def calls_in(self, round_number: int) -> List[TraceEntry]:
        '''
        Args:
            round_number (int): The round.

        Returns:
            List[TraceEntry]: Its calls, in the order they ran.
        '''
        return [e for e in self.entries if e.round == round_number]

    def said_in(self, round_number: int) -> List[Narration]:
        '''
        Args:
            round_number (int): The round.

        Returns:
            List[Narration]: What the model said that round.
        '''
        return [n for n in self.narration if n.round == round_number]

    @property
    def rounds(self) -> int:
        return max((e.round for e in self.entries), default=0)

    @property
    def calls(self) -> int:
        return len(self.entries)

    @property
    def failures(self) -> List[TraceEntry]:
        return [e for e in self.entries if e.error]

    @property
    def tools_used(self) -> List[str]:
        seen = []
        for entry in self.entries:
            if entry.tool not in seen:
                seen.append(entry.tool)

        return seen

    # what the trace says about how retrieval went
    @property
    def reading(self) -> List[str]:
        '''
        Observations an operator would otherwise have to derive by eye.

        Counts pinned at the limit mean truncation, and a tool that exists
        but is never called is a model-choice problem rather than a tool
        problem — the things an operator would otherwise derive by eye.

        Returns:
            List[str]: Plain-language notes, empty when nothing stands out.
        '''
        notes = []

        if self.degraded:
            notes.append(f'Static pipeline used: {self.degraded}')

        if not self.entries:
            notes.append('No tools were called — the answer used no retrieval.')

            return notes

        empty = [e for e in self.entries if e.ok and e.count == 0]
        if len(empty) == len(self.entries):
            notes.append(
                'Every call came back empty. The corpus may not cover this, '
                'or it may not be indexed.'
            )

        if self.failures:
            notes.append(
                f'{len(self.failures)} call(s) failed: '
                + '; '.join(e.error for e in self.failures[:3])
            )

        return notes

    @property
    def summary(self) -> str:
        if self.degraded:
            return f'static pipeline ({self.degraded})'

        if not self.entries:
            return 'no tool calls'

        return (
            f'{self.calls} calls over {self.rounds} rounds, '
            f'{", ".join(self.tools_used)}'
        )

    # the trace as lines to print
    def lines(self) -> List[str]:
        '''
        Returns:
            List[str]: One line per call, grouped by round, with the model's \
                narration where it spoke. Readable against a trace from any \
                other provider, because nothing here is provider-shaped.
        '''
        out: List[str] = []

        for number in self.round_numbers:
            out.append(f'round {number}')
            # A turn carries its words and its calls together, so what the
            # model said introduces the calls rather than reporting on them.
            for said in self.said_in(number):
                out.append(f'  said: {said.text}')
            for entry in self.calls_in(number):
                out.append(f'  {entry.label}')

        return out


def _argument(key: str, value: Any, width: int = 48) -> str:
    '''
    One argument as `key=value`, with a ref shown by where it ends.

    A ref is a path, and one long enough to need truncating loses its filename
    to it — leaving the drive and the case folder, which every ref in a
    project shares. The last two segments identify the document instead.
    '''
    if key == 'ref' and isinstance(value, str):
        parts = PurePosixPath(value.replace('\\', '/')).parts
        if len(parts) > 2:
            value = '…/' + '/'.join(parts[-2:])

    return f'{key}={_short(value, width)}'


def _short(value: Any, width: int = 48) -> str:
    text = repr(value)

    return text if len(text) <= width else text[:width - 1] + '…'
