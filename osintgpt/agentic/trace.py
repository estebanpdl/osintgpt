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

# type hints
from typing import Any, Dict, List, Optional


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
    seconds: float = 0.0
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def label(self) -> str:
        '''
        Returns:
            str: The call in one line, argument values included — a trace \
                that says only which tool ran cannot be read against another.
        '''
        shown = ', '.join(
            f'{key}={_short(value)}' for key, value in self.arguments.items()
            if value not in (None, '', [], {})
        )
        outcome = f'error: {self.error}' if self.error else f'{self.count} results'

        return f'{self.tool}({shown}) — {outcome}, {self.seconds:.2f}s'


# Trace class
@dataclass
class Trace:
    '''
    Everything a run did, in order.
    '''
    entries: List[TraceEntry] = field(default_factory=list)
    # What the model said between rounds. Its reasoning is what makes a trace
    # explain a bad answer rather than merely display one.
    narration: List[str] = field(default_factory=list)
    # Set when the tool loop could not run and the static pipeline answered.
    degraded: str = ''

    def record(
        self,
        round_number: int,
        tool: str,
        arguments: Dict[str, Any],
        count: int = 0,
        seconds: float = 0.0,
        error: str = ''
    ) -> TraceEntry:
        entry = TraceEntry(
            round=round_number, tool=tool, arguments=dict(arguments),
            count=count, seconds=seconds, error=error
        )
        self.entries.append(entry)

        return entry

    def say(self, text: str) -> None:
        cleaned = (text or '').strip()
        if cleaned:
            self.narration.append(cleaned)

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
        current: Optional[int] = None

        for entry in self.entries:
            if entry.round != current:
                current = entry.round
                out.append(f'round {current}')
            out.append(f'  {entry.label}')

        for said in self.narration:
            out.append(f'  said: {said}')

        return out


def _short(value: Any, width: int = 48) -> str:
    text = repr(value)

    return text if len(text) <= width else text[:width - 1] + '…'
