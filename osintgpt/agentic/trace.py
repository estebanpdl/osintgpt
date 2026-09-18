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
from typing import Any, Dict, List, Optional, Sequence, Tuple


# TouchedDocument class
@dataclass(frozen=True)
class TouchedDocument:
    '''
    A document one call reached, and what the call learned about it.

    Everything but `ref` is reported where the tool produced it and left empty
    where it did not: a directory listing knows nothing about relevance, and
    an invented zero would read as a real measurement.
    '''
    ref: str
    # Results from this document. The ranking shows only the best one, so a
    # document carrying six good passages and one carrying a single lucky hit
    # are indistinguishable without this.
    passages: int = 0
    # The best score among them. Cosine within one call, and meaningless
    # against another call's — the unit is not shared.
    best: Optional[float] = None
    # Heading path of the best-scoring passage: where in a long document the
    # match actually sits.
    section: str = ''
    timestamp: str = ''
    author: str = ''
    # Chunks matched, for exact search. A count, not a score.
    matches: int = 0
    # Which searched terms this document contains.
    terms: Tuple[str, ...] = ()

    @property
    def scored(self) -> bool:
        '''
        Returns:
            bool: Whether the call measured this document at all. False for a \
                listing, which reports existence and nothing else.
        '''
        return bool(
            self.passages or self.best is not None or self.matches
            or self.terms or self.timestamp or self.author or self.section
        )


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
    # Documents this call touched, in the order it returned them. A trace
    # that says a search found twenty passages but not which documents they
    # came from records that retrieval happened, not what it read.
    documents: Tuple[TouchedDocument, ...] = ()
    seconds: float = 0.0
    error: str = ''

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def refs(self) -> Tuple[str, ...]:
        '''
        Returns:
            Tuple[str, ...]: The documents by ref alone, for every caller that \
                wants the list rather than what was measured about it.
        '''
        return tuple(document.ref for document in self.documents)

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
        unit: str = 'result',
        documents: Sequence[TouchedDocument] = ()
    ) -> TraceEntry:
        entry = TraceEntry(
            round=round_number, tool=tool, arguments=dict(arguments),
            count=count, unit=unit, documents=tuple(documents),
            seconds=seconds, error=error
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


def _argument(key: str, value: Any) -> str:
    '''
    One argument as `key=value`, whole.

    The terms and queries here are what the model chose, and the trace is read
    to tune retrieval against exactly those — a clipped one cannot be compared
    with the next round's, or pasted back into a search.

    A ref is the exception, and not a value the model wrote: every ref in a
    project shares its leading path, so the last two segments identify the
    document where the whole path would only repeat the case folder. The full
    ref is carried alongside, under the call that read it.
    '''
    if key == 'ref' and isinstance(value, str):
        parts = PurePosixPath(value.replace('\\', '/')).parts
        if len(parts) > 2:
            value = '…/' + '/'.join(parts[-2:])

    return f'{key}={value!r}'
