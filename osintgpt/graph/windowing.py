'''Bound source requests by UTF-8 size, including instructions and entity context.'''

import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import InitVar, dataclass, field

from osintgpt.ingestion.chunking import _SENTENCE_END

WINDOWING_VERSION = 3
_PARAGRAPH_END = re.compile(r'\r?\n[ \t]*\r?\n\s*')
_SENTENCE_TAIL = re.compile(r'["\'”’»）)\]」』】]*\s*')


def units(text):
    # A conservative input estimate across scripts, not an English chars/token ratio.
    return len(text.encode('utf-8'))


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


@dataclass(frozen=True)
class ExtractionOptions:
    context_budget: int = 32768
    # Keep the retired positional slot solely to reject old library calls with
    # guidance instead of interpreting their output cap as an entity allowance.
    output_tokens: InitVar[int | None] = None
    context_units: InitVar[int | None] = None
    max_calls: InitVar[int | None] = None
    repair_quotes: bool = True
    model_input_units: int | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self, output_tokens, context_units, max_calls):
        if output_tokens is not None:
            raise ValueError('Graph output_tokens has been removed; the selected provider determines the output allowance.')
        if context_units is not None or max_calls is not None:
            import warnings
            warnings.warn('Graph context_units and max_calls are retired and ignored; identity context and continuation are automatic.',
                          DeprecationWarning, stacklevel=2)
        if type(self.context_budget) is not int or self.context_budget < 64:
            raise ValueError('Source window allowance must be at least 64 UTF-8 bytes.')

    @property
    def input_units(self):
        return self.model_input_units

    def fits(self, system, user, schema=None):
        overhead = units(encode(schema)) if schema is not None else 0
        return self.input_units is None or units(system) + units(user) + overhead <= self.input_units


@dataclass(frozen=True)
class Window:
    start: int
    end: int


def windows(text, max_chars, max_units, overlap=1000):
    if max_chars <= 0 or max_units < 64 or overlap < 0:
        raise ValueError('The extraction budget leaves insufficient room for source text.')
    paragraphs = [match.end() for match in _PARAGRAPH_END.finditer(text)]
    sentences = sorted({0, len(text), *paragraphs,
                        *(_SENTENCE_TAIL.match(text, match.end()).end()
                          for match in _SENTENCE_END.finditer(text))})
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        while units(encode(text[start:end])) > max_units:
            end = start + max(1, (end - start) * 3 // 4)
        if end < len(text):
            floor = start + (end - start) // 2
            for boundaries in (paragraphs, sentences):
                position = bisect_right(boundaries, end) - 1
                if position >= 0 and boundaries[position] > floor:
                    end = boundaries[position]
                    break
        if text[start:end].strip():
            result.append(Window(start, end))
        if end == len(text):
            break
        start = _overlap_start(text, start, end, overlap, max_units, sentences)
    return result


def _overlap_start(text, start, end, overlap, max_units, boundaries):
    if not overlap:
        return end
    target = end - min(overlap, (end - start) // 4)
    stop = bisect_left(boundaries, end)
    if stop < len(boundaries) and boundaries[stop] == end:
        first = bisect_right(boundaries, start + (end - start) // 2)
        candidates = sorted(boundaries[first:stop], key=lambda position: abs(position - target))
        for position in candidates:
            if units(encode(text[position:end])) <= max_units // 2:
                return position
    # Punctuation-free text and oversized sentences still need bounded overlap and progress.
    return max(start + 1, target)
