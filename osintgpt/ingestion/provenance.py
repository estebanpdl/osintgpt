'''Locations in extracted text; offsets are Unicode characters, end exclusive.'''

import re
import unicodedata
from array import array
from bisect import bisect_left
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    page: int | None = None
    section: str = ''
    paragraph: int = 0
    transcribed: bool = False
    missing: bool = False


def text_spans(text):
    spans, headings = [], []
    start, end, offset = None, 0, 0

    def flush():
        nonlocal start
        if start is not None:
            spans.append(TextSpan(start, end, section=' > '.join(h[1] for h in headings),
                                  paragraph=len(spans) + 1))
            start = None

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        heading = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if not stripped or heading:
            flush()
        if heading:
            level = len(heading[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading[2]))
        if stripped:
            if start is None:
                start = offset + len(line) - len(line.lstrip())
            end = offset + len(line.rstrip())
        offset += len(line)
    flush()
    return tuple(spans)


def page_spans(text, pages):
    result, cursor = [], 0
    paragraphs = text_spans(text)
    for page in pages:
        if not page.text:
            continue
        start = text.index(page.text, cursor)
        end = start + len(page.text)
        for paragraph in paragraphs:
            left, right = max(start, paragraph.start), min(end, paragraph.end)
            if left < right:
                result.append(TextSpan(left, right, page.number, paragraph.section,
                                       paragraph.paragraph, page.transcribed, page.empty))
        cursor = end
    return tuple(result)


def locations(spans, start, end):
    return [
        {**asdict(span), 'start': max(start, span.start), 'end': min(end, span.end)}
        for span in spans if span.start < end and span.end > start
    ]


def _normalized_map(text):
    characters, starts, ends = [], array('I'), array('I')
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and (unicodedata.combining(text[end]) or (
            0x1100 <= ord(text[end]) <= 0x11FF
            and unicodedata.normalize('NFC', text[index:end + 1])
            != unicodedata.normalize('NFC', text[index:end]) + text[end]
        )):
            end += 1
        for character in unicodedata.normalize('NFC', text[index:end]):
            if character.isspace():
                if characters and characters[-1] == ' ':
                    ends[-1] = end
                    continue
                character = ' '
            characters.append(character)
            starts.append(index)
            ends.append(end)
        index = end
    return ''.join(characters), starts, ends


def quote_locations(document, quote, *, after=0, before=None, first_only=False):
    '''Keep every matching occurrence instead of choosing an ambiguous page.'''
    if not quote.strip():
        return []
    matches = _find_quote(document, quote, document.quote_index, after, before, first_only, 'normalized')
    if matches:
        return matches
    from .quotation_projection import is_markdown, comparable
    if not is_markdown(document):
        return []
    return _find_quote(document, comparable(quote, markdown=True), document.presentation_quote_index,
                       after, before, first_only, 'formatting')


def _find_quote(document, quote, index_data, after, before, first_only, method):
    text, starts, ends = index_data
    needle = ' '.join(unicodedata.normalize('NFC', quote).split())
    matches, cursor = [], bisect_left(starts, after)
    stop = len(text) if before is None else bisect_left(starts, before)
    while needle and (index := text.find(needle, cursor, stop)) >= 0:
        start = starts[index]
        end = ends[index + len(needle) - 1]
        if start < after or (before is not None and end > before):
            cursor = index + len(needle)
            continue
        spans = locations(document.spans, start, end)
        if spans and not any(span['missing'] for span in spans):
            excerpt = document.text[start:end]
            matches.append({'start': start, 'end': end, 'spans': spans,
                            'method': 'exact' if excerpt == quote else method,
                            'source_excerpt': excerpt})
            if first_only:
                break
        cursor = index + len(needle)
    return matches
