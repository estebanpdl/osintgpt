'''Conservative presentation projection with a reversible character map.

Only recognized Markdown presentation syntax is omitted. Code, mathematical
punctuation, escaped delimiters, URL destinations, and plain text stay literal.
Unsupported syntax stays literal too; correction can still supply a raw quote.
'''

import re
from pathlib import PurePosixPath


def is_markdown(document):
    explicit = document.text_format
    if explicit:
        return explicit == 'markdown'
    # Older snapshots predate text_format; these loaders already emitted Markdown.
    return PurePosixPath(document.ref.split('#', 1)[0]).suffix.lower() in {'.md', '.markdown', '.pdf', '.docx'}


def markdown_map(text):
    keep = bytearray(b'\1') * len(text)
    protected = bytearray(len(text))
    # Do not interpret presentation inside fenced/indented/inline code or math.
    fence, offset = None, 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r' {0,3}(`{3,}|~{3,})', line)
        if fence or marker or line.startswith(('    ', '\t')):
            protected[offset:offset + len(line)] = b'\1' * len(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
        offset += len(line)
    for match in re.finditer(r'(`+)([^`\n]*?)\1|\$[^$\n]+\$|\\[^\n]|https?://[^\s<>]+', text):
        protected[match.start():match.end()] = b'\1' * (match.end() - match.start())

    def omit(start, end):
        keep[start:end] = b'\0' * (end - start)

    for match in re.finditer(r'^ {0,3}(#{1,6}[ \t]+|>[ \t]?)', text, re.M):
        if not any(protected[match.start():match.end()]):
            omit(match.start(), match.end())
    # A link's label is display text; a URL alone must never become its label.
    for match in re.finditer(r'(?<!!)\[([^\[\]\n]+)\]\(([^()\n]+)\)', text):
        if not any(protected[match.start():match.start(2)]):
            omit(match.start(), match.start(1))
            omit(match.end(1), match.end())
    for marker in ('**', '__', '*', '_'):
        pattern = re.compile(r'(?<![\w\\])' + re.escape(marker) +
            r'(?=\S)([^\n]+?)(?<=\S)' + re.escape(marker) + r'(?!\w)')
        for match in pattern.finditer(text):
            left, right = match.start(), match.end() - len(marker)
            if (any(protected[match.start():match.end()]) or not all(keep[left:left + len(marker)])
                    or not all(keep[right:match.end()]) or marker[0] in match[1]):
                continue
            omit(left, left + len(marker))
            omit(right, match.end())
    positions = [index for index, retained in enumerate(keep) if retained]
    return ''.join(text[index] for index in positions), positions


def projection_index(document):
    from .provenance import _normalized_map
    text, positions = markdown_map(document.text)
    normalized, starts, ends = _normalized_map(text)
    return normalized, [positions[start] for start in starts], [positions[end - 1] + 1 for end in ends]


def comparable(text, *, markdown=False):
    from .provenance import _normalized_map
    return _normalized_map(markdown_map(text)[0] if markdown else text)[0].strip()


def supported_wording(value, quotations, document):
    '''Qualifiers retain case and wording, allowing only presentation changes.'''
    markdown = is_markdown(document)
    needle = comparable(value, markdown=markdown)
    return bool(needle) and any(needle in comparable(quote, markdown=markdown) for quote in quotations)
