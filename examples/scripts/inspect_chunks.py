# -*- coding: utf-8 -*-

"""
Inspect the chunks a document would produce.

Chunking decides what retrieval can ever return, and its failures are silent —
nothing errors, the index populates, answers are merely worse. The only way to
judge it is to read the chunks.

Usage:

    python inspect_chunks.py <path> [selection] [options]

Selection (default: --random 3):

    --random N        N chunks chosen at random
    --index N         one chunk by position, negative counts from the end
    --first N         the first N
    --last N          the last N
    --longest N       the N largest, where packing problems show
    --shortest N      the N smallest, where fragments show
    --orphans         only chunks carrying no section path — the ones that
                      arrive with nothing saying where they came from
    --search TERM     only chunks containing TERM, case-insensitive
    --all             every chunk

Options:

    --stats           size distribution and context coverage, no chunk bodies
    --max-chars N     try a different ceiling without touching the library
    --full            print whole chunks rather than trimming long ones
    --seed N          make --random repeatable
    --map FILE:key=value   field roles for a structured file; repeatable
"""

# import modules
import argparse
import random
import sys

# import submodules
from pathlib import Path

# import osintgpt modules
from osintgpt.ingestion import (
    MAX_CHARS,
    FieldMapping,
    UnmappedSourceError,
    chunk_document,
    describe_fields,
    load_documents
)

LIST_KEYS = {'content', 'metadata'}
SINGLE_KEYS = {'timestamp', 'author', 'identity', 'records'}

# Long enough to judge a boundary, short enough to read several at once.
PREVIEW_CHARS = 700


def parse_mappings(pairs):
    """Turn repeated --map arguments into a mapping per file."""
    collected = {}

    for pair in pairs or []:
        if ':' not in pair or '=' not in pair:
            raise SystemExit(f'--map {pair!r} should look like FILE:key=value')

        name, assignment = pair.split(':', 1)
        key, _, value = assignment.partition('=')
        key = key.strip()

        if key in LIST_KEYS:
            values = tuple(v.strip() for v in value.split(',') if v.strip())
            collected.setdefault(name, {})[key] = (
                collected.get(name, {}).get(key, ()) + values
            )
        elif key in SINGLE_KEYS:
            collected.setdefault(name, {})[key] = value.strip()
        else:
            valid = ', '.join(sorted(LIST_KEYS | SINGLE_KEYS))
            raise SystemExit(f'unknown mapping key {key!r}; use one of: {valid}')

    return {name: FieldMapping(**roles) for name, roles in collected.items()}


def context_of(chunk):
    """
    Where a chunk sits: its section path, or the heading it opens with.

    A chunk with neither arrives at the model as free-floating text, which is
    the condition worth being able to count.
    """
    if chunk.path:
        return chunk.path
    if chunk.text.lstrip().startswith('#'):
        return chunk.text.splitlines()[0].lstrip('#').strip()

    return ''


def select(chunks, arguments):
    """Apply whichever selection was asked for, in a stable order."""
    numbered = list(enumerate(chunks))

    if arguments.all:
        return numbered
    if arguments.orphans:
        return [(i, c) for i, c in numbered if not context_of(c)]
    if arguments.search:
        needle = arguments.search.lower()
        return [(i, c) for i, c in numbered if needle in c.rendered.lower()]
    if arguments.index is not None:
        try:
            position = range(len(chunks))[arguments.index]
        except IndexError:
            raise SystemExit(
                f'--index {arguments.index} is outside 0..{len(chunks) - 1}'
            )
        return [(position, chunks[position])]
    if arguments.first:
        return numbered[:arguments.first]
    if arguments.last:
        return numbered[-arguments.last:]
    if arguments.longest:
        picked = sorted(numbered, key=lambda p: -len(p[1]))[:arguments.longest]
        return sorted(picked)
    if arguments.shortest:
        picked = sorted(numbered, key=lambda p: len(p[1]))[:arguments.shortest]
        return sorted(picked)

    count = min(arguments.random, len(numbered))

    return sorted(random.sample(numbered, count)) if numbered else []


def report_stats(chunks, ceiling, documents):
    """
    Size distribution, and how many chunks know where they came from.

    Records are reported differently from prose on purpose: a row has no
    headings to sit under, so counting it as context-less would read as a
    defect rather than as what a record is.
    """
    sizes = sorted(len(chunk) for chunk in chunks)
    placed = [chunk for chunk in chunks if context_of(chunk)]
    tiny = sum(1 for n in sizes if n < 200)

    print(f'documents      {len(documents):,}')
    print(f'chunks         {len(chunks):,}')
    print(f'size           min {sizes[0]}  median {sizes[len(sizes) // 2]}  '
          f'max {sizes[-1]}  cap {ceiling}')
    print(f'at the cap     {sum(1 for n in sizes if n > ceiling * 0.93)} '
          f'(>93% of ceiling)')
    print(f'under 200      {tiny} ({tiny / len(chunks):.0%}) — '
          'short chunks carry little for a vector to match on')

    if len(documents) > 1:
        with_metadata = sum(1 for d in documents if d.metadata or d.timestamp)
        print(f'context        records, not sections: {with_metadata:,} of '
              f'{len(documents):,} carry metadata')
        return

    print(f'context        {len(placed)} of {len(chunks)} '
          f'({len(placed) / len(chunks):.0%}) carry a heading or section path')
    orphaned = len(chunks) - len(placed)
    if orphaned:
        print(f'               {orphaned} arrive with nothing saying where '
              'they came from')


def report_chunks(selection, total, arguments):
    """Print the selected chunks with their position, size and context."""
    if not selection:
        print('nothing selected')
        return

    for position, chunk in selection:
        context = context_of(chunk) or '(no context)'
        print()
        print(f'{"─" * 74}')
        print(f'chunk {position + 1}/{total}   {len(chunk)} chars   {context}')
        print(f'{"─" * 74}')

        body = chunk.rendered
        print(body if arguments.full else body[:PREVIEW_CHARS])
        if not arguments.full and len(body) > PREVIEW_CHARS:
            print(f'… [{len(body) - PREVIEW_CHARS} more chars]')


def main():
    parser = argparse.ArgumentParser(
        description='Read the chunks a document would produce.'
    )
    parser.add_argument('path', help='file to chunk')

    picker = parser.add_mutually_exclusive_group()
    picker.add_argument('--random', type=int, default=3, metavar='N')
    picker.add_argument('--index', type=int, metavar='N')
    picker.add_argument('--first', type=int, metavar='N')
    picker.add_argument('--last', type=int, metavar='N')
    picker.add_argument('--longest', type=int, metavar='N')
    picker.add_argument('--shortest', type=int, metavar='N')
    picker.add_argument('--orphans', action='store_true')
    picker.add_argument('--search', metavar='TERM')
    picker.add_argument('--all', action='store_true')

    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--max-chars', type=int, default=MAX_CHARS, metavar='N')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--seed', type=int, metavar='N')
    parser.add_argument(
        '--map', action='append', dest='maps', metavar='FILE:key=value'
    )
    arguments = parser.parse_args()

    path = Path(arguments.path)
    if not path.is_file():
        raise SystemExit(f'no such file: {path}')
    if arguments.seed is not None:
        random.seed(arguments.seed)

    mappings = parse_mappings(arguments.maps)
    try:
        documents = load_documents(path, mappings.get(path.name))
    except UnmappedSourceError as error:
        fields = ', '.join(describe_fields(path))
        raise SystemExit(
            f'{error}\n\n'
            f'  fields: {fields}\n'
            f'  try:    --map {path.name}:content=<field>'
        )
    if not documents:
        raise SystemExit(f'{path.name} produced no documents')

    chunks = [
        chunk for document in documents
        for chunk in chunk_document(document.text, max_chars=arguments.max_chars)
    ]
    if not chunks:
        raise SystemExit(f'{path.name} produced no chunks')

    print(f'{path}  —  {len(documents):,} documents, {len(chunks):,} chunks')
    if documents[0].metadata:
        print(f'metadata: {documents[0].metadata}')
    print()

    report_stats(chunks, arguments.max_chars, documents)

    if not arguments.stats:
        report_chunks(select(chunks, arguments), len(chunks), arguments)

    return 0


if __name__ == '__main__':
    sys.exit(main())
