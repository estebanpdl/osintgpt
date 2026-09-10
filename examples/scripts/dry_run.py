# -*- coding: utf-8 -*-

"""
Dry run — what osintgpt would index in a folder, without indexing it.

Reads and chunks every supported file, reports documents, chunks, tokens and a
cost estimate, and lists the structured files still waiting for someone to say
which of their fields carry content. Embeds nothing, so it is free to run and
free to re-run after changing a mapping.

Usage:

    python dry_run.py <path> [--map FILE:content=col[,col] [key=value ...]]

    python dry_run.py ../data
    python dry_run.py ../data --map records.csv:content=text
    python dry_run.py ../data --map records.csv:content=title,body \\
                              --map records.csv:timestamp=captured_at

A mapping key is one of: content, metadata, timestamp, author, identity,
records. Give content and metadata comma-separated lists; the rest take one
field name. Nested formats address fields by dotted path, e.g. user.name.
"""

# import modules
import argparse
import sys

# import submodules
from pathlib import Path

# import osintgpt modules
from osintgpt.config import DEFAULT_EMBEDDING_MODEL
from osintgpt.ingestion import FieldMapping, dry_run

LIST_KEYS = {'content', 'metadata'}
SINGLE_KEYS = {'timestamp', 'author', 'identity', 'records'}


def parse_mappings(pairs):
    """
    Turn repeated --map arguments into a mapping per file.

    Several --map arguments for the same file are merged, so field roles can be
    given one at a time rather than in a single long string.
    """
    collected = {}

    for pair in pairs or []:
        if ':' not in pair or '=' not in pair:
            raise SystemExit(
                f'--map {pair!r} should look like FILE:key=value'
            )

        name, assignment = pair.split(':', 1)
        key, _, value = assignment.partition('=')
        key = key.strip()

        if key in LIST_KEYS:
            values = tuple(v.strip() for v in value.split(',') if v.strip())
            existing = collected.setdefault(name, {}).get(key, ())
            collected[name][key] = existing + values
        elif key in SINGLE_KEYS:
            collected.setdefault(name, {})[key] = value.strip()
        else:
            valid = ', '.join(sorted(LIST_KEYS | SINGLE_KEYS))
            raise SystemExit(f'unknown mapping key {key!r}; use one of: {valid}')

    return {name: FieldMapping(**roles) for name, roles in collected.items()}


def report_unconfigured(run):
    """
    Print the fields of every structured file still awaiting a decision.

    This is a description, not a recommendation: length and uniqueness say what
    a value looks like, and which field is content remains the operator's call.
    """
    for preview in run.unconfigured:
        print(f'\n{preview.path.name} — needs a field mapping')
        for name, report in preview.fields.items():
            flags = 'unique' if report['unique'] else ''
            print(
                f"  {name:<24} avg {report['average_length']:>5} chars  "
                f"{report['filled']:>3}/{report['sampled']:<3} filled  "
                f"{flags:<7} e.g. {report['example'][:44]!r}"
            )
        print(f'\n  example: --map {preview.path.name}:content=<field>')


def report_files(run):
    """Print what each readable file would contribute."""
    if not run.readable:
        return

    print('\nwould index')
    for preview in sorted(run.readable, key=lambda p: -p.tokens):
        print(
            f'  {preview.path.name:<32} {preview.documents:>6,} docs  '
            f'{preview.chunks:>6,} chunks  {preview.tokens:>9,} tokens'
        )


def report_problems(run):
    """Print anything that could not be read, and anything not recognised."""
    for preview in run.failed:
        print(f'\nunreadable: {preview.path.name} — {preview.problem}')

    if run.unsupported:
        names = ', '.join(sorted({p.suffix for p in run.unsupported}))
        print(f'\nunsupported ({len(run.unsupported)} files): {names}')


def main():
    parser = argparse.ArgumentParser(
        description='Report what osintgpt would index, without indexing it.'
    )
    parser.add_argument('path', help='folder or file to preview')
    parser.add_argument(
        '--map', action='append', dest='maps', metavar='FILE:key=value',
        help='field roles for a structured file; repeatable'
    )
    parser.add_argument(
        '--embedding-model', default=DEFAULT_EMBEDDING_MODEL,
        help='model whose encoding counts the tokens '
             f'(default: {DEFAULT_EMBEDDING_MODEL})'
    )
    parser.add_argument(
        '--max-chars', type=int, default=None,
        help='chunk ceiling in characters, for trying a different size'
    )
    arguments = parser.parse_args()

    root = Path(arguments.path)
    if not root.exists():
        raise SystemExit(f'no such path: {root}')

    options = {
        'mappings': parse_mappings(arguments.maps),
        'embedding_model': arguments.embedding_model
    }
    if arguments.max_chars:
        options['max_chars'] = arguments.max_chars

    run = dry_run(root, **options)

    print(f'{root}\n{run.summary}')
    report_files(run)
    report_unconfigured(run)
    report_problems(run)

    # A corpus is not ready while part of it still needs a decision, so say so
    # in the exit code as well as the output.
    return 1 if run.unconfigured else 0


if __name__ == '__main__':
    sys.exit(main())
