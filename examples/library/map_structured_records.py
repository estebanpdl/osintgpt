"""Describe a structured file's fields, preview a mapping, and optionally register it.

Usage:
    python examples/library/map_structured_records.py FILE [--content FIELD ...]
        [--timestamp FIELD] [--author FIELD] [--identity FIELD]
        [--metadata FIELD ...] [--records PATH] [--min-chars N]
        [--register PROJECT]
"""

import argparse
from pathlib import Path

from osintgpt import Project
from osintgpt.ingestion import (
    Corpus,
    FieldMapping,
    describe_fields,
    load_documents,
    preview_mapping,
)


def print_fields(path: Path) -> None:
    """Print what each field looks like; choosing content stays the caller's call."""
    print(f'{path.name} fields')
    for name, facts in describe_fields(path).items():
        unique = 'unique' if facts['unique'] else ''
        print(
            f'  {name:<16} filled {facts["filled"]}/{facts["sampled"]}  '
            f'avg {facts["average_length"]:>4}  {unique:<6}  '
            f'e.g. {str(facts["example"])[:40]!r}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Choose field roles for a CSV, XLSX, JSON, or JSONL file.'
    )
    parser.add_argument('file', type=Path, help='structured file')
    parser.add_argument('--content', nargs='+', help='embedded field(s), body first')
    parser.add_argument('--timestamp', help='record date field')
    parser.add_argument('--author', help='author or channel field')
    parser.add_argument('--identity', help='unique record id field')
    parser.add_argument('--metadata', nargs='+', default=(), help='kept, not embedded')
    parser.add_argument('--records', help='JSON path to the record list, e.g. data.items')
    parser.add_argument('--min-chars', type=int, help='skip shorter content')
    parser.add_argument('--register', type=Path, metavar='PROJECT',
                        help='register the file with this mapping in a project')
    arguments = parser.parse_args()

    if not arguments.content:
        print_fields(arguments.file)
        print('\nChoose --content, then run again to preview.')
        return

    mapping = FieldMapping(
        content=tuple(arguments.content),
        metadata=tuple(arguments.metadata),
        timestamp=arguments.timestamp or '',
        author=arguments.author or '',
        identity=arguments.identity or '',
        records=arguments.records or '',
        min_chars=arguments.min_chars or 0,
    )

    preview = preview_mapping(arguments.file, mapping)
    print(preview.summary)
    print(f'shortest kept record: {preview.example!r}')

    for document in load_documents(arguments.file, mapping)[:3]:
        print(f'\n{document.ref}')
        print(f'  text      {document.text[:60]}')
        print(f'  author    {document.author}')
        print(f'  timestamp {document.timestamp}')
        print(f'  metadata  {document.metadata}')

    if arguments.register:
        project = Project.load(arguments.register)
        Corpus.load(project.paths.sources).register(arguments.file.resolve(), mapping)
        print(f'\nRegistered in {project.slug}; index the project to embed it.')


if __name__ == '__main__':
    main()
