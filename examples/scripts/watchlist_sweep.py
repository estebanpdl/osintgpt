# -*- coding: utf-8 -*-

"""
Watchlist sweep — find every indexed passage that mentions a list of literal terms.

Handles, URLs, hashes, account numbers and case identifiers are matched as
literal substrings in the indexed text and author fields. Runs over the
project's store only: no embedding call, no generation call, no API key.

Usage:

    python watchlist_sweep.py <project-dir> <watchlist.txt> [--out hits.csv]

    python watchlist_sweep.py ~/.osintgpt/projects/my-case terms.txt
    python watchlist_sweep.py ~/.osintgpt/projects/my-case terms.txt --out hits.jsonl

The watchlist holds one term per line; blank lines and lines starting with #
are skipped. --out writes CSV or JSONL, chosen by the file suffix.
"""

# import modules
import argparse
import csv
import json
import sys

# import submodules
from pathlib import Path

# import osintgpt modules
from osintgpt import Project, Settings, lexical_search
from osintgpt.vector_store import store_for

# Enough text either side of a match to judge it without opening the source.
CONTEXT_CHARS = 60
FIELDS = ('term', 'citation', 'ref', 'timestamp', 'author', 'context')


def read_watchlist(path):
    """Return the terms in file order, without duplicates or comments."""
    terms = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        term = line.strip()
        if term and not term.startswith('#') and term not in terms:
            terms.append(term)

    return terms


def context_for(text, term):
    """The match with its surroundings, or the text start when the author matched."""
    position = text.lower().find(term.lower())
    if position < 0:
        return ' '.join(text[:CONTEXT_CHARS * 2].split())

    start = max(position - CONTEXT_CHARS, 0)
    end = position + len(term) + CONTEXT_CHARS
    snippet = ' '.join(text[start:end].split())

    return f'…{snippet}…' if start or end < len(text) else snippet


def model_for(store, requested):
    """Pick the embedding model whose chunks to read, so no passage is counted twice."""
    models = store.models()
    if requested:
        return requested
    if len(models) > 1:
        raise SystemExit(
            f'the store holds chunks from {", ".join(models)}; '
            'choose one with --embedding-model'
        )

    return models[0] if models else None


def write_rows(rows, path):
    """Write rows as JSONL or CSV, by suffix."""
    path = Path(path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        with path.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return

    # The BOM lets spreadsheet software detect UTF-8 for non-Latin scripts.
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Sweep an indexed project for a list of literal terms.'
    )
    parser.add_argument('project', help='project directory')
    parser.add_argument('watchlist', help='one term per line')
    parser.add_argument('--out', help='write hits to .csv or .jsonl')
    parser.add_argument(
        '--limit', type=int, default=200, help='passages to collect per term'
    )
    parser.add_argument('--embedding-model', help='model whose chunks to read')
    parser.add_argument('--env-file', help='.env file, for remote stores only')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    terms = read_watchlist(arguments.watchlist)
    if not terms:
        raise SystemExit(f'{arguments.watchlist} holds no terms')

    config = project.settings_for(Settings.from_env(arguments.env_file))
    store = store_for(project, config)
    rows = []
    try:
        model = model_for(store, arguments.embedding_model)
        # One term per search, so every hit says which term found it.
        for term in terms:
            for hit in lexical_search(
                project, [term], limit=arguments.limit,
                embedding_model=model, store=store
            ):
                chunk = hit.chunk
                rows.append({
                    'term': term,
                    'citation': chunk.citation,
                    'ref': chunk.ref,
                    'timestamp': chunk.timestamp,
                    'author': chunk.author,
                    'context': context_for(chunk.text, term)
                })
    finally:
        if hasattr(store, 'close'):
            store.close()

    found = {row['term'] for row in rows}
    for term in terms:
        count = sum(1 for row in rows if row['term'] == term)
        print(f'{term:<28} {count:>4} passage(s)')
    missing = [term for term in terms if term not in found]
    if missing:
        print(f'\nnot found: {", ".join(missing)}')

    if arguments.out:
        write_rows(rows, arguments.out)
        print(f'\n{len(rows)} hits written to {arguments.out}')
    else:
        for row in rows:
            print(f'\n[{row["term"]}] {row["citation"]}\n  {row["context"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
