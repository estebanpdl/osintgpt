"""Export a closed, consistent legacy SQLite snapshot to a new CSV directory.

Usage:
    python examples/library/export_legacy_logs.py old-logs.sqlite new-export

The database must be a closed snapshot with no uncheckpointed WAL. This script
uses immutable read-only access and never modifies or upgrades its input.
"""

import argparse
from contextlib import closing
import csv
from pathlib import Path
import sqlite3


def export_logs(database: Path, destination: Path) -> None:
    database = database.resolve(strict=True)
    queries = {
        'chat_gpt_index': 'SELECT id, created_at FROM chat_gpt_index ORDER BY id',
        'chat_gpt_conversations': (
            'SELECT id, ref_id, chat_id, role, message '
            'FROM chat_gpt_conversations ORDER BY id'
        ),
    }
    uri = database.as_uri() + '?mode=ro&immutable=1'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        # Check both table shapes before creating any output.
        for query in queries.values():
            connection.execute(query + ' LIMIT 0')
        destination.mkdir(parents=True, exist_ok=False)
        for table, query in queries.items():
            cursor = connection.execute(query)
            with (destination / f'{table}.csv').open('x', encoding='utf-8', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(column[0] for column in cursor.description)
                writer.writerows(cursor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path, help='closed SQLite snapshot')
    parser.add_argument('destination', type=Path, help='new output directory')
    args = parser.parse_args()
    export_logs(args.database, args.destination)
    print(f'Exported both legacy tables to {args.destination}')


if __name__ == '__main__':
    main()
