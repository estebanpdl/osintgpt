"""The migration example preserves historical files and conversation ordering."""

from contextlib import closing
import csv
from pathlib import Path
import runpy
import sqlite3

import pytest

from osintgpt import Project
from osintgpt.projects.conversations import append_turn, start_conversation


EXAMPLE = Path(__file__).resolve().parents[2] / 'examples/library/export_legacy_logs.py'
export_logs = runpy.run_path(str(EXAMPLE))['export_logs']


@pytest.fixture
def legacy_project(tmp_path):
    project = Project.create('Historical case', home=tmp_path)
    with closing(sqlite3.connect(project.paths.store)) as connection:
        connection.executescript('''
            CREATE TABLE chat_gpt_index (id TEXT PRIMARY KEY, created_at VARCHAR(20) NOT NULL);
            CREATE TABLE chat_gpt_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ref_id TEXT NOT NULL,
                chat_id TEXT NOT NULL, role TEXT NOT NULL, message TEXT NOT NULL,
                FOREIGN KEY(ref_id) REFERENCES chat_gpt_index(id));
            INSERT INTO chat_gpt_index VALUES ('thread-a', '2020-01-01');
            INSERT INTO chat_gpt_index VALUES ('thread-b', '2020-02-01');
        ''')
        connection.executemany(
            'INSERT INTO chat_gpt_conversations(ref_id, chat_id, role, message) VALUES (?, ?, ?, ?)',
            [('thread-a', 'reply-a', 'user', '¿Quién?\nAlpha, Beta'),
             ('thread-b', 'reply-b', 'user', 'A separate question'),
             ('thread-a', 'reply-a', 'assistant', 'An answer')],
        )
        connection.commit()
    return project


def test_export_and_modern_thread_leave_old_database_unchanged(legacy_project, tmp_path):
    database = legacy_project.paths.store
    original = database.read_bytes()
    destination = tmp_path / 'export'
    export_logs(database, destination)
    thread = start_conversation(legacy_project, 'A modern question')
    assert append_turn(thread, 'A modern question', {'answer': 'Modern answer'})
    assert database.read_bytes() == original
    with (destination / 'chat_gpt_conversations.csv').open(encoding='utf-8', newline='') as output:
        rows = list(csv.DictReader(output))
    assert [row['id'] for row in rows] == ['1', '2', '3']
    assert [row['ref_id'] for row in rows] == ['thread-a', 'thread-b', 'thread-a']
    assert rows[0]['message'] == '¿Quién?\nAlpha, Beta'
    with (destination / 'chat_gpt_index.csv').open(encoding='utf-8', newline='') as output:
        assert len(list(csv.DictReader(output))) == 2


def test_existing_export_is_not_overwritten(legacy_project, tmp_path):
    destination = tmp_path / 'export'
    export_logs(legacy_project.paths.store, destination)
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        export_logs(legacy_project.paths.store, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_missing_database_is_not_created(tmp_path):
    database = tmp_path / 'missing.sqlite'
    with pytest.raises(FileNotFoundError):
        export_logs(database, tmp_path / 'export')
    assert not database.exists()
    assert not (tmp_path / 'export').exists()
