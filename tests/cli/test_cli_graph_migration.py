'''The graph migration command is explicit, keyless, and usable without enabling retrieval.'''

import json
import sqlite3
from pathlib import Path

from osintgpt.graph import graph_for, saved_graph_status
from test_cli_graph import clear_credentials, create_project, home, invoke, runner


def test_migration_preview_apply_and_repeat_without_provider(runner, home, monkeypatch):
    project = create_project(runner, home)
    clear_credentials(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Migration must not construct a provider')
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)
    monkeypatch.setattr('osintgpt.llm.build_embedding_provider', forbidden)
    path = project.paths.root / 'graph.sqlite'
    path.touch()
    args = ('graph', 'migrate', '--project', project.slug, '--json')
    preview = invoke(runner, home, *args, '--dry-run')
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.stdout)['status'] == 'migration_required'
    assert path.stat().st_size == 0 and not list(path.parent.glob('graph.backup-*'))
    result = invoke(runner, home, *args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data['status'] == 'migrated' and Path(data['backup']).is_file()
    assert data['provider_calls'] == 0 and not project.settings.graph_enabled
    assert saved_graph_status(project)['status'] == 'never_built'
    repeated = invoke(runner, home, *args)
    assert repeated.exit_code == 0 and json.loads(repeated.stdout)['status'] == 'current'
    assert len(list(path.parent.glob('graph.backup-*'))) == 1


def test_unknown_version_fails_with_json_and_preserves_file(runner, home):
    project = create_project(runner, home)
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=99')
    original = path.read_bytes()
    result = invoke(runner, home, 'graph', 'migrate', '--project', project.slug, '--json')
    assert result.exit_code != 0
    assert 'Unsupported graph schema version 99' in json.loads(result.stdout)['error']
    assert path.read_bytes() == original


def test_no_graph_is_reported_without_creating_one(runner, home):
    project = create_project(runner, home)
    result = invoke(runner, home, 'graph', 'migrate', '--project', project.slug)
    assert result.exit_code == 0 and 'no_graph' in result.output
    assert not (project.paths.root / 'graph.sqlite').exists()


def test_help_explains_migration_and_options(runner, home):
    result = invoke(runner, home, 'graph', 'migrate', '--help')
    assert result.exit_code == 0
    for term in ('backup', 'no model calls', '--dry-run', '--project', '--json'):
        assert term in result.output
    assert 'migrate' in invoke(runner, home, 'graph', '--help').output


def test_schema_seven_notice_and_upgrade_are_explicit_and_keyless(runner, home, monkeypatch):
    project = create_project(runner, home)
    clear_credentials(monkeypatch)
    with graph_for(project):
        pass
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=7')
    before = path.read_bytes()
    status = invoke(runner, home, 'graph', 'status', '--project', project.slug, '--json')
    assert status.exit_code != 0
    message = json.loads(status.stdout)['error']
    assert 'schema version 7' in message and 'current version is 8' in message and 'migrate --dry-run' in message
    args = ('graph', 'migrate', '--project', project.slug, '--json')
    preview = json.loads(invoke(runner, home, *args, '--dry-run').stdout)
    assert preview['from_version'] == 7 and preview['to_version'] == 8 and not preview['requires_reextraction']
    assert path.read_bytes() == before and not list(path.parent.glob('graph.backup-*'))
    applied = json.loads(invoke(runner, home, *args).stdout)
    assert applied['status'] == 'migrated' and applied['provider_calls'] == 0
    assert Path(applied['backup']).is_file() and not project.settings.graph_enabled
    assert saved_graph_status(project)['schema_version'] == 8
