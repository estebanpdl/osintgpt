'''Graph migrations preserve original data and expose legacy extraction limits.'''

import json
import sqlite3
from pathlib import Path

import pytest

from osintgpt import Project
from osintgpt.graph import GraphStore, build_graph, graph_for, query_graph, saved_graph_status
from osintgpt.graph.migration import migrate_graph
from osintgpt.graph.records import Edge, Entity
from osintgpt.graph.state import SCHEMA_VERSION, build_lock
from osintgpt.ingestion import Corpus
from test_build_status import Generator

LEGACY_SCHEMA = '''
CREATE TABLE entities (key TEXT PRIMARY KEY, name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT '', mentions INTEGER NOT NULL DEFAULT 0);
CREATE TABLE edges (id INTEGER PRIMARY KEY, source_key TEXT NOT NULL, target_key TEXT NOT NULL,
    relation TEXT NOT NULL, ref TEXT NOT NULL, evidence TEXT NOT NULL,
    UNIQUE (source_key, target_key, relation, ref, evidence));
CREATE INDEX edges_by_source ON edges(source_key);
CREATE INDEX edges_by_target ON edges(target_key);
CREATE INDEX edges_by_ref ON edges(ref);
'''


@pytest.fixture
def legacy(tmp_path):
    project = Project.create('Legacy graph', home=tmp_path).with_settings(graph_enabled=True)
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_SCHEMA)
    return project, path


def populate(path, ref='source.md'):
    with sqlite3.connect(path) as connection:
        connection.executemany('INSERT INTO entities VALUES (?, ?, ?, ?)',
                               [('alpha', 'Álpha', 'person', 7), ('beta', '贝塔', 'organization', 3)])
        connection.execute('INSERT INTO edges VALUES (42, ?, ?, ?, ?, ?)',
                           ('alpha', 'beta', 'apoya', ref, 'Álpha apoya a 贝塔.'))


def snapshot(path):
    with sqlite3.connect(path) as connection:
        return '\n'.join(connection.iterdump())


def test_dry_run_leaves_database_and_directory_unchanged(legacy):
    project, path = legacy
    populate(path)
    before, files = path.read_bytes(), set(path.parent.iterdir())
    result = migrate_graph(project, dry_run=True)
    assert result['status'] == 'migration_required' and result['from_version'] == 0
    assert result['entities'] == 2 and result['edges'] == 1
    assert result['requires_reextraction'] and result['backup'] is None
    assert path.read_bytes() == before and set(path.parent.iterdir()) == files


def test_empty_legacy_is_backed_up_and_becomes_never_built(legacy):
    project, path = legacy
    original = snapshot(path)
    result = migrate_graph(project)
    assert result['status'] == 'migrated' and result['provider_calls'] == 0
    assert snapshot(Path(result['backup'])) == original
    assert path.name == 'graph.sqlite'
    status = saved_graph_status(project)
    assert status['schema_version'] == SCHEMA_VERSION and status['status'] == 'never_built'
    assert not status['available'] and not any(status['legacy'].values())
    assert build_graph(project, Generator()).status == 'complete'


def test_populated_migration_preserves_claims_and_does_not_invent_evidence(legacy):
    project, path = legacy
    populate(path)
    (path.parent / 'source.md').write_text('Álpha apoya a 贝塔.', encoding='utf-8')
    Corpus.load(project.paths.sources).register('source.md')
    original = snapshot(path)
    result = migrate_graph(project)
    assert snapshot(Path(result['backup'])) == original
    with graph_for(project) as graph:
        assert {e.key: e.mentions for e in graph.entities()} == {'alpha': 7, 'beta': 3}
        edge = graph.edges()[0]
        assert (edge.source, edge.target, edge.relation) == ('Álpha', '贝塔', 'apoya')
        assert edge.evidence == 'Álpha apoya a 贝塔.' and edge.ref == 'source.md'
        assert edge.source_id == 'alpha' and edge.target_id == 'beta'
        assert edge.relation_polarity == edge.assertion == 'unknown'
        assert not edge.event_time and not edge.provenance.get('source_version')
        assert edge.provenance['requires_reextraction']
        assert graph.connection.execute('SELECT id FROM edges').fetchone()[0] == 42
    status = saved_graph_status(project)
    assert status['status'] == 'complete_with_gaps' and status['available']
    assert migrate_graph(project)['requires_reextraction']
    assert status['legacy'] == {'entities': 2, 'edges': 1}
    assert status['usage']['complete'] is False and status['counts'] == {}
    assert status['started_at'] == '' and status['previous_runs'] == []
    result = query_graph(project, 'alpha', direction='outgoing')
    assert result['status'] == 'ok' and 'legacy' in result['note']
    assert result['claims'][0]['verification']['snapshot'] == 'unavailable'
    assert result['claims'][0]['verification']['current'] == 'found'
    assert build_graph(project, Generator()).status == 'complete'
    assert saved_graph_status(project)['legacy'] == {'entities': 0, 'edges': 0}


def test_migrated_claims_survive_failed_extraction(legacy):
    project, path = legacy
    populate(path)
    (path.parent / 'source.md').write_text('Source text', encoding='utf-8')
    Corpus.load(project.paths.sources).register('source.md')
    migrate_graph(project)
    result = build_graph(project, Generator(RuntimeError('Injected provider failure')))
    assert result.status == 'complete_with_gaps'
    assert saved_graph_status(project)['legacy']['edges'] == 1


def test_structured_reference_uses_existing_file_resolver(legacy):
    project, path = legacy
    populate(path, 'records#2026.csv#case/1')
    (path.parent / 'records#2026.csv').write_text('id,text\ncase/1,Example', encoding='utf-8')
    Corpus.load(project.paths.sources).register('records#2026.csv')
    migrate_graph(project)
    with graph_for(project) as graph:
        edge = graph.edges()[0]
        assert edge.ref == 'records#2026.csv#case/1'
        assert edge.provenance['source_ref'] == 'records#2026.csv'


def test_missing_graph_is_not_created_and_current_graph_is_unchanged(tmp_path):
    project = Project.create('No graph', home=tmp_path)
    path = project.paths.root / 'graph.sqlite'
    assert migrate_graph(project)['status'] == 'no_graph'
    assert not path.exists()
    with graph_for(project):
        pass
    before, files = path.read_bytes(), set(path.parent.iterdir())
    assert migrate_graph(project)['status'] == 'current'
    assert path.read_bytes() == before and set(path.parent.iterdir()) == files


def test_repeating_migration_does_not_add_backups(legacy):
    project, path = legacy
    migrate_graph(project)
    files = set(path.parent.iterdir())
    assert migrate_graph(project)['status'] == 'current'
    assert set(path.parent.iterdir()) == files


@pytest.mark.parametrize('damage', ['unknown_version', 'extra_table', 'altered_layout', 'trigger', 'corrupt'])
def test_unsupported_or_corrupt_graph_is_untouched(legacy, damage):
    project, path = legacy
    if damage == 'corrupt':
        path.write_bytes(b'Not a SQLite file')
    else:
        statement = {'unknown_version': 'PRAGMA user_version=99', 'extra_table': 'CREATE TABLE history (data TEXT)',
                     'altered_layout': 'ALTER TABLE entities ADD COLUMN extra TEXT',
                     'trigger': 'CREATE TRIGGER x AFTER INSERT ON entities BEGIN DELETE FROM edges; END'}[damage]
        with sqlite3.connect(path) as connection:
            connection.execute(statement)
    before, files = path.read_bytes(), set(path.parent.iterdir())
    with pytest.raises((ValueError, sqlite3.Error)):
        migrate_graph(project)
    assert path.read_bytes() == before and set(path.parent.iterdir()) == files


def test_partial_conversion_rolls_back_and_keeps_recoverable_backup(legacy, monkeypatch):
    project, path = legacy
    populate(path)
    original = snapshot(path)
    def failed(connection, project, version):
        connection.execute('ALTER TABLE entities ADD COLUMN new_column TEXT')
        connection.execute('DELETE FROM edges')
        raise ValueError('Injected conversion failure')
    monkeypatch.setattr('osintgpt.graph.migration.upgrade_records', failed)
    with pytest.raises(ValueError, match='original database was preserved.*Injected conversion failure'):
        migrate_graph(project)
    assert snapshot(path) == original
    backups = list(path.parent.glob('graph.backup-*.sqlite'))
    assert len(backups) == 1 and snapshot(backups[0]) == original


def test_backup_failure_does_not_modify_graph(legacy, monkeypatch):
    project, path = legacy
    original = snapshot(path)
    def failed(*args):
        raise OSError('No space for backup')
    monkeypatch.setattr('osintgpt.graph.migration._backup', failed)
    with pytest.raises(ValueError, match='No space for backup'):
        migrate_graph(project)
    assert snapshot(path) == original


def test_active_build_prevents_migration(legacy):
    project, path = legacy
    original = snapshot(path)
    with build_lock(path), pytest.raises(ValueError, match='already running'):
        migrate_graph(project)
    assert snapshot(path) == original and not list(path.parent.glob('graph.backup-*.sqlite'))


def test_other_sqlite_writer_prevents_migration(legacy):
    project, path = legacy
    original = snapshot(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute('BEGIN IMMEDIATE')
        with pytest.raises(ValueError, match='locked'):
            migrate_graph(project)
    finally:
        connection.close()
    assert snapshot(path) == original and not list(path.parent.glob('graph.backup-*.sqlite'))


def test_wal_backup_includes_committed_rows_and_old_readers_keep_their_snapshot(legacy):
    project, path = legacy
    writer = sqlite3.connect(path)
    reader = sqlite3.connect(path)
    try:
        writer.execute('PRAGMA journal_mode=WAL')
        populate(path)
        reader.execute('BEGIN')
        assert reader.execute('PRAGMA user_version').fetchone()[0] == 0
        result = migrate_graph(project)
        assert reader.execute('PRAGMA user_version').fetchone()[0] == 0
        with sqlite3.connect(result['backup']) as backup:
            assert backup.execute('SELECT COUNT(*) FROM edges').fetchone()[0] == 1
            assert backup.execute('PRAGMA user_version').fetchone()[0] == 0
        with graph_for(project) as graph:
            assert graph.edge_count == 1
    finally:
        reader.close()
        writer.close()


def test_missing_endpoint_rolls_back_instead_of_dropping_claim(legacy):
    project, path = legacy
    populate(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM entities WHERE key='beta'")
    original = snapshot(path)
    with pytest.raises(ValueError, match='counts'):
        migrate_graph(project)
    assert snapshot(path) == original


def test_entity_only_graph_retains_mentions_and_explicit_legacy_notice(legacy):
    project, path = legacy
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO entities VALUES ('x', 'Example', '', 12)")
    migrate_graph(project)
    assert saved_graph_status(project)['legacy'] == {'entities': 1, 'edges': 0}
    with graph_for(project) as graph:
        assert graph.entities()[0].mentions == 12


def test_zero_byte_graph_can_be_initialized(legacy):
    project, path = legacy
    path.write_bytes(b'')
    assert migrate_graph(project)['status'] == 'migrated'
    assert saved_graph_status(project)['status'] == 'never_built'


def test_v5_upgrade_retains_qualifications_checkpoints_and_audit(tmp_path):
    project = Project.create('Version five', home=tmp_path)
    path = project.paths.root / 'graph.sqlite'
    edge = Edge('Alpha', 'Beta', 'funds', 'source.md', 'No funding.', relation_polarity='negated', assertion='alleged',
                event_time='2020', provenance={'source_version': 'saved-v1'})
    with graph_for(project) as graph:
        graph.add([Entity('alpha', 'Alpha'), Entity('beta', 'Beta')], [edge])
        graph.connection.execute("INSERT INTO graph_runs VALUES (1, ?)", (json.dumps({'status': 'failed'}),))
        graph.connection.execute("INSERT INTO graph_responses VALUES ('source.md', 'source.md', 0, 'extract', 'hash', '{}')")
        graph.connection.commit()
        row = tuple(graph.connection.execute('SELECT * FROM edges').fetchone())
        graph.connection.executescript('DROP TABLE edges;')
        graph.connection.executescript('DROP TABLE entity_links; DROP TABLE identity_decisions;')
        graph.connection.executescript('''CREATE TABLE edges (
            id INTEGER PRIMARY KEY, source_key TEXT NOT NULL, target_key TEXT NOT NULL,
            relation TEXT NOT NULL, ref TEXT NOT NULL, evidence TEXT NOT NULL,
            source_ref TEXT NOT NULL, data TEXT NOT NULL,
            UNIQUE (source_key, target_key, relation, ref, evidence, data));
            PRAGMA user_version=5;''')
        data = json.loads(row[7])
        data.pop('assertion_id')
        graph.connection.execute('INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (*row[:7], json.dumps(data)))
        graph.connection.commit()
        history = graph.connection.execute('SELECT * FROM graph_runs').fetchall()
        responses = graph.connection.execute('SELECT * FROM graph_responses').fetchall()
    result = migrate_graph(project)
    assert result['from_version'] == 5 and not result['requires_reextraction']
    with graph_for(project) as graph:
        upgraded = graph.edges()[0]
        assert upgraded.relation_polarity == 'negated' and upgraded.assertion == 'alleged'
        assert upgraded.event_time == '2020' and upgraded.provenance['source_version'] == 'saved-v1'
        assert upgraded.assertion_id == edge.assertion_id
        assert graph.connection.execute('SELECT * FROM graph_runs').fetchall() == history
        assert graph.connection.execute('SELECT * FROM graph_responses').fetchall() == responses
