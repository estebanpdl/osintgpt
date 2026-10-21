'''Schema-7 upgrades preserve accepted meaning, historical bytes, and reusable paid work.'''

import json
import sqlite3
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from osintgpt.graph import ExtractionOptions, GraphStore, build_graph, graph_for, query_graph
from osintgpt.graph.checkpoints import Checkpoints
from osintgpt.graph.migration import migrate_graph
from osintgpt.graph.records import RejectedClaim
from osintgpt.graph.recipes import extraction_recipe, fingerprint
from test_graph_migration import snapshot
from test_relation_polarity import NoGeneration, legacy_recipe


def old_field(row):
    return {**{key: value for key, value in row.items() if key != 'relation_polarity'},
            'polarity': row['relation_polarity']}


def downgrade(project):
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('PRAGMA user_version=7')
    return path


@pytest.fixture
def version_seven(graph_case):
    project = graph_case.project
    quote = 'Álpha niega vínculos con 贝塔. The literal field "polarity" is mentioned.'
    edge = graph_case.publish('source.md', quote, source='Álpha', target='贝塔', assertion='alleged',
                              relation='denies links to', relation_polarity='negative', event_time='2026')
    edges = [edge, replace(edge, relation_polarity='positive'), replace(edge, relation_polarity='unknown')]
    prior_recipe = legacy_recipe(extraction_recipe(NoGeneration()))
    rejection = RejectedClaim(edge.ref, edge.ref, edge.provenance['source_version'], 1,
                              old_field(asdict(edge)), 'Original diagnostic containing polarity.')
    with graph_for(project) as graph:
        graph.add([], edges[1:])
        for index, row in enumerate(graph.connection.execute('SELECT id,data FROM edges').fetchall()):
            data = json.loads(row['data'])
            # Exercise old, new, and equal dual spellings without touching source/provenance text.
            if index != 1:
                data = old_field(data)
                if index == 2:
                    data['relation_polarity'] = data['polarity']
            graph.connection.execute('UPDATE edges SET data=? WHERE id=?', (json.dumps(data), row['id']))
        graph.state.completed(edge.ref, edge.provenance['source_fingerprint'], edge.provenance['source_version'],
                              recipe=fingerprint(prior_recipe), recipe_data=prior_recipe)
        journal = Checkpoints(graph)
        journal.prepare(edge.ref, edge.provenance['source_version'], fingerprint(prior_recipe), prior_recipe)
        record = journal.record(edge.ref, edge.ref, (), 1)
        record.save(1, [replace(e, provenance=edge.provenance) for e in graph.entities()], edges, [rejection])
        window = json.loads(graph.connection.execute('SELECT data FROM graph_windows').fetchone()[0])
        window['edges'] = [old_field(row) for row in window['edges']]
        graph.connection.execute('UPDATE graph_windows SET data=?', (json.dumps(window),))
        graph.connection.execute('INSERT INTO entity_links VALUES (?, ?, ?)', (edge.target_id, edge.source_id, 'review-1'))
        graph.connection.execute('INSERT INTO identity_decisions VALUES (?, ?)',
                                 ('review-1', json.dumps({'id': 'review-1', 'reason': 'Historical polarity wording.'})))
        graph.connection.execute("UPDATE graph_build SET counts=?,usage=?,details=? WHERE id=1",
                                 ('{"requests_attempted":17}', '{"tokens":12000}', '[{"polarity":"historical"}]'))
        graph.connection.execute('INSERT INTO graph_runs VALUES (1, ?)', ('{"status":"failed","polarity":"historical"}',))
        for purpose in ('extraction', 'lookup-0', 'journal:original'):
            raw = '{"text": "{\\"edges\\":[{\\"polarity\\":\\"negative\\"}]}", "extra": "原始"}'
            graph.connection.execute('INSERT INTO graph_responses VALUES (?, ?, ?, ?, ?, ?)',
                                     (edge.ref, edge.ref, 2, purpose, 'exact-original-request', raw))
        graph.connection.commit()
    return project, downgrade(project), edges


def table_rows(path):
    with sqlite3.connect(path) as connection:
        return {name: connection.execute(f'SELECT * FROM {name} ORDER BY rowid').fetchall()
                for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_preview_and_upgrade_preserve_audit_ids_and_qualifications(version_seven, monkeypatch):
    project, path, edges = version_seven
    def forbidden(*args, **kwargs):
        raise AssertionError('A schema rename must not construct a provider.')
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)
    monkeypatch.setattr('osintgpt.llm.build_embedding_provider', forbidden)
    original, before, files = table_rows(path), path.read_bytes(), set(path.parent.iterdir())
    preview = migrate_graph(project, dry_run=True)
    assert preview['from_version'] == 7 and preview['to_version'] == 8
    assert preview['field_renames'] == {'published_assertions': 2, 'saved_assertions': 3, 'saved_windows': 1}
    assert not preview['requires_reextraction'] and preview['provider_calls'] == 0
    assert path.read_bytes() == before and set(path.parent.iterdir()) == files
    result = migrate_graph(project)
    assert 'backup-v7-' in result['backup']
    assert table_rows(Path(result['backup'])) == original
    current = table_rows(path)
    for table in original.keys() - {'edges', 'graph_windows'}:
        assert current[table] == original[table], table
    for old, new in zip(original['edges'], current['edges']):
        assert old[:7] == new[:7] and old[8:] == new[8:]
        old_data, new_data = json.loads(old[7]), json.loads(new[7])
        expected = {**old_data, 'relation_polarity': old_data.get('relation_polarity', old_data.get('polarity'))}
        expected.pop('polarity', None)
        assert new_data == expected and 'polarity' not in new_data
    old_window, new_window = (json.loads(rows['graph_windows'][0][3]) for rows in (original, current))
    assert old_window['entities'] == new_window['entities']
    assert old_window['rejected'] == new_window['rejected']  # Historical proposals retain their original spelling.
    assert [row['assertion_id'] for row in new_window['edges']] == [edge.assertion_id for edge in edges]
    assert all('polarity' not in row for row in new_window['edges'])
    with GraphStore(path, read_only=True) as graph:
        assert graph.connection.execute('PRAGMA user_version').fetchone()[0] == 8
        assert graph.edge_count == 3
    after, files = path.read_bytes(), set(path.parent.iterdir())
    assert migrate_graph(project)['status'] == 'current'
    assert path.read_bytes() == after and set(path.parent.iterdir()) == files


@pytest.mark.parametrize('location', ['published', 'window'])
def test_conflicting_fields_reject_preview_and_application_without_changes(version_seven, location):
    project, path, _ = version_seven
    with sqlite3.connect(path) as connection:
        table = 'edges' if location == 'published' else 'graph_windows'
        key, raw = connection.execute(f'SELECT rowid,data FROM {table} LIMIT 1').fetchone()
        data = json.loads(raw)
        claim = data if location == 'published' else data['edges'][0]
        claim.update(polarity='negative', relation_polarity='positive')
        connection.execute(f'UPDATE {table} SET data=? WHERE rowid=?', (json.dumps(data), key))
    before, files = path.read_bytes(), set(path.parent.iterdir())
    for dry_run in (True, False):
        with pytest.raises(ValueError, match='Conflicting polarity'):
            migrate_graph(project, dry_run=dry_run)
        assert path.read_bytes() == before and set(path.parent.iterdir()) == files


def test_failure_after_conversion_rolls_back_version_and_keeps_full_backup(version_seven, monkeypatch):
    project, path, _ = version_seven
    original = snapshot(path)
    def fail(*args):
        raise ValueError('Injected post-conversion failure')
    monkeypatch.setattr('osintgpt.graph.migration_records.validate_records', fail)
    with pytest.raises(ValueError, match='original database was preserved'):
        migrate_graph(project)
    assert snapshot(path) == original
    backups = list(path.parent.glob('graph.backup-v7-*.sqlite'))
    assert len(backups) == 1 and snapshot(backups[0]) == original


def test_empty_seven_requires_explicit_upgrade(graph_case):
    project = graph_case.project
    with graph_for(project):
        pass
    path = downgrade(project)
    before = path.read_bytes()
    for read_only in (True, False):
        with pytest.raises(ValueError, match='schema version 7.*current version is 8'):
            GraphStore(path, read_only=read_only)
        assert path.read_bytes() == before
    assert migrate_graph(project)['status'] == 'migrated'
    with graph_for(project) as graph:
        assert not graph.entities() and not graph.edges()
        assert graph.state.read()['status'] == 'never_built'


@pytest.mark.parametrize('unfinished', [False, True])
def test_migrated_completed_sources_and_saved_windows_do_not_reextract(graph_case, unfinished):
    project = graph_case.project
    edge = graph_case.publish('one.md', 'Alpha supports Beta.', relation_polarity='positive')
    provider = NoGeneration()
    previous = legacy_recipe(extraction_recipe(provider))
    with graph_for(project) as graph:
        graph.state.completed(edge.ref, edge.provenance['source_fingerprint'], edge.provenance['source_version'],
                              recipe=fingerprint(previous), recipe_data=previous)
        if unfinished:
            journal = Checkpoints(graph)
            journal.prepare(edge.ref, edge.provenance['source_version'], fingerprint(previous), previous)
            record = journal.record(edge.ref, edge.ref, (), 1)
            record.save(1, [replace(e, provenance=edge.provenance) for e in graph.entities()], [edge], [])
            saved = json.loads(graph.connection.execute('SELECT data FROM graph_windows').fetchone()[0])
            saved['edges'] = [old_field(row) for row in saved['edges']]
            graph.connection.execute('UPDATE graph_windows SET data=?', (json.dumps(saved),))
        graph.connection.commit()
    downgrade(project)
    migrate_graph(project)
    result = build_graph(project, provider)
    assert result.status == 'complete' and result.calls == 0
    assert result.accounting.windows_reused == int(unfinished)
    assert result.skipped == int(not unfinished)
    assert query_graph(project, 'Alpha')['claims'][0]['assertion_id'] == edge.assertion_id


@pytest.mark.parametrize('change', ['source', 'model', 'extraction_rules'])
def test_migration_does_not_make_incompatible_work_reusable(graph_case, change):
    from osintgpt.llm.generation import GenerationResponse

    class LocalModel(NoGeneration):
        model = 'test'
        requests = 0

        def graph_options(self):
            return {'provider': 'fixture', 'model': self.model}

        def generate_response(self, *args, **kwargs):
            self.requests += 1
            return GenerationResponse('{"entities":[],"edges":[]}', status='completed')

    project, provider = graph_case.project, LocalModel()
    edge = graph_case.publish('one.md', 'Alpha supports Beta.')
    previous = legacy_recipe(extraction_recipe(provider))
    with graph_for(project) as graph:
        graph.state.completed(edge.ref, edge.provenance['source_fingerprint'], edge.provenance['source_version'],
                              recipe=fingerprint(previous), recipe_data=previous)
        journal = Checkpoints(graph)
        journal.prepare(edge.ref, edge.provenance['source_version'], fingerprint(previous), previous)
        record = journal.record(edge.ref, edge.ref, (), 1)
        record.save(1, [], [], [])
    downgrade(project)
    migrate_graph(project)
    if change == 'source':
        (project.paths.root / 'one.md').write_text('Changed source content.', encoding='utf-8')
    elif change == 'model':
        provider.model = 'different'
    options = ExtractionOptions(repair_quotes=change != 'extraction_rules')
    result = build_graph(project, provider, options=options)
    assert result.status == 'complete' and provider.requests == 1
    assert result.accounting.checkpoints_discarded == 1 and result.accounting.windows_reused == 0
    assert result.skipped == 0
