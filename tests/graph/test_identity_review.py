'''Existing graphs can be reviewed and corrected without re-extraction.'''

from pathlib import Path
import sqlite3

import pytest

from osintgpt import Project
from osintgpt.graph import graph_for, extract_document, query_graph, saved_graph_status
from osintgpt.graph.identity_review import reconcile_identities, review_identities
from osintgpt.graph.migration import migrate_graph
from osintgpt.graph.state import build_lock
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot
from test_extraction_quality import Script, proposal


def publish(project, ref, target):
    text = f'Orion supports {target}.'
    path = project.paths.root / ref
    path.write_text(text, encoding='utf-8')
    Corpus.load(project.paths.sources).register(ref)
    snapshot = load_snapshot(path, ref=ref, cache_dir=project.paths.extracts)
    result = extract_document(Script(proposal(text, ('Orion', target))), ref, text,
                              document=snapshot.document(ref), snapshot=snapshot)
    with graph_for(project) as graph:
        graph.replace_source(snapshot, result.entities, result.edges, False, 'recipe', {})
    return result


@pytest.fixture
def pair(tmp_path):
    project = Project.create('Identity review', home=tmp_path).with_settings(graph_enabled=True)
    first, second = publish(project, 'one.md', 'Lyra'), publish(project, 'two.md', 'Vega')
    return project, first, second


def test_reconciliation_preserves_assertions_and_old_ids_then_undoes(pair):
    project, first, second = pair
    source, target = first.entities[0].key, second.entities[0].key
    path = project.paths.root / 'graph.sqlite'
    before = path.read_bytes()
    preview = reconcile_identities(project, source, target, reason='Reviewed evidence in both sources.')
    assert path.read_bytes() == before and preview['status'] == 'preview'
    applied = reconcile_identities(project, source, target, reason='Reviewed evidence in both sources.',
                                   apply=True, expected_revision=preview['revision'])
    assert Path(applied['backup']).is_file() and applied['provider_calls'] == 0
    assert saved_graph_status(project)['entities'] == 3
    result = query_graph(project, source)
    assert result['selected']['entity']['id'] == target
    assert {claim['assertion_id'] for claim in result['claims']} == {first.edges[0].assertion_id, second.edges[0].assertion_id}
    assert review_identities(project, target)['candidates'][0]['provenance']['mentions']
    reconcile_identities(project, undo=applied['id'], apply=True)
    assert len(query_graph(project, source)['claims']) == 1
    assert saved_graph_status(project)['entities'] == 4


def test_explicit_source_split_corrects_a_model_reuse_without_deleting_evidence(pair):
    project, first, second = pair
    source, target = first.entities[0].key, second.entities[0].key
    reconcile_identities(project, source, target, reason='Initial review.', apply=True)
    result = reconcile_identities(project, target, split_source='one.md', reason='Source describes a different namesake.', apply=True)
    separate = query_graph(project, result['to_id'])
    assert {row['assertion_id'] for row in separate['claims']} == {first.edges[0].assertion_id}
    assert len(query_graph(project, target)['claims']) == 1
    assert not query_graph(project, result['to_id'], target)['connected']
    with graph_for(project) as graph:
        assert graph.edge_count == 2


def test_stale_preview_build_lock_and_failed_write_preserve_graph(pair, monkeypatch):
    project, first, second = pair
    source, target = first.entities[0].key, second.entities[0].key
    with pytest.raises(ValueError, match='changed after preview'):
        reconcile_identities(project, source, target, reason='Review', apply=True, expected_revision='stale')
    with build_lock(project.paths.root / 'graph.sqlite'), pytest.raises(ValueError, match='already running'):
        reconcile_identities(project, source, target, reason='Review', apply=True)
    def fail(*args):
        raise OSError('Backup unavailable')
    monkeypatch.setattr('osintgpt.graph.identity_review._backup', fail)
    with pytest.raises(OSError, match='Backup unavailable'):
        reconcile_identities(project, source, target, reason='Review', apply=True)
    assert saved_graph_status(project)['entities'] == 4


def test_version_six_upgrade_preserves_paid_data_and_checkpoints(pair):
    project, first, second = pair
    path = project.paths.root / 'graph.sqlite'
    with sqlite3.connect(path) as connection:
        connection.executescript('DROP TABLE entity_links; DROP TABLE identity_decisions; PRAGMA user_version=6;')
        original = connection.execute('SELECT assertion_id, data FROM edges ORDER BY id').fetchall()
        connection.execute("INSERT INTO graph_responses VALUES ('pending.md','pending.md',1,'extraction','hash','{}')")
    original_bytes = path.read_bytes()
    preview = migrate_graph(project, dry_run=True)
    assert preview['from_version'] == 6 and preview['to_version'] == 8
    assert path.read_bytes() == original_bytes
    applied = migrate_graph(project)
    with sqlite3.connect(applied['backup']) as backup:
        assert backup.execute('PRAGMA user_version').fetchone()[0] == 6
        assert backup.execute('SELECT assertion_id,data FROM edges ORDER BY id').fetchall() == original
        assert backup.execute('SELECT COUNT(*) FROM graph_responses').fetchone()[0] == 1
    with graph_for(project) as graph:
        assert [tuple(row) for row in graph.connection.execute('SELECT assertion_id,data FROM edges ORDER BY id')] == original
        assert graph.connection.execute('SELECT COUNT(*) FROM graph_responses').fetchone()[0] == 1
    assert migrate_graph(project)['status'] == 'current'


def test_split_canonical_source_does_not_leave_stale_mentions_or_break_remaining_links(pair):
    project, first, second = pair
    source, target = first.entities[0].key, second.entities[0].key
    reconcile_identities(project, source, target, reason='Reviewed.', apply=True)
    split = reconcile_identities(project, target, split_source='two.md', reason='Distinct namesake.', apply=True)
    remaining = review_identities(project, target)['candidates'][0]
    assert {item['source_ref'] for item in remaining['evidence']} == {'one.md'}
    assert len(query_graph(project, target)['claims']) == len(query_graph(project, split['to_id'])['claims']) == 1


def test_source_removal_after_reconciliation_preserves_other_contributions(pair):
    project, first, second = pair
    source, target = first.entities[0].key, second.entities[0].key
    reconcile_identities(project, source, target, reason='Reviewed.', apply=True)
    with graph_for(project) as graph:
        graph.forget(['two.md'])
    remaining = query_graph(project, source)
    assert len(remaining['claims']) == 1 and remaining['claims'][0]['ref'] == 'one.md'
    assert remaining['selected']['entity']['id'] == target
