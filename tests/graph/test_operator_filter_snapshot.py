'''Filters and exports share cached, revision-checked evidence without provider calls.'''

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from osintgpt.graph import Entity, graph_for
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.verification_cache import VerificationChanged
from osintgpt.graph import verify
from test_progressive_exploration import finish


def many(case, count=75):
    quote = 'Long source passage. ' * 30 + 'Coincidencia final 日本語.'
    edge = case.publish('notes.md', quote, relation='supports', assertion='alleged')
    with graph_for(case.project) as graph:
        graph.add([Entity(f'node-{i}', f'Node {i}') for i in range(1, count)], [
            replace(edge, target=f'Node {i}', target_id=f'node-{i}', relation='supports' if i % 2 else 'reports')
            for i in range(1, count)])
    return edge, quote


def test_cached_checks_full_quote_filters_and_export_scopes(graph_case, monkeypatch):
    edge, quote = many(graph_case)
    loads, checks = [], []
    original_load, original_check = verify.load_snapshot, verify.quote_locations
    def load(*args, **kwargs):
        loads.append(args[0])
        return original_load(*args, **kwargs)
    def check(*args, **kwargs):
        checks.append(args[1])
        return original_check(*args, **kwargs)
    monkeypatch.setattr(verify, 'load_snapshot', load)
    monkeypatch.setattr(verify, 'quote_locations', check)
    query = OperatorExploration(graph_case.project, 'alpha', direction='outgoing', refs=['notes.md'])
    finish(query, 7)
    assert len(loads) == 1 and len(query.rows) == 75
    checked = len(checks)
    filters = {'text': 'COINCIDENCIA FINAL 日本語', 'relations': ['supports'], 'verification': 'found'}
    matches = query.matching(**filters)
    assert len(matches) > 30 and all('Coincidencia' not in row['quotation_preview'] for row in matches)
    for offset in range(0, 75, 10):
        query.page(offset=offset, page_size=10)
    query.inspect_assertion(edge.assertion_id)
    all_results = query.prepare_export(filters=filters)
    page = query.prepare_export(scope='table_page', filters=filters,
                                assertion_ids=[row['assertion_id'] for row in matches[30:]],
                                display={'table_page': 2, 'rows_per_page': 30})
    diagram = query.prepare_export(scope='diagram', filters=filters,
                                   assertion_ids=[matches[0]['assertion_id'], matches[-1]['assertion_id']])
    assert len(loads) == 1 and len(checks) == checked
    assert len(all_results['edges']) == len(matches)
    assert all_results['metadata']['selection']['complete_matching_scope']
    assert all_results['metadata']['filters']['sources'] == ['notes.md']
    assert all_results['metadata']['filters']['direction'] == 'outgoing'
    assert all_results['metadata']['query']['verification_state']['checked_assertions'] == 75
    assert len(page['edges']) == len(matches) - 30 and not page['metadata']['selection']['complete_matching_scope']
    assert len(diagram['edges']) == 2 and diagram['metadata']['scope'] == 'diagram'
    for package in (all_results, page, diagram):
        assert package['metadata']['query']['query_snapshot']['id'] == query.token
        assert not package['metadata']['truncated']
        assert all(row['evidence_quotes'] == [quote] and row['assertion'] == 'alleged' for row in package['edges'])
        assert all(row['source_version'] == edge.provenance['source_version'] for row in package['edges'])
    with pytest.raises(ValueError, match='unique matches'):
        query.prepare_export(scope='diagram', filters={'relations': ['absent']}, assertion_ids=[edge.assertion_id])
    query.close()


def test_evidence_filters_distinguish_absence_from_unreadable(graph_case):
    graph_case.publish('found.md', 'Confirmed passage.')
    graph_case.publish('absent.md', 'Previous passage.')
    graph_case.publish('missing.md', 'Unavailable passage.')
    (graph_case.project.paths.root / 'absent.md').write_text('Different passage.', encoding='utf8')
    (graph_case.project.paths.root / 'missing.md').unlink()
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query, 1)
    for state, ref in [('found', 'found.md'), ('not_found', 'absent.md'), ('unreadable', 'missing.md')]:
        package = query.prepare_export(filters={'verification': state})
        assert [row['ref'] for row in package['edges']] == [ref]
        assert package['metadata']['selection']['matching_total'] == 1
    query.close()


def test_partial_export_keeps_unknown_total_and_complete_path_requires_all_steps(graph_case):
    many(graph_case)
    query = OperatorExploration(graph_case.project, 'alpha')
    query.advance(batch_size=3); query.pause()
    package = query.prepare_export()
    selection = package['metadata']['selection']
    assert selection['matching_discovered'] == len(query.rows) > 0
    assert selection['matching_total'] is None and not selection['query_complete']
    assert package['metadata']['truncated'] and not selection['complete_matching_scope']
    query.close()
    graph_case.publish('chain.md', 'Beta connects Gamma.', source='Beta', target='Gamma')
    query = OperatorExploration(graph_case.project, 'alpha', 'gamma', direction='outgoing', depth=2)
    finish(query)
    assert query.prepare_export()['metadata']['path_complete']
    partial = query.prepare_export(scope='table_page', assertion_ids=[query.rows[0]['assertion_id']])
    assert not partial['metadata']['path_complete']
    query.close()


@pytest.mark.parametrize('change', ['same_size', 'missing', 'registration', 'saved_text', 'transcription'])
def test_source_changes_invalidate_cached_query_and_export(graph_case, change):
    edge = graph_case.publish('notes.md', 'Alpha links Beta.')
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    prepared = query.prepare_export()
    path = graph_case.project.paths.root / 'notes.md'
    if change == 'same_size':
        stat = path.stat()
        path.write_text('Alpha links Zeta.', encoding='utf8')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'missing':
        path.unlink()
    elif change == 'registration':
        from osintgpt.ingestion import Corpus
        Corpus.load(graph_case.project.paths.sources).register('another.md')
    elif change == 'saved_text':
        (graph_case.project.paths.extracts / 'text' / f"{edge.provenance['source_version']}.json").write_text('{}')
    else:
        name = 'a' * 64 + '.md'
        path = graph_case.project.paths.extracts / name
        path.write_text('Cached transcription.', encoding='utf8')
        query.verification.watch_transcriptions(SimpleNamespace(transcriptions={name: 'old'}))
        path.write_text('Updated transcription.', encoding='utf8')
    assert not query.check_current() and query.state == 'invalidated'
    assert query.result['truncation_reasons'] == ['source_changed']
    assert not query.result['verification_state']['valid']
    with pytest.raises(VerificationChanged):
        query.prepare_export()
    assert prepared['edges'][0]['verification']['current'] == 'found'  # Historical files remain fixed snapshots.
    if change in ('same_size', 'missing'):
        fresh = OperatorExploration(graph_case.project, 'alpha')
        finish(fresh)
        assert fresh.rows[0]['verification']['current'] == ('not_found' if change == 'same_size' else 'unreadable')
        fresh.close()


def test_source_change_during_check_does_not_publish_mixed_rows(graph_case, monkeypatch):
    graph_case.publish('notes.md', 'Alpha links Beta.')
    original = verify.quote_locations
    def change(*args, **kwargs):
        (graph_case.project.paths.root / 'notes.md').write_text('Changed source.', encoding='utf8')
        return original(*args, **kwargs)
    monkeypatch.setattr(verify, 'quote_locations', change)
    query = OperatorExploration(graph_case.project, 'alpha')
    query.advance()
    assert query.state == 'invalidated' and not query.rows


def test_new_cached_transcription_invalidates_missing_page_checks(graph_case, monkeypatch):
    from osintgpt.ingestion.transcription import cache_path
    monkeypatch.setattr('osintgpt.ingestion.pdf.extract_page_texts', lambda path: [''])
    monkeypatch.setattr('osintgpt.ingestion.pdf.render_page', lambda *args: b'synthetic page')
    graph_case.publish('scan.pdf', 'fake PDF', quotes=['Alpha links Beta.'])
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    assert query.rows[0]['verification']['current'] == 'not_found'
    transcript = cache_path(graph_case.project.paths.extracts, b'synthetic page')
    transcript.write_text('Alpha links Beta.', encoding='utf8')
    assert not query.check_current() and query.state == 'invalidated'
    fresh = OperatorExploration(graph_case.project, 'alpha')
    finish(fresh)
    assert fresh.rows[0]['verification']['current'] == 'found'
    fresh.close()


@pytest.mark.parametrize('graph_change', [False, True])
def test_changes_during_export_are_rejected(graph_case, monkeypatch, graph_change):
    graph_case.publish('notes.md', 'Alpha links Beta.')
    query = OperatorExploration(graph_case.project, 'alpha')
    finish(query)
    from osintgpt.graph import operator_exports
    original = operator_exports.selection_package
    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        if graph_change:
            with graph_for(graph_case.project) as graph:
                graph.connection.execute("UPDATE entities SET name='Alpha reviewed' WHERE key='alpha'")
                graph.connection.commit()
        else:
            (graph_case.project.paths.root / 'notes.md').write_text('Changed source.', encoding='utf8')
        return result
    monkeypatch.setattr(operator_exports, 'selection_package', change)
    with pytest.raises(VerificationChanged):
        query.prepare_export()
    assert query.state == 'invalidated'
