'''Source context retains record boundaries and historical evidence versions.'''

from dataclasses import replace

from osintgpt.graph import Edge, assertion_context, graph_for, query_graph
from osintgpt.graph.retrieval import payload_size
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.ingestion.snapshots import load_snapshot


def test_context_reads_saved_version_after_current_source_changes(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Before.\n\nAlpha funds Beta.\n\nAfter.', quotes=['Alpha funds Beta.'])
    (case.project.paths.root / 'a.md').write_text('Alpha no longer funds Beta.')
    result = assertion_context(case.project, edge.assertion_id)
    assert result['status'] == 'ok' and result['text_basis'] == 'saved_snapshot'
    assert 'Before.' in result['text'] and 'After.' in result['text']
    assert 'no longer' not in result['text'] and result['quote_complete']
    assert result['verification']['current'] == 'not_found'
    assert result['verification']['snapshot'] == 'found'
    assert result['verification']['source_state'] == 'changed'
    (case.project.paths.root / 'a.md').unlink()
    assert assertion_context(case.project, edge.assertion_id)['verification']['source_state'] == 'missing'


def test_context_can_select_each_quote_and_occurrence(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.\n\nAlpha funds Beta.\n\nDesde 2024.',
                        quotes=['Alpha funds Beta.', 'Desde 2024.'])
    first = assertion_context(case.project, edge.assertion_id, context_chars=0)
    second = assertion_context(case.project, edge.assertion_id, occurrence=1, context_chars=0)
    assert first['text'] == second['text'] == 'Alpha funds Beta.'
    assert first['start'] < second['start'] and second['occurrences'] == 2
    other = assertion_context(case.project, edge.assertion_id, quote_index=1, context_chars=0)
    assert other['text'] == 'Desde 2024.' and other['quote_count'] == 2


def test_secondary_quote_without_saved_locations_is_located_in_its_own_source(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.\n\nSince 2024.', quotes=['Alpha funds Beta.', 'Since 2024.'])
    edge = replace(edge, provenance={key: value for key, value in edge.provenance.items() if key != 'supporting'})
    with graph_for(case.project) as graph:
        graph.add([], [edge])
    result = assertion_context(case.project, edge.assertion_id, quote_index=1, context_chars=0)
    assert result['status'] == 'ok' and result['text'] == 'Since 2024.'


def test_large_multilingual_quote_is_pageable_without_gaps(graph_case):
    case = graph_case
    text = '组织甲没有资助组织乙。' * 800
    edge = case.publish('a.md', text)
    collected, offset, pages = '', 0, []
    while True:
        result = assertion_context(case.project, edge.assertion_id, context_chars=0,
                                   offset=offset, limit_chars=8000, max_bytes=4096)
        assert result['status'] == 'ok' and payload_size(result) <= 4096
        assert result['text'] and not result['quote_complete']
        collected += result['text']
        pages.append(result)
        if 'next_offset' not in result:
            break
        assert result['next_offset'] > offset
        offset = result['next_offset']
    assert collected == text and len(pages) > 1
    assert 'response_bytes' in pages[0]['truncation_reasons']


def test_missing_or_wrong_snapshot_never_substitutes_current_text(graph_case):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.')
    other = case.publish('b.md', 'Private different record.')
    forged = replace(edge, provenance={**edge.provenance, 'source_version': other.provenance['source_version']})
    with graph_for(case.project) as graph:
        graph.add([], [forged])
    wrong = assertion_context(case.project, forged.assertion_id)
    assert wrong['status'] == 'evidence_unavailable' and 'different source' in wrong['problem']
    version = edge.provenance['source_version']
    (case.project.paths.extracts / 'text' / f'{version}.json').unlink()
    (case.project.paths.root / 'a.md').write_text('New source text.')
    missing = assertion_context(case.project, edge.assertion_id)
    assert missing['status'] == 'evidence_unavailable' and 'text' not in missing


def test_unknown_assertion_and_invalid_quote_are_explicit(graph_case):
    edge = graph_case.publish('a.md', 'Alpha funds Beta.')
    assert assertion_context(graph_case.project, 'assert_missing')['status'] == 'assertion_not_found'
    assert assertion_context(graph_case.project, edge.assertion_id, quote_index=4)['status'] == 'evidence_unavailable'
    row = query_graph(graph_case.project, 'Alpha')['claims'][0]
    assert row['read_context'] == {'tool': 'graph_evidence', 'assertion_id': edge.assertion_id, 'quote_index': 0}


def test_removed_registration_does_not_read_current_file(graph_case, monkeypatch):
    case = graph_case
    edge = case.publish('a.md', 'Alpha funds Beta.')
    Corpus.load(case.project.paths.sources).unregister('a.md')

    def forbidden(*args, **kwargs):
        raise AssertionError('Current unregistered material must not be opened')

    monkeypatch.setattr('osintgpt.graph.verify.load_snapshot', forbidden)
    result = assertion_context(case.project, edge.assertion_id)
    assert result['text'] == edge.evidence
    assert result['verification']['source_state'] == 'unregistered'
    assert result['verification']['current'] == 'unreadable'


def test_context_stays_inside_selected_structured_record(graph_case):
    project = graph_case.project
    path = project.paths.root / 'records.csv'
    path.write_text('id,body\none,Alpha funds Beta.\ntwo,Gamma funds Delta.\n')
    mapping = FieldMapping(content=('body',), identity='id')
    Corpus.load(project.paths.sources).register(path.name, mapping)
    snapshot = load_snapshot(path, mapping, ref=path.name, cache_dir=project.paths.extracts)
    document = snapshot.document('records.csv#one')
    edge = Edge('Alpha', 'Beta', 'funds', document.ref, document.text,
                provenance=snapshot.provenance(document, document.text))
    with graph_for(project) as graph:
        graph.add([], [edge])
    result = assertion_context(project, edge.assertion_id, context_chars=4000)
    assert result['text'] == 'Alpha funds Beta.' and result['source_ref'] == 'records.csv'
    assert result['ref'] == 'records.csv#one' and 'Gamma' not in result['text']
