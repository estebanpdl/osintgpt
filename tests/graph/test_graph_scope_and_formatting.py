'''Content-agnostic source scope and source-location preservation regressions.'''

import pytest

from osintgpt.graph import Entity, graph_for, discover_entities, query_graph, extract_document
from osintgpt.graph.export_snapshot import graph_package
from osintgpt.graph.windowing import ExtractionOptions
from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import TextSpan, quote_locations, text_spans
from test_extraction_quality import Script, proposal


def document(text, ref='fiction.md', text_format=''):
    return Document(ref, text, spans=text_spans(text), text_format=text_format)


@pytest.mark.parametrize('source,quote', [
    ('**Orion Institute** supports Lyra.', 'Orion Institute supports Lyra.'),
    ('**Instituto Orión** no financia a Lira.', 'Instituto Orión no financia a Lira.'),
    ('**مؤسسة نور** لا تدعم المجموعة.', 'مؤسسة نور لا تدعم المجموعة.'),
    ('**星河研究所** 不支持该项目。', '星河研究所 不支持该项目。'),
    ('# Results\n\n[Orion](https://example.invalid) supports **Lyra**.', 'Orion supports Lyra.'),
    ('__Orion__ supports *Lyra*.', 'Orion supports Lyra.'),
])
def test_presentation_matching_keeps_original_offsets_and_text(source, quote):
    doc = document(source)
    matches = quote_locations(doc, quote)
    assert matches and matches[0]['method'] == 'formatting'
    match = matches[0]
    assert source[match['start']:match['end']] == match['source_excerpt']
    assert match['spans']


@pytest.mark.parametrize('ref,fmt', [('fiction.txt', ''), ('fiction.pdf', 'plain'), ('fiction.docx', 'plain')])
def test_plain_representations_do_not_interpret_markdown(ref, fmt):
    assert not quote_locations(document('**Orion** supports Lyra.', ref, fmt), 'Orion supports Lyra.')


@pytest.mark.parametrize('ref', ['fiction.pdf', 'fiction.docx'])
def test_converted_markdown_keeps_page_metadata(ref):
    source = '**Orion** supports Lyra.'
    doc = Document(ref, source, text_format='markdown', spans=(TextSpan(0, len(source), page=3),))
    assert quote_locations(doc, 'Orion supports Lyra.')[0]['spans'][0]['page'] == 3


@pytest.mark.parametrize('source,quote', [
    ('**Orion** does not support Lyra.', 'Orion supports Lyra.'),
    ('**Orion** supports Lyra', 'Orion supports Lyra.'),
    ('`**Orion** supports Lyra.`', 'Orion supports Lyra.'),
    ('```text\n**Orion** supports Lyra.\n```', 'Orion supports Lyra.'),
    ('x * y = z', 'x y = z'),
    (r'\*Orion\* supports Lyra.', 'Orion supports Lyra.'),
    ('Orion pays $x * y$ units.', 'Orion pays $x y$ units.'),
    ('[Orion](https://example.invalid) supports Lyra.', 'https://example.invalid supports Lyra.'),
])
def test_meaningful_words_punctuation_code_and_urls_are_not_erased(source, quote):
    assert not quote_locations(document(source), quote)


def test_multiple_locations_and_ranges_are_not_silently_collapsed():
    source = '**Orion** supports Lyra.\n\n**Orion** supports Lyra.'
    doc = document(source)
    matches = quote_locations(doc, 'Orion supports Lyra.')
    assert len(matches) == 2
    assert not quote_locations(doc, 'Orion supports Lyra.', before=10)
    assert len(quote_locations(doc, 'Orion supports Lyra.', after=25)) == 1


def test_formatted_date_and_optional_bad_alias_keep_supported_claim():
    text = '**Orion** supports Lyra until **June 2**.'
    data = proposal('Orion supports Lyra until June 2.', ('Orion', 'Lyra'))
    data['edges'][0]['event_time'] = 'June 2'
    data['entities'][0]['aliases'] = [{'name': 'Invented alias', 'evidence': [text]}]
    data['entities'][0]['evidence'] += ['A fabricated extra quotation.']
    result = extract_document(Script(data), 'fiction.md', text, options=ExtractionOptions(repair_quotes=False))
    assert len(result.edges) == 1 and result.edges[0].event_time == 'June 2'
    assert result.accounting.field_warnings == 2
    assert result.accounting.rejected_entities == result.accounting.rejected_claims == 0
    data['edges'][0]['event_time'] = '2028-06-02'
    rejected = extract_document(Script(data), 'fiction.md', text, options=ExtractionOptions(repair_quotes=False))
    assert not rejected.edges and rejected.accounting.rejected_claims == 1


def test_all_sources_and_all_selected_find_source_beyond_twenty(graph_case):
    refs = [f'{index:02}.md' for index in range(25)]
    for index, ref in enumerate(refs):
        graph_case.publish(ref, f'N{index} funds B{index}.', source=f'N{index}', target=f'B{index}')
    project = graph_case.project
    unrestricted = query_graph(project, 'N24')
    selected = query_graph(project, 'N24', refs=refs)
    assert unrestricted['claims'] == selected['claims'] and len(selected['claims']) == 1
    assert query_graph(project, 'N24', refs=refs[:20])['empty_reason'] == 'source_filter'
    assert query_graph(project, 'N24', refs=[])['empty_reason'] == 'empty_source_selection'
    candidate = discover_entities(project, 'N24', refs=refs[:20])['candidates'][0]
    assert candidate['relationships'] == {'all_sources': 1, 'selected_sources': 0,
                                          'distinct_neighbors_all_sources': 1, 'distinct_neighbors_selected_sources': 0}
    from osintgpt.agentic.registry import run_tool
    from osintgpt.agentic.tools import ToolContext
    context = ToolContext(project, None)
    assert len(run_tool(context, 'graph_query', {'entity': 'N24', 'refs': refs}).payload['claims']) == 1
    found = run_tool(context, 'graph_entities', {'query': 'N24', 'refs': []})
    assert found.payload['candidates'][0]['relationships']['selected_sources'] == 0
    assert run_tool(context, 'graph_entities', {'query': 'N24', 'refs': 'invalid'}).error
    with graph_for(project) as graph:
        assert len(graph_package(graph, project, refs)['edges']) == 25
        assert graph_package(graph, project, [])['edges'] == []
        # Larger selections do not run into SQLite bind-parameter limits.
        assert len(graph.edges(refs + [f'absent-{i}' for i in range(40000)])) == 25


def test_isolated_identity_and_zero_step_path_are_distinct(graph_case):
    project = graph_case.project
    graph_case.publish('one.md', 'Orion funds Lyra.', source='Orion', target='Lyra')
    with graph_for(project) as graph:
        graph.add([Entity('isolated', 'Isolated')], [])
    assert query_graph(project, 'isolated')['empty_reason'] == 'isolated_identity'
    result = query_graph(project, 'isolated', 'isolated')
    assert result['connected'] and result['path_complete'] and 'zero-step' in result['note']
