'''Bounds apply to every script, to carried context, and to correction calls.'''

import pytest

from osintgpt.graph import ExtractionOptions, extract_document
from osintgpt.graph.windowing import Window, encode, units, windows
from test_extraction_quality import Script, proposal


@pytest.mark.parametrize('text', ['word ' * 30000, '甲公司没有资助乙公司。' * 10000,
                                 'ألفا لم تمول بيتا. ' * 10000, '\\"\n' * 30000],
                         ids=['latin', 'cjk', 'arabic', 'json-escaping'])
def test_all_source_text_is_covered_with_bounded_multilingual_requests(text):
    generator = Script({'entities': [], 'edges': []})
    options = ExtractionOptions(context_budget=16384, context_units=1024)
    result = extract_document(generator, 'long.md', text, options=options)
    assert result.ok and len(generator.calls) > 1
    covered = 0
    for system, payload, cap in generator.calls:
        source = payload['source']
        assert source['start'] <= covered
        assert source['text'] == text[source['start']:source['end']]
        covered = source['end']
        assert options.fits(system, encode(payload)) and cap is None
    assert covered == len(text)
    assert result.accounting.windows_completed == result.accounting.requests_succeeded == len(generator.calls)


def test_windows_prefer_paragraph_boundaries():
    text = 'First paragraph.\n\nSecond paragraph.\n\nThird paragraph.'
    parts = windows(text, 42, 500, overlap=0)
    assert text[:parts[0].end] == 'First paragraph.\n\nSecond paragraph.\n\n'
    assert ''.join(text[part.start:part.end] for part in parts) == text


def test_retired_record_limit_does_not_truncate_source_coverage():
    generator = Script({'entities': [], 'edges': []})
    with pytest.warns(DeprecationWarning, match='retired'):
        options = ExtractionOptions(max_calls=2)
    result = extract_document(generator, 'long.md', 'word ' * 50000, options=options)
    assert result.ok and len(generator.calls) > 2
    assert result.accounting.windows_completed == result.accounting.windows_total


def test_retired_record_limit_does_not_prevent_evidence_correction():
    generator = Script(proposal('Invented.'))
    with pytest.warns(DeprecationWarning, match='retired'):
        options = ExtractionOptions(max_calls=1)
    result = extract_document(generator, 'one.md', 'Alpha funds Beta.', options=options)
    assert result.ok and not result.edges
    assert result.accounting.repairs_attempted == 2
    assert result.accounting.structured_correction_requests == 1
    assert 'budget exhausted' not in result.rejected[0].repair_problem


def test_custom_generator_needs_no_output_limit():
    class Legacy:
        def generate(self, system, user):
            return '{"entities":[],"edges":[]}'

    result = extract_document(Legacy(), 'one.md', 'Source.')
    assert result.ok and result.accounting.output_limits_unenforced == 0


@pytest.mark.parametrize('sentence', [
    'The organization issued a statement. ',
    '¿La organización publicó el informe? ',
    'Организация опубликовала заявление. ',
    'هل أصدرت المنظمة البيان؟ ',
    '甲公司没有资助乙公司。',
    'アルファは声明を発表した。',
    'संस्था ने एक बयान जारी किया। ',
    'The report said “No funding.” ',
    '報告は「資金提供はない。」',
])
def test_overlap_carries_whole_sentences_across_scripts(sentence):
    text = sentence * 30
    parts = windows(text, len(sentence) * 6 + len(sentence) // 2, 5000, overlap=len(sentence))
    assert len(parts) > 1 and parts[0].start == 0 and parts[-1].end == len(text)
    for previous, current in zip(parts, parts[1:]):
        assert previous.start < current.start < previous.end < current.end
        assert current.start % len(sentence) == previous.end % len(sentence) == 0
        assert text[current.start:previous.end] == sentence


@pytest.mark.parametrize('text', ['ไม่มีเครื่องหมายวรรคตอน' * 200,
                               'x' * 5000, 'An oversized sentence ' * 300 + '.'])
def test_oversized_or_unpunctuated_sentences_fall_back_without_gaps(text):
    parts = windows(text, 160, 200, overlap=40)
    assert parts[0].start == 0 and parts[-1].end == len(text)
    for part in parts:
        assert units(encode(text[part.start:part.end])) <= 200
    for previous, current in zip(parts, parts[1:]):
        assert previous.start < current.start <= previous.end < current.end
        assert previous.end - current.start <= 40


def test_short_document_remains_whole_and_preserves_offsets():
    text = '  Première phrase.\r\n\r\n第二句话。  '
    assert windows(text, 1000, 1000) == [Window(0, len(text))]
