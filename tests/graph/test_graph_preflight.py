'''Preflight enumerates the corpus without opening a provider or changing the graph.'''

import pytest

from osintgpt.graph import ExtractionOptions, build_graph
from osintgpt.graph.preflight import preflight_graph
from osintgpt.ingestion import Corpus, FieldMapping
from test_build_status import EMPTY, Generator, project
from test_graph_recovery import long_source


def test_fresh_preflight_matches_actual_window_count_and_does_not_create_graph(project):
    long_source(project)
    generator = Generator()
    estimate = preflight_graph(project, generator)
    assert generator.calls == 0
    assert not (project.paths.root / 'graph.sqlite').exists()
    counts = estimate['counts']
    assert counts['records_unstarted'] == 1 and counts['windows_pending'] > 1
    actual = build_graph(project, generator)
    assert counts['windows_total'] == actual.accounting.windows_total
    assert counts['extraction_calls_without_retries'] == actual.calls
    before = (project.paths.root / 'graph.sqlite').read_bytes()
    estimate = preflight_graph(project, generator)
    assert estimate['counts']['documents_unchanged'] == 1
    assert estimate['counts']['maximum_http_attempts'] == 0
    assert estimate['budget_cost_estimate_usd'] == 0
    assert before == (project.paths.root / 'graph.sqlite').read_bytes()


def test_preflight_includes_unstarted_structured_records_and_saved_windows(project):
    path = project.paths.root / 'rows.csv'
    path.write_text('id,body\none,First record\ntwo,Second record\nthree,Third record\n')
    corpus = Corpus.load(project.paths.sources)
    corpus.unregister('material')
    corpus.register(path, FieldMapping(content=('body',), identity='id'))
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, KeyboardInterrupt()))
    estimate = preflight_graph(project, Generator())
    counts = estimate['counts']
    assert counts['records'] == counts['windows_total'] == 3
    assert counts['windows_saved'] == counts['records_unstarted'] == 1
    assert counts['windows_pending'] == 2
    assert estimate['details'][0]['action'] == 'resume'
    rebuilt = preflight_graph(project, Generator(), rebuild=True)
    assert rebuilt['counts']['windows_saved'] == 0
    assert rebuilt['counts']['records_unstarted'] == 3


def test_preflight_changed_source_window_ignores_incompatible_work_without_claiming_call_maximum(project):
    long_source(project)
    from test_build_status import EMPTY
    with pytest.raises(KeyboardInterrupt):
        build_graph(project, Generator(EMPTY, KeyboardInterrupt()))
    estimate = preflight_graph(project, Generator())
    assert estimate['counts']['windows_saved'] == 1
    assert estimate['counts']['maximum_http_attempts'] is None
    changed = preflight_graph(project, Generator(), options=ExtractionOptions(context_budget=24000))
    assert changed['counts']['windows_saved'] == 0
    assert changed['counts']['maximum_http_attempts'] is None
    assert changed['counts']['checkpoints_incompatible'] == 1


def test_preflight_reports_unavailable_and_removed_sources_without_mutating(project):
    build_graph(project, Generator())
    path = project.paths.root / 'material' / 'a.md'
    path.unlink()
    before = (project.paths.root / 'graph.sqlite').read_bytes()
    estimate = preflight_graph(project, Generator())
    assert estimate['counts']['documents_unavailable'] == 1
    Corpus.load(project.paths.sources).unregister('material')
    estimate = preflight_graph(project, Generator())
    assert estimate['counts']['documents_removed'] == 1
    assert before == (project.paths.root / 'graph.sqlite').read_bytes()
