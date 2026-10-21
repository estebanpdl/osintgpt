'''Shared text flows from indexing to graph claims and source inspection.'''

import json

import pytest

from osintgpt import Project, index_project
from osintgpt.agentic import ToolContext, fetch_source, graph_query
from osintgpt.graph import build_graph, graph_for, verify_evidence, to_cypherl, to_json, saved_graph_status
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.ingestion.snapshots import load_snapshot
from osintgpt.graph import ExtractionOptions
from osintgpt.vector_store import SQLiteVectorStore


class Embedder:
    model = 'offline'
    supports_images = False

    def embed(self, texts, **kwargs):
        return [[1.0, 0.0] for text in texts]


class Generator:
    def __init__(self, quote='Alpha funds Beta.'):
        self.quote, self.prompts = quote, []

    def generate(self, system, user):
        self.prompts.append(user)
        source, _, target = self.quote.rstrip('.').split()
        return json.dumps({'entities': [{'name': source}, {'name': target}], 'edges': [{
            'source': source, 'target': target, 'relation': 'funds', 'evidence': self.quote}]})


@pytest.fixture
def project(tmp_path):
    project = Project.create('Provenance', path=tmp_path / 'project').with_settings(graph_enabled=True)
    project.save()
    return project


def source(project, name='paper.md', text='# Funding\n\nAlpha funds Beta.'):
    path = project.paths.root / name
    path.write_text(text, encoding='utf-8')
    Corpus.load(project.paths.sources).register(name)
    return path


def test_index_graph_verify_and_fetch_share_one_version(project, monkeypatch):
    source(project)
    assert not index_project(project, Embedder()).failed
    with SQLiteVectorStore(project.paths.store) as store:
        provenance = store.chunks_for('paper.md')[0].metadata['_osintgpt_provenance']
    def forbidden(*args, **kwargs):
        raise AssertionError('Text must be reused')
    monkeypatch.setattr('osintgpt.ingestion.snapshots.load_documents', forbidden)
    assert build_graph(project, Generator()).status == 'complete'
    with graph_for(project) as graph:
        edge = graph.edges()[0]
        assert edge.provenance['source_version'] == provenance['source_version']
        assert edge.provenance['matches'][0]['spans'][0]['section'] == 'Funding'
        assert json.loads(to_json(graph))['edges'][0]['provenance'] == edge.provenance
        assert 'provenance:' in to_cypherl(graph)
    result = verify_evidence(project).results[0]
    assert result.status == result.snapshot_status == 'found'
    assert result.source_state == 'unchanged'
    context = ToolContext(project, Embedder())
    fetched = fetch_source(context, 'paper.md', source_version=provenance['source_version'])
    assert fetched.ok and fetched.payload['provenance']['source_version'] == provenance['source_version']
    assert graph_query(context, entity='Alpha').payload['claims'][0]['provenance'] == edge.provenance


def test_changed_source_verification_preserves_original_evidence(project):
    path = source(project)
    build_graph(project, Generator())
    with graph_for(project) as graph:
        version = graph.edges()[0].provenance['source_version']
    path.write_text('Alpha no longer funds Beta.')
    result = verify_evidence(project).results[0]
    assert result.status == 'not_found' and result.snapshot_status == 'found'
    assert result.source_state == 'changed' and result.current_version != version
    context = ToolContext(project, Embedder())
    saved = fetch_source(context, 'paper.md', source_version=version)
    current = fetch_source(context, 'paper.md')
    assert 'Alpha funds Beta.' in saved.payload['text']
    assert saved.payload['provenance']['source_state'] == 'changed'
    assert 'no longer' in current.payload['text']
    path.unlink()
    result = verify_evidence(project).results[0]
    assert result.status == 'unreadable' and result.snapshot_status == 'found'
    assert result.source_state == 'missing'


def test_saved_version_cannot_be_used_to_read_another_source(project):
    first = source(project)
    source(project, 'other.md', 'Other text')
    snapshot = load_snapshot(first, ref='paper.md', cache_dir=project.paths.extracts)
    result = fetch_source(ToolContext(project, Embedder()), 'other.md', source_version=snapshot.version)
    assert not result.ok and 'different source' in result.error


def test_structured_records_are_separate_model_calls_and_evidence_scopes(project):
    path = source(project, 'records.csv', 'id,body\none,Alpha funds Beta.\ntwo,Gamma funds Delta.\n')
    Corpus.load(project.paths.sources).register(path.name, FieldMapping(content=('body',), identity='id'))
    generator = Generator()
    report = build_graph(project, generator, options=ExtractionOptions(repair_quotes=False))
    assert report.calls == 2 and report.accounting.records_loaded == 2
    assert 'Gamma funds Delta.' not in generator.prompts[0]
    assert 'Alpha funds Beta.' not in generator.prompts[1]
    with graph_for(project) as graph:
        edges = graph.edges(['records.csv'])
        assert {edge.ref for edge in edges} == {'records.csv#one'}
        assert edges[0].provenance['match_status'] == 'found'
        assert any(item.ref == 'records.csv#two' for item in graph.rejections())
    result = fetch_source(ToolContext(project, Embedder()), 'records.csv#two')
    assert result.ok and result.payload['text'] == 'Gamma funds Delta.'
    assert verify_evidence(project).not_found == 0
    build_graph(project, Generator('Gamma funds Delta.'), rebuild=True,
                options=ExtractionOptions(repair_quotes=False))
    with graph_for(project) as graph:
        assert graph.edge_count == 1 and graph.edges()[0].ref == 'records.csv#two'


def test_scanned_pdf_indexing_transcribes_once_and_all_other_reads_reuse_it(project, monkeypatch):
    source(project, 'scan.pdf', 'fake PDF')
    monkeypatch.setattr('osintgpt.ingestion.pdf.extract_page_texts', lambda path: [''])
    monkeypatch.setattr('osintgpt.ingestion.pdf.render_page', lambda *args: b'page')
    calls = []
    def transcriber(image, number):
        calls.append(number)
        return 'Alpha funds Beta.'
    assert not index_project(project, Embedder(), transcriber=transcriber).failed
    assert calls == [1]
    build_graph(project, Generator())
    with graph_for(project) as graph:
        assert graph.edges()[0].provenance['matches'][0]['spans'][0]['page'] == 1
    assert verify_evidence(project).found == 1
    result = fetch_source(ToolContext(project, Embedder()), 'scan.pdf')
    assert result.payload['text'] == 'Alpha funds Beta.' and calls == [1]


def test_untranscribed_pages_are_visible_and_not_marked_completed(project, monkeypatch):
    source(project, 'scan.pdf', 'fake PDF')
    monkeypatch.setattr('osintgpt.ingestion.pdf.extract_page_texts', lambda path: [''])
    monkeypatch.setattr('osintgpt.ingestion.pdf.render_page', lambda *args: b'page')
    generator = Generator()
    report = build_graph(project, generator)
    assert report.status == 'complete_with_gaps' and report.calls == 0
    assert 'pages [1]' in report.failed[0].problem
    with graph_for(project) as graph:
        assert graph.state.documents() == {}
    result = fetch_source(ToolContext(project, Embedder()), 'scan.pdf')
    assert result.payload['provenance']['missing_pages'] == [1]


def test_corrected_transcription_marks_graph_stale_and_verifies_both_versions(project, monkeypatch):
    from osintgpt.ingestion.transcription import cache_path
    source(project, 'scan.pdf', 'fake PDF')
    monkeypatch.setattr('osintgpt.ingestion.pdf.extract_page_texts', lambda path: [''])
    monkeypatch.setattr('osintgpt.ingestion.pdf.render_page', lambda *args: b'page')
    transcript = cache_path(project.paths.extracts, b'page')
    transcript.write_text('Alpha funds Beta.')
    assert build_graph(project, Generator()).status == 'complete'
    transcript.write_text('Alpha funds Gamma.')
    assert saved_graph_status(project)['status'] == 'stale'
    result = verify_evidence(project).results[0]
    assert result.snapshot_status == 'found' and result.status == 'not_found'
    assert result.source_state == 'transcription_changed'


def test_record_citations_and_window_locations_are_specific(project):
    path = source(project, 'records.csv', 'id,body\none,Alpha funds Beta.\ntwo,Gamma funds Delta.\n')
    Corpus.load(project.paths.sources).register(path.name, FieldMapping(content=('body',), identity='id'))
    assert not index_project(project, Embedder()).failed
    with SQLiteVectorStore(project.paths.store) as store:
        chunks = store.chunks_for(path.name)
        assert [chunk.citation for chunk in chunks] == ['records.csv#one', 'records.csv#two']
    fetched = fetch_source(ToolContext(project, Embedder()), 'records.csv', offset=3, limit=2)
    assert [record['ref'] for record in fetched.payload['provenance']['records']] == ['records.csv#two']


@pytest.mark.parametrize('identity', ['https://site.test/a#section', '../other', 'a#b'])
def test_record_id_is_not_resolved_as_a_file_path(project, identity):
    path = source(project, 'records.csv', f'id,body\n{identity},Alpha funds Beta.\n')
    Corpus.load(project.paths.sources).register(path.name, FieldMapping(content=('body',), identity='id'))
    result = fetch_source(ToolContext(project, Embedder()), f'records.csv#{identity}')
    assert result.ok and result.payload['text'] == 'Alpha funds Beta.'
