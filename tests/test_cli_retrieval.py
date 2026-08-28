'''Grounded answer and semantic search commands through Typer.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import retrieval as cli_retrieval
from osintgpt.projects import Registry
from osintgpt.vector_store import SQLiteVectorStore, StoredChunk


class Embedder:
    model = 'test-model'

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class Generator:
    model = 'test-chat'

    def generate(self, system, user):
        return 'The project says alpha. [1]'


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def project_with_chunks(runner, home, chunks=3):
    created = invoke(runner, home, 'project', 'create', 'Case Search')
    assert created.exit_code == 0
    project = Registry.load(home).open('case-search').with_settings(
        embedding_model='test-model', generation_model='test-chat'
    )
    project.save()
    store = SQLiteVectorStore(project.paths.store)
    for index in range(chunks):
        chunk = StoredChunk(
            ref=f'doc-{index}.md', sequence=0,
            text=f'alpha passage {index}', embedding_model='test-model',
            path=f'Section {index}'
        )
        store.upsert(chunk.ref, [chunk], [[1.0, index / 10]])
    store.close()

    return project


def stub_providers(monkeypatch, generator_factory=None):
    monkeypatch.setattr(
        cli_retrieval, 'build_embedding_provider',
        lambda provider, settings, model=None: Embedder()
    )
    monkeypatch.setattr(
        cli_retrieval, 'build_generation_provider',
        generator_factory or (
            lambda provider, settings, model=None: Generator()
        )
    )


def test_ask_prints_answer_and_sources(runner, home, monkeypatch):
    project_with_chunks(runner, home)
    stub_providers(monkeypatch)

    result = invoke(
        runner, home, 'ask', 'What is alpha?', '--project', 'case-search'
    )

    assert result.exit_code == 0
    assert 'The project says alpha.' in result.output
    assert 'Sources' in result.output
    assert 'doc-0.md' in result.output


def test_ask_json_carries_the_answer_and_the_trace(
    runner, home, monkeypatch
):
    '''
    The trace is not behind a flag in JSON. A script collecting answers should
    be collecting the reasoning that produced them.
    '''
    project_with_chunks(runner, home)
    stub_providers(monkeypatch)

    result = invoke(
        runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
        '--json'
    )

    payload = json.loads(result.output)
    assert set(payload) == {
        'answer', 'sources', 'followups', 'degraded', 'trace'
    }
    assert set(payload['trace']) == {'rounds', 'calls', 'narration', 'reading'}
    assert '\x1b[' not in result.output


def test_static_json_keeps_the_passage_shape(runner, home, monkeypatch):
    '''
    --static is the single-retrieval path, and it reports what it retrieved.
    '''
    project_with_chunks(runner, home)
    stub_providers(monkeypatch)

    result = invoke(
        runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
        '--passages', '2', '--static', '--json'
    )

    payload = json.loads(result.output)
    assert set(payload) == {'answer', 'passages', 'followups'}
    assert len(payload['passages']) == 2
    assert set(payload['passages'][0]) == {'text', 'score', 'citation'}


def test_ask_unindexed_exits_zero_without_building_generator(
    runner, home, monkeypatch
):
    invoke(runner, home, 'project', 'create', 'Empty Case')

    def forbidden(*args, **kwargs):
        raise AssertionError('generator should not be built')

    stub_providers(monkeypatch, forbidden)
    result = invoke(
        runner, home, 'ask', 'Anything?', '--project', 'empty-case'
    )

    assert result.exit_code == 0
    assert 'Nothing in this project matches' in result.output


def test_search_returns_ranked_hits_and_honours_top_k(
    runner, home, monkeypatch
):
    project_with_chunks(runner, home)
    stub_providers(monkeypatch)

    result = invoke(
        runner, home, 'search', 'alpha', '--project', 'case-search',
        '--top-k', '2', '--json'
    )

    payload = json.loads(result.output)
    assert [row['rank'] for row in payload['results']] == [1, 2]
    assert len(payload['results']) == 2
    assert payload['results'][0]['score'] >= payload['results'][1]['score']


@pytest.mark.parametrize('command', [('ask', 'question'), ('search', 'query')])
def test_retrieval_requires_a_selected_project(runner, home, command):
    result = invoke(runner, home, *command)

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output
