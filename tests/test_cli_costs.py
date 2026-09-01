'''Cost reporting and hard per-run ceilings at the CLI boundary.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import corpus as cli_corpus
from osintgpt.cli import retrieval as cli_retrieval
from osintgpt.ingestion import Corpus
from osintgpt.llm import Usage
from osintgpt.projects import Registry
from osintgpt.vector_store import SQLiteVectorStore, StoredChunk


class RecordingEmbedder:
    model = 'text-embedding-3-small'

    def __init__(
        self, recorder, billable=True, counted=True, tokens=100
    ) -> None:
        self.recorder = recorder
        self.billable = billable
        self.counted = counted
        self.tokens = tokens

    def embed(self, texts):
        self.recorder.record(Usage(
            'stub', self.model, input_tokens=self.tokens,
            billable=self.billable, counted=self.counted
        ))

        return [[1.0, 0.0] for _ in texts]


class RecordingGenerator:
    model = 'gpt-4o'

    def __init__(self, recorder, billable=True, counted=True) -> None:
        self.recorder = recorder
        self.billable = billable
        self.counted = counted

    def generate(self, system, user):
        self.recorder.record(Usage(
            'stub', self.model, input_tokens=100, output_tokens=20,
            billable=self.billable, counted=self.counted
        ))

        return 'A grounded answer. [1]'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def searchable_project(runner, home):
    result = invoke(runner, home, 'project', 'create', 'Cost Case')
    assert result.exit_code == 0
    project = Registry.load(home).open('cost-case').with_settings(
        embedding_model='text-embedding-3-small',
        generation_model='gpt-4o',
        suggest_followups=False
    )
    project.save()
    with SQLiteVectorStore(project.paths.store) as store:
        chunk = StoredChunk(
            ref='evidence.md', sequence=0, text='Grounded evidence.',
            embedding_model='text-embedding-3-small'
        )
        store.upsert(chunk.ref, [chunk], [[1.0, 0.0]])

    return project


def test_one_run_records_embedding_and_generation_together(
    runner, home, monkeypatch
):
    searchable_project(runner, home)
    recorders = []

    def embedding_factory(provider, settings, model=None, recorder=None):
        recorders.append(recorder)
        return RecordingEmbedder(recorder)

    def generation_factory(provider, settings, model=None, recorder=None):
        recorders.append(recorder)
        return RecordingGenerator(recorder)

    monkeypatch.setattr(
        cli_retrieval, 'build_embedding_provider', embedding_factory
    )
    monkeypatch.setattr(
        cli_retrieval, 'build_generation_provider', generation_factory
    )

    result = invoke(
        runner, home, 'ask', 'What is grounded?', '--static',
        '--project', 'cost-case', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert recorders[0] is recorders[1]
    assert payload['usage']['calls'] == 2
    assert payload['usage']['billable_calls'] == 2
    assert set(payload['usage']['by_model']) == {
        'text-embedding-3-small', 'gpt-4o'
    }


def test_local_calls_do_not_print_a_zero_dollar_measurement(
    runner, home, monkeypatch
):
    searchable_project(runner, home)
    monkeypatch.setattr(
        cli_retrieval, 'build_embedding_provider',
        lambda provider, settings, model=None, recorder=None: (
            RecordingEmbedder(recorder, billable=False, counted=False)
        )
    )
    monkeypatch.setattr(
        cli_retrieval, 'build_generation_provider',
        lambda provider, settings, model=None, recorder=None: (
            RecordingGenerator(recorder, billable=False, counted=False)
        )
    )

    human = invoke(
        runner, home, 'ask', 'What is grounded?', '--static',
        '--project', 'cost-case'
    )
    machine = invoke(
        runner, home, 'ask', 'What is grounded?', '--static',
        '--project', 'cost-case', '--json'
    )

    assert human.exit_code == 0
    assert 'Usage:' not in human.output
    assert '$0.00' not in human.output
    usage = json.loads(machine.output)['usage']
    assert usage['billable_calls'] == 0
    assert usage['estimated_cost_usd'] is None


def test_missing_provider_usage_marks_the_estimate_incomplete(
    runner, home, monkeypatch
):
    searchable_project(runner, home)
    monkeypatch.setattr(
        cli_retrieval, 'build_embedding_provider',
        lambda provider, settings, model=None, recorder=None: (
            RecordingEmbedder(recorder, counted=False)
        )
    )

    result = invoke(
        runner, home, 'search', 'grounded',
        '--project', 'cost-case', '--json'
    )

    usage = json.loads(result.output)['usage']
    assert result.exit_code == 0
    assert usage['complete'] is False
    assert usage['uncounted_calls'] == 1
    assert usage['estimated_cost_usd'] is None


def indexing_project(runner, home, ceiling):
    result = invoke(runner, home, 'project', 'create', 'Index Cost')
    assert result.exit_code == 0
    project = Registry.load(home).open('index-cost').with_settings(
        embedding_model='text-embedding-3-small',
        cost_ceiling_usd=ceiling
    )
    project.save()
    material = project.paths.root / 'material'
    material.mkdir()
    (material / 'a.md').write_text('First document.', encoding='utf-8')
    (material / 'b.md').write_text('Second document.', encoding='utf-8')
    Corpus.load(project.paths.sources).register('material')

def stub_index_embedder(monkeypatch):
    monkeypatch.setattr(
        cli_corpus, 'build_embedding_provider',
        lambda provider, settings, model=None, recorder=None: (
            RecordingEmbedder(recorder)
        )
    )


def test_ceiling_stops_nonzero_and_the_next_run_resumes(
    runner, home, monkeypatch
):
    indexing_project(runner, home, ceiling=0.000003)
    stub_index_embedder(monkeypatch)

    stopped = invoke(
        runner, home, 'index', '--project', 'index-cost', '--json'
    )

    stopped_payload = json.loads(stopped.output)
    assert stopped.exit_code != 0
    assert '1 documents indexed; 1 remain' in stopped_payload['error']
    assert stopped_payload['indexed_documents'] == 1
    assert stopped_payload['remaining_documents'] == 1

    resumed = invoke(
        runner, home, 'index', '--project', 'index-cost', '--json'
    )
    resumed_payload = json.loads(resumed.output)

    assert resumed.exit_code == 0
    assert [row['ref'] for row in resumed_payload['indexed']] == [
        'material/b.md'
    ]
    assert resumed_payload['unchanged'] == 1


def test_no_ceiling_never_stops_the_run(
    runner, home, monkeypatch
):
    indexing_project(runner, home, ceiling=None)
    stub_index_embedder(monkeypatch)

    result = invoke(
        runner, home, 'index', '--project', 'index-cost', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert len(payload['indexed']) == 2
    assert payload['usage']['calls'] == 2
