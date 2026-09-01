'''Retrieval evaluation through the command-line interface.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import evaluate as cli_evaluate
from osintgpt.evaluation import Question, save_questions
from osintgpt.llm import Usage
from osintgpt.projects import Registry
from osintgpt.vector_store import SQLiteVectorStore, StoredChunk


class Embedder:
    model = 'test-model'

    def __init__(self, recorder=None):
        self.recorder = recorder

    def embed(self, texts):
        if self.recorder is not None:
            self.recorder.record(Usage(
                'stub', self.model, input_tokens=len(texts)
            ))

        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def project_with_chunks(runner, home):
    created = invoke(runner, home, 'project', 'create', 'Case Evaluate')
    assert created.exit_code == 0
    project = Registry.load(home).open('case-evaluate').with_settings(
        embedding_model='test-model'
    )
    project.save()
    store = SQLiteVectorStore(project.paths.store)
    chunks = [
        (
            StoredChunk(
                ref='a-exact.md', sequence=0,
                text='The record names literal SIG-77.',
                embedding_model='test-model'
            ),
            [0.0, 1.0]
        ),
        (
            StoredChunk(
                ref='z-semantic.md', sequence=0,
                text='A conceptually similar record.',
                embedding_model='test-model'
            ),
            [1.0, 0.0]
        )
    ]
    for chunk, vector in chunks:
        store.upsert(chunk.ref, [chunk], [vector])
    store.close()

    return project


def questions_file(tmp_path, questions):
    path = tmp_path / 'questions.toml'
    save_questions(path, questions)

    return path


def stub_embedder(monkeypatch):
    monkeypatch.setattr(
        cli_evaluate,
        'build_embedding_provider',
        lambda provider, settings, model=None, recorder=None: (
            Embedder(recorder)
        )
    )


def test_report_has_expected_scores_and_names_the_method_and_model(
    runner, home, tmp_path, monkeypatch
):
    project_with_chunks(runner, home)
    question_set = questions_file(tmp_path, [
        Question('concept', ['z-semantic.md']),
        Question('concept', ['a-exact.md'])
    ])
    stub_embedder(monkeypatch)

    result = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--top-k', '1', '--json'
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['retrieval'] == 'semantic'
    assert payload['embedding_model'] == 'test-model'
    assert payload['hit_rate'] == 0.5
    assert payload['mean_reciprocal_rank'] == 0.5
    assert payload['found'] == 1
    assert payload['scored'] == 2
    assert payload['usage']['calls'] == 2


def test_unknown_ref_is_unscorable_instead_of_a_miss(
    runner, home, tmp_path, monkeypatch
):
    project_with_chunks(runner, home)
    question_set = questions_file(
        tmp_path, [Question('concept', ['missing.md'])]
    )
    stub_embedder(monkeypatch)

    result = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload['scored'] == 0
    assert payload['misses'] == []
    assert len(payload['unscorable']) == 1
    assert 'not in the store' in payload['unscorable'][0]


def test_hybrid_finds_an_exact_only_result_below_semantic_top_k(
    runner, home, tmp_path, monkeypatch
):
    project_with_chunks(runner, home)
    question_set = questions_file(tmp_path, [
        Question('concept', ['a-exact.md'], terms=['SIG-77'])
    ])
    stub_embedder(monkeypatch)

    semantic = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--top-k', '1', '--json'
    )
    hybrid = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--top-k', '1',
        '--retrieval', 'hybrid', '--json'
    )

    assert json.loads(semantic.output)['found'] == 0
    hybrid_payload = json.loads(hybrid.output)
    assert hybrid.exit_code == 0
    assert hybrid_payload['found'] == 1
    assert hybrid_payload['retrieval'] == 'hybrid'


def test_human_report_names_scores_misses_and_unscorable(
    runner, home, tmp_path, monkeypatch
):
    project_with_chunks(runner, home)
    question_set = questions_file(tmp_path, [
        Question('missed question', ['a-exact.md']),
        Question('bad fixture', ['missing.md'])
    ])
    stub_embedder(monkeypatch)

    result = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--top-k', '1'
    )

    assert result.exit_code == 0
    assert 'semantic' in result.output
    assert 'test-model' in result.output
    assert 'Hit rate' in result.output
    assert 'MRR' in result.output
    assert 'Misses' in result.output
    assert 'missed question' in result.output
    assert 'Unscorable' in result.output
    assert 'bad fixture' in result.output


def test_evaluation_opens_the_store_once(
    runner, home, tmp_path, monkeypatch
):
    project = project_with_chunks(runner, home)
    question_set = questions_file(
        tmp_path, [Question('concept', ['z-semantic.md'])]
    )
    stub_embedder(monkeypatch)
    store = SQLiteVectorStore(project.paths.store)
    opened = []

    def factory(selected, config):
        opened.append(selected.slug)
        return store

    monkeypatch.setattr(cli_evaluate, 'store_for', factory)

    result = invoke(
        runner, home, 'evaluate', str(question_set),
        '--project', 'case-evaluate', '--json'
    )

    assert result.exit_code == 0
    assert opened == ['case-evaluate']


def test_questions_path_is_required(runner, home):
    project_with_chunks(runner, home)

    result = invoke(
        runner, home, 'evaluate', '--project', 'case-evaluate'
    )

    assert result.exit_code != 0
    assert 'QUESTIONS' in result.output
