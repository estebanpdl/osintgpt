'''Offline-first project diagnostics through the CLI.'''

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import doctor as cli_doctor
from osintgpt.ingestion import Corpus
from osintgpt.projects import Registry
from osintgpt.vector_store import SQLiteVectorStore, StoredChunk


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def create_project(runner, home):
    result = invoke(runner, home, 'project', 'create', 'Case Doctor')
    assert result.exit_code == 0

    return Registry.load(home).open('case-doctor')


def test_doctor_handles_a_project_without_a_store(runner, home):
    create_project(runner, home)

    result = invoke(
        runner, home, 'doctor', '--project', 'case-doctor', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload['storage']['exists'] is False
    assert payload['storage']['chunks'] == 0
    assert payload['storage']['documents'] == 0


def test_doctor_reports_embedding_model_mismatch(runner, home):
    project = create_project(runner, home).with_settings(
        embedding_model='configured-model'
    )
    project.save()
    store = SQLiteVectorStore(project.paths.store)
    chunk = StoredChunk(
        ref='evidence.md', sequence=0, text='evidence',
        embedding_model='stored-model'
    )
    store.upsert(chunk.ref, [chunk], [[1.0, 0.0]])
    store.close()

    result = invoke(
        runner, home, 'doctor', '--project', 'case-doctor', '--json'
    )

    assert result.exit_code == 0
    assert 'model mismatch' in result.output
    assert 'configured-model' in result.output
    assert 'stored-model' in result.output


def test_doctor_reports_registered_source_coverage(runner, home):
    project = create_project(runner, home)
    evidence = project.paths.root / 'evidence'
    evidence.mkdir()
    (evidence / 'one.txt').write_text('one', encoding='utf-8')
    (evidence / 'two.md').write_text('two', encoding='utf-8')
    Corpus.load(project.paths.sources).register('evidence')

    result = invoke(
        runner, home, 'doctor', '--project', 'case-doctor', '--json'
    )

    payload = json.loads(result.output)
    assert payload['sources'] == [
        {'path': 'evidence', 'files': 2, 'problem': None}
    ]


def test_doctor_makes_no_provider_call_by_default(
    runner, home, monkeypatch
):
    create_project(runner, home)

    def forbidden(*args, **kwargs):
        raise AssertionError('network-facing factory was called')

    monkeypatch.setattr(cli_doctor, 'build_embedding_provider', forbidden)
    monkeypatch.setattr(cli_doctor, 'build_generation_provider', forbidden)

    result = invoke(runner, home, 'doctor', '--project', 'case-doctor')

    assert result.exit_code == 0
    assert 'Doctor: Case Doctor' in result.output


def test_doctor_strict_can_fail_on_findings(runner, home):
    create_project(runner, home)

    result = invoke(
        runner, home, 'doctor', '--project', 'case-doctor', '--strict'
    )

    assert result.exit_code != 0
    assert 'store file does not exist' in result.output


def test_doctor_requires_a_selected_project(runner, home):
    result = invoke(runner, home, 'doctor')

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output
