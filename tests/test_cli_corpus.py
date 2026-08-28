# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_cli_corpus.py
# Description: Corpus commands through Typer's public command surface.
# =================================================================================

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import corpus as cli_corpus
from osintgpt.indexing import DocumentResult, IndexReport
from osintgpt.ingestion import Corpus
from osintgpt.projects import Registry


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def records(tmp_path):
    path = tmp_path / 'registros.csv'
    path.write_text(
        'record_id,body\nr1,primer registro\nr2,segundo registro\n',
        encoding='utf-8'
    )

    return path


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def create_project(runner, home, name):
    result = invoke(runner, home, 'project', 'create', name)
    assert result.exit_code == 0

    return Registry.load(home).open(name.lower().replace(' ', '-'))


def test_add_refuses_to_guess_a_structured_mapping(runner, home, records):
    create_project(runner, home, 'Caso A')

    result = invoke(
        runner, home, 'add', str(records), '--project', 'caso-a'
    )

    assert result.exit_code != 0
    assert 'record_id' in result.output
    assert 'body' in result.output
    assert '--map content=<field>' in result.output


def test_add_requires_content_even_when_another_role_was_mapped(
    runner, home, records
):
    create_project(runner, home, 'Caso A')

    result = invoke(
        runner, home, 'add', str(records), '--project', 'caso-a',
        '--map', 'identity=record_id'
    )

    assert result.exit_code != 0
    assert 'content field mapping' in result.output


def test_add_with_a_mapping_registers_and_sources_shows_it(
    runner, home, records
):
    create_project(runner, home, 'Caso A')

    added = invoke(
        runner, home, 'add', str(records), '--project', 'caso-a',
        '--map', 'content=body', '--map', 'identity=record_id'
    )
    sources = invoke(runner, home, 'sources', '--project', 'caso-a')

    assert added.exit_code == 0
    assert sources.exit_code == 0
    assert records.name in sources.output
    assert 'body' in sources.output


def test_sources_json_parses_without_decoration(runner, home, records):
    create_project(runner, home, 'Caso A')
    invoke(
        runner, home, 'add', str(records), '--project', 'caso-a',
        '--map', 'content=body'
    )

    result = invoke(
        runner, home, 'sources', '--project', 'caso-a', '--json'
    )

    payload = json.loads(result.output)
    assert payload['project'] == 'caso-a'
    assert payload['sources'][0]['path'].endswith(records.name)
    assert '\x1b[' not in result.output


def test_remove_drops_registration_and_leaves_the_file(runner, home, records):
    project = create_project(runner, home, 'Caso A')
    invoke(
        runner, home, 'add', str(records), '--project', 'caso-a',
        '--map', 'content=body'
    )

    removed = invoke(
        runner, home, 'remove', str(records), '--project', 'caso-a'
    )

    assert removed.exit_code == 0
    assert records.exists()
    corpus = Corpus.load(project.paths.sources)
    assert len(corpus) == 1
    assert corpus.find('canon') is not None


def test_explicit_project_overrides_the_selection(runner, home, records):
    first = create_project(runner, home, 'Caso A')
    second = create_project(runner, home, 'Caso B')
    invoke(runner, home, 'project', 'use', 'caso-a')

    result = invoke(
        runner, home, 'add', str(records), '--project', 'caso-b',
        '--map', 'content=body'
    )

    assert result.exit_code == 0
    assert len(Corpus.load(first.paths.sources)) == 1
    assert len(Corpus.load(second.paths.sources)) == 2


def test_index_prints_progress_and_passes_maintenance_flags(
    runner, home, monkeypatch
):
    create_project(runner, home, 'Caso A')
    received = {}
    monkeypatch.setattr(
        cli_corpus, 'build_embedding_provider',
        lambda provider, settings: SimpleNamespace(model='modelo-local')
    )

    def index(project, embedder, **options):
        received.update(options)
        options['on_progress']('documento.md', 1, 1)

        return IndexReport(
            indexed=[DocumentResult('documento.md', chunks=2)],
            embedding_model=embedder.model
        )

    monkeypatch.setattr(cli_corpus, 'index_project', index)

    result = invoke(
        runner, home, 'index', '--project', 'caso-a', '--force',
        '--purge-other-models'
    )

    assert result.exit_code == 0
    assert '1/1 documento.md' in result.output
    assert received['force'] is True
    assert received['purge_other_models'] is True


def test_index_json_suppresses_progress_and_reports_failures(
    runner, home, monkeypatch
):
    create_project(runner, home, 'Caso A')
    monkeypatch.setattr(
        cli_corpus, 'build_embedding_provider',
        lambda provider, settings: SimpleNamespace(model='modelo-local')
    )

    def index(project, embedder, **options):
        assert options['on_progress'] is None

        return IndexReport(
            failed=[DocumentResult('roto.json', problem='invalid data')],
            embedding_model=embedder.model
        )

    monkeypatch.setattr(cli_corpus, 'index_project', index)

    result = invoke(
        runner, home, 'index', '--project', 'caso-a', '--json'
    )

    payload = json.loads(result.output)
    assert result.exit_code != 0
    assert payload['failed'][0]['ref'] == 'roto.json'
    assert '1/1' not in result.output
