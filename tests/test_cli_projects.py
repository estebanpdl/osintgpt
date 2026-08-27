# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_cli_projects.py
# Description: Project commands as operators and scripts invoke them.
# =================================================================================

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.projects import Registry


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


def test_create_then_list_shows_the_project(runner, home):
    created = invoke(runner, home, 'project', 'create', 'Caso Norte')
    listed = invoke(runner, home, 'project', 'list')

    assert created.exit_code == 0
    assert listed.exit_code == 0
    assert 'caso-norte' in listed.output
    assert 'Caso Norte' in listed.output


def test_showing_a_missing_project_exits_nonzero(runner, home):
    result = invoke(runner, home, 'project', 'show', 'missing')

    assert result.exit_code != 0
    assert 'not found' in result.output


def test_json_output_is_only_json(runner, home):
    result = invoke(
        runner, home, 'project', 'create', 'Caso JSON', '--json'
    )

    payload = json.loads(result.output)
    assert payload['slug'] == 'caso-json'
    assert 'Project created' not in result.output
    assert '\x1b[' not in result.output


def test_use_records_the_selected_project(runner, home):
    invoke(runner, home, 'project', 'create', 'Caso Elegido')

    selected = invoke(runner, home, 'project', 'use', 'caso-elegido')
    shown = invoke(runner, home, 'project', 'show', '--json')

    assert selected.exit_code == 0
    assert json.loads(shown.output)['slug'] == 'caso-elegido'


def test_a_command_without_a_project_explains_selection(runner, home):
    result = invoke(runner, home, 'sources')

    assert result.exit_code != 0
    assert 'project use <slug>' in result.output
    assert '--project <slug>' in result.output


def test_delete_requires_yes_and_never_prompts(runner, home):
    invoke(runner, home, 'project', 'create', 'Caso Conservado')

    result = invoke(runner, home, 'project', 'delete', 'caso-conservado')

    assert result.exit_code != 0
    assert '--yes' in result.output
    assert Registry.load(home).find('caso-conservado') is not None


def test_delete_removes_the_directory_and_registry_entry(runner, home):
    invoke(runner, home, 'project', 'create', 'Caso Borrado')
    project = Registry.load(home).open('caso-borrado')

    result = invoke(
        runner, home, 'project', 'delete', 'caso-borrado', '--yes'
    )

    assert result.exit_code == 0
    assert project.paths.root.exists() is False
    assert Registry.load(home).find('caso-borrado') is None
