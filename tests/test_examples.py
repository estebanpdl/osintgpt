"""Keep the published examples aligned with import and CLI surfaces."""

import ast
import importlib
import re
import runpy
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from typer.main import get_command

from osintgpt.cli import app
from osintgpt.config import ENV_VARS

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / 'examples'
PYTHON_EXAMPLES = sorted(EXAMPLES.rglob('*.py'))
LIBRARY_EXAMPLES = sorted((EXAMPLES / 'library').glob('*.py'))
BASH_BLOCK = re.compile(r'```bash\s*\n(.*?)```', re.DOTALL)
DEPRECATED = {'OpenAIEmbeddingGenerator', 'OpenAIGPT'}


def imported_symbols(path: Path):
    """Yield each osintgpt module import and imported symbol."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'osintgpt' or alias.name.startswith('osintgpt.'):
                    yield alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == 'osintgpt' or node.module.startswith('osintgpt.'):
                for alias in node.names:
                    yield node.module, alias.name


@pytest.mark.parametrize('path', PYTHON_EXAMPLES, ids=lambda path: path.name)
def test_python_examples_compile_and_import(path):
    source = path.read_text(encoding='utf-8')
    compile(source, str(path), 'exec')

    for module_name, symbol in imported_symbols(path):
        module = importlib.import_module(module_name)
        if symbol is not None:
            assert hasattr(module, symbol), f'{path}: no {module_name}.{symbol}'


def test_only_the_migration_example_names_deprecated_imports():
    for path in PYTHON_EXAMPLES:
        if path.name == 'migrating_from_0_1.py':
            continue
        imported = {
            symbol
            for _, symbol in imported_symbols(path)
            if symbol is not None
        }
        assert not imported & DEPRECATED, f'{path}: deprecated import'


def test_environment_template_matches_the_settings_source_of_truth():
    template = (EXAMPLES / 'config' / '.env.template').read_text(
        encoding='utf-8'
    )
    variables = {
        line.partition('=')[0]
        for line in template.splitlines()
        if line and not line.startswith('#') and '=' in line
    }

    assert variables == set(ENV_VARS.values())


@pytest.mark.parametrize('path', LIBRARY_EXAMPLES, ids=lambda path: path.name)
def test_library_examples_offer_help_without_provider_access(path, monkeypatch):
    output = StringIO()
    monkeypatch.setattr(sys, 'argv', [str(path), '--help'])

    with pytest.raises(SystemExit) as stopped:
        with redirect_stdout(output), redirect_stderr(output):
            runpy.run_path(str(path), run_name='__main__')

    assert stopped.value.code == 0, f'{path}: help exited non-zero'
    assert 'usage:' in output.getvalue().lower(), f'{path}: no help output'


def shell_commands(path: Path):
    """Yield complete commands from fenced bash blocks."""
    for block in BASH_BLOCK.findall(path.read_text(encoding='utf-8')):
        pending = ''
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            pending += line[:-1].rstrip() + ' ' if line.endswith('\\') else line
            if not line.endswith('\\'):
                yield pending.strip()
                pending = ''
        if pending:
            yield pending.strip()


def option_names(command):
    """Return every long option accepted by a Click command."""
    return {
        option
        for parameter in command.params
        for option in getattr(parameter, 'opts', ())
        if option.startswith('--')
    }


@pytest.mark.parametrize(
    'path', sorted((EXAMPLES / 'cli').glob('*.md')), ids=lambda path: path.name
)
def test_documented_cli_commands_and_flags_exist(path):
    root = get_command(app)
    root_options = option_names(root)

    for line in shell_commands(path):
        tokens = shlex.split(line)
        if not tokens or tokens[0] != 'osintgpt':
            continue

        assert len(tokens) > 1, f'{path}: incomplete command: {line}'
        name = tokens[1]
        assert name in root.commands, f'{path}: no command {name!r}'
        command = root.commands[name]
        argument_start = 2

        if hasattr(command, 'commands'):
            assert len(tokens) > 2, f'{path}: incomplete group command: {line}'
            child_name = tokens[2]
            assert child_name in command.commands, (
                f'{path}: no command {name} {child_name}'
            )
            command = command.commands[child_name]
            argument_start = 3

        valid_options = root_options | option_names(command)
        for token in tokens[argument_start:]:
            if not token.startswith('--'):
                continue
            option = token.partition('=')[0]
            assert option in valid_options, f'{path}: no option {option}: {line}'
