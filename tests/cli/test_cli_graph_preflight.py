'''Exercise graph planning and retry controls through the public CLI.'''

import json
import pytest

from osintgpt.ingestion import Corpus
from test_cli_graph import clear_credentials, create_project, home, invoke, runner


def test_cli_dry_run_is_keyless_and_does_not_construct_provider(runner, home, monkeypatch):
    project = create_project(runner, home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'source.md').write_text('Alpha supports Beta.')
    Corpus.load(project.paths.sources).register('source.md')
    clear_credentials(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must not construct a generation provider')
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)
    result = invoke(runner, home, 'graph', 'build', '--project', project.slug, '--dry-run', '--rpm', '12', '--tpm', '50000',
                    '--retry-attempts', '2', '--retry-seconds', '60', '--json')
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data['dry_run'] and data['counts']['windows_pending'] == 1
    assert data['request_limits']['rates']['rpm'] == 12
    assert data['request_limits']['retry']['max_attempts'] == 2
    assert data['counts']['maximum_http_attempts'] is None  # useful follow-up work has no fixed request maximum
    assert not (project.paths.root / 'graph.sqlite').exists()
    assert not (home / 'generation-rates.sqlite').exists()


def test_graph_help_describes_preflight_and_retry_flags(runner, home):
    result = invoke(runner, home, 'graph', 'build', '--help')
    assert result.exit_code == 0
    for flag in ('--dry-run', '--rpm', '--tpm', '--retry-attempts', '--retry-seconds'):
        assert flag in result.output
    assert '--output-tokens' not in result.output


def test_retired_output_option_fails_before_project_or_provider_work(runner, home, monkeypatch):
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', lambda *a, **k: pytest.fail('No provider allowed'))
    result = invoke(runner, home, 'graph', 'build', '--output-tokens', '4096', '--json')
    assert result.exit_code != 0
    assert '--output-tokens was removed' in result.output


def test_retired_call_ceiling_and_diagnostics_are_keyless(runner, home, monkeypatch):
    project = create_project(runner, home)
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', lambda *a, **k: pytest.fail('No provider allowed'))
    result = invoke(runner, home, 'graph', 'build', '--max-record-calls', '500', '--json')
    assert result.exit_code != 0 and '--max-record-calls is retired' in result.output
    result = invoke(runner, home, 'graph', 'diagnostics', '--project', project.slug, '--json')
    assert result.exit_code == 0 and json.loads(result.stdout)['results'] == []
    assert not (project.paths.root / 'graph.sqlite').exists()
    from osintgpt.cli import app
    help_result = runner.invoke(app, ['graph', 'build', '--help'], terminal_width=180, env={'COLUMNS': '180'})
    assert '--source-window-bytes' in help_result.output and '--max-record-calls' not in help_result.output
