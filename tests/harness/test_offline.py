'''Behavioral checks for collection, subprocess isolation and retained output.'''

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

from ci.offline_environment import (
    SANDBOX_VARIABLE, UserStateBlocked, clear_configuration, protect_user_state,
)
from osintgpt.config import ENV_VARS


ROOT = Path(__file__).resolve().parents[2]


def test_all_application_settings_and_service_selectors_are_cleared():
    names = [*ENV_VARS.values(), 'GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS',
             'OSINTGPT_TEST_POSTGRES_DSN', 'OSINTGPT_TEST_QDRANT_HOST']
    environment = dict.fromkeys(names, 'synthetic-value')
    environment[SANDBOX_VARIABLE] = 'keep-sandbox'
    clear_configuration(environment)
    assert environment == {SANDBOX_VARIABLE: 'keep-sandbox'}


@pytest.mark.parametrize('method', ['connect', 'connect_ex', 'create_connection'])
def test_outbound_connections_are_refused(method):
    with socket.socket() as client, pytest.raises(RuntimeError, match='outbound connection'):
        if method == 'create_connection':
            socket.create_connection(('192.0.2.1', 443), timeout=0.1)
        else:
            getattr(client, method)(('192.0.2.1', 443))


def test_guard_refuses_explicit_user_state_paths_without_opening_them(tmp_path):
    protected = tmp_path / 'stand-in' / '.osintgpt'
    protect_user_state([protected])
    for path in (protected / 'credentials.toml', protected / '..' / '.osintgpt' / 'store.sqlite'):
        with pytest.raises(UserStateBlocked):
            path.open('rb')
    import sqlite3
    with pytest.raises(UserStateBlocked):
        sqlite3.connect(protected / 'store.sqlite')
    with pytest.raises(UserStateBlocked):
        sqlite3.connect((protected / 'store.sqlite').as_uri() + '?mode=ro', uri=True)


def test_explicit_configuration_outside_sandbox_is_refused(tmp_path):
    # A nonexistent stand-in is enough: the guard must run before filesystem I/O.
    with pytest.raises(UserStateBlocked):
        (tmp_path.parent / 'outside-fixture' / '.env').read_text()
    fixture = tmp_path / '.env'
    fixture.write_text('OPENAI_API_KEY=synthetic\n', encoding='utf-8')
    assert 'synthetic' in fixture.read_text(encoding='utf-8')


def test_child_python_inherits_guard_and_worktree_origin(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-inherited-key')
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
        import os, socket
        from pathlib import Path
        import osintgpt
        import sitecustomize
        assert 'OPENAI_API_KEY' not in os.environ
        assert Path.home() == Path(os.environ['OSINTGPT_TEST_SANDBOX']) / 'home'
        assert str(Path(osintgpt.__file__).resolve()).startswith(os.environ['EXPECTED_SOURCE'])
        try:
            socket.create_connection(('192.0.2.1', 443))
        except sitecustomize.NetworkBlocked:
            print('child guarded')
        else:
            raise AssertionError('outbound connection accepted')
    ''')], env=dict(os.environ, EXPECTED_SOURCE=str(ROOT)),
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'child guarded' in result.stdout


@pytest.fixture
def probe_project(tmp_path):
    project = tmp_path / 'probe'
    project.mkdir()
    shutil.copyfile(ROOT / 'tests' / 'conftest.py', project / 'conftest.py')
    (project / 'test_probe.py').write_text(textwrap.dedent('''
        import os
        import socket
        import warnings
        from pathlib import Path
        import pytest

        # These assertions run while the test module is being collected.
        assert not os.environ.get('OPENAI_API_KEY')
        assert not os.environ.get('OSINTGPT_TEST_POSTGRES_DSN')
        assert not os.environ.get('OSINTGPT_TEST_QDRANT_HOST')
        assert Path.home() == Path(os.environ['OSINTGPT_TEST_SANDBOX']) / 'home'
        with pytest.raises(RuntimeError, match='outbound connection'):
            socket.create_connection(('192.0.2.1', 443))

        def test_pass():
            warnings.warn('synthetic visible warning', UserWarning)

        def test_skip():
            pytest.skip('synthetic optional dependency')

        def test_failure():
            assert False, 'intentional capture probe'
    '''), encoding='utf-8')
    return project


def test_failure_capture_preserves_header_summary_warning_and_duration(probe_project, tmp_path):
    log = tmp_path / 'intentional-failure.log'
    env = dict(os.environ, OPENAI_API_KEY='synthetic-key',
               OSINTGPT_TEST_POSTGRES_DSN='synthetic-dsn', OSINTGPT_TEST_QDRANT_HOST='synthetic-host')
    # Exercise ordinary local pytest: no startup hook or inherited sandbox.
    # The copied conftest must establish isolation before test collection.
    env.pop(SANDBOX_VARIABLE, None)
    env.pop('OSINTGPT_TEST_PROTECTED_PATHS', None)
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'ci' / 'run_tests.py'), '--log', str(log), '--',
         '-c', str(ROOT / 'pyproject.toml'), str(probe_project), '-ra', '--durations=2'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    saved = log.read_text(encoding='utf-8')
    assert 'test session starts' in saved
    assert 'collected 3 items' in saved
    assert re.search(r'1 failed, 1 passed, 1 skipped, \d+ warnings? in [\d.]+s', saved)
    assert 'synthetic visible warning' in saved
    assert 'slowest 2 durations' in saved
    assert 'created: 2/2 workers' not in saved
    assert 'coverage:' not in saved.lower()
    receipt = json.loads(log.with_suffix('.json').read_text(encoding='utf-8'))
    assert receipt['exit_code'] == 1
    assert receipt['wall_seconds'] > 0


def test_plain_and_evidence_profiles_collect_the_same_cases(probe_project):
    base = [sys.executable, '-m', 'pytest', '-c', str(ROOT / 'pyproject.toml'),
            str(probe_project), '--collect-only']
    outputs = []
    for profile in ([], ['-ra', '--durations=20']):
        result = subprocess.run(base + profile, cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append([line.strip() for line in result.stdout.splitlines() if '<Function ' in line])
    assert outputs[0] == outputs[1] == ['<Function test_pass>', '<Function test_skip>', '<Function test_failure>']


def test_unknown_markers_remain_collection_errors(probe_project):
    (probe_project / 'test_probe.py').write_text(
        'import pytest\n@pytest.mark.unknown_service\ndef test_no_run(): pass\n', encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-c', str(ROOT / 'pyproject.toml'),
         str(probe_project), '--collect-only'],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2
    assert 'unknown_service' in result.stdout
    assert 'not found in `markers` configuration option' in result.stdout
