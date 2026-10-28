"""Build both distribution formats offline in disposable source directories."""

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
REMOVED = ('osintgpt/llms', 'osintgpt/embeddings', 'osintgpt/semantic_operations',
           'osintgpt/databases', 'tests/legacy')


def build(source, destination, kind):
    result = subprocess.run(
        [sys.executable, '-c',
         f'from setuptools.build_meta import build_{kind}; build_{kind}({str(destination)!r})'],
        cwd=source, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope='module')
def distributions(tmp_path_factory):
    scratch = tmp_path_factory.mktemp('distributions')
    source = scratch / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'CHANGELOG.md', 'MANIFEST.in'):
        shutil.copy2(ROOT / name, source / name)
    for name in ('osintgpt', 'tests', 'docs', 'examples'):
        shutil.copytree(ROOT / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    output = scratch / 'dist'
    output.mkdir()
    build(source, output, 'sdist')
    archive, = output.glob('*.tar.gz')
    # Rebuild the wheel from the sdist, so an incomplete sdist cannot pass.
    unpacked = scratch / 'unpacked'
    with tarfile.open(archive) as bundle:
        bundle.extractall(unpacked, filter='data')
    extracted, = unpacked.iterdir()
    build(extracted, output, 'wheel')
    wheel, = output.glob('*.whl')
    return archive, wheel


def assert_no_retired_paths(names):
    assert not [name for name in names
                if any(name == retired or name.startswith(retired + '/')
                       for retired in REMOVED)]


def test_sdist_contains_migration_docs_and_no_retired_code(distributions):
    archive, _ = distributions
    with tarfile.open(archive) as bundle:
        names = [name.partition('/')[2] for name in bundle.getnames()]
    assert_no_retired_paths(names)
    assert {'CHANGELOG.md', 'docs/migration-legacy-api.md'} <= set(names)
    assert {'examples/library/export_legacy_logs.py',
            'examples/library/migrating_from_0_1.py'} <= set(names)
    for path in (ROOT / 'osintgpt').rglob('*.py'):
        assert path.relative_to(ROOT).as_posix() in names


def test_wheel_contains_surviving_sources_and_assets_only(distributions):
    _, wheel = distributions
    with ZipFile(wheel) as bundle:
        names = bundle.namelist()
        assert_no_retired_paths(names)
        for path in (ROOT / 'osintgpt').rglob('*'):
            if path.is_file() and (path.suffix == '.py' or '/assets/' in path.as_posix()
                                   or '/prompts/templates/' in path.as_posix()):
                assert bundle.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()
