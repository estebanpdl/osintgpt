# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_packaging.py
# Description: Packaging invariants — one source for the version, and metadata
#   that matches what the package actually supports.
# =================================================================================

# import modules
import pytest

# tomllib landed in 3.11; 3.10 is still a supported floor.
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# import submodules
from pathlib import Path

# import osintgpt
import osintgpt

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='module')
def pyproject():
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as handle:
        return tomllib.load(handle)


class TestSingleSource:
    def test_setup_py_is_gone(self):
        assert not (REPO_ROOT / 'setup.py').exists()

    def test_requirements_txt_is_gone(self):
        '''Dependencies live in pyproject; a second list would drift.'''
        assert not (REPO_ROOT / 'requirements.txt').exists()

    def test_no_hardcoded_version_literal(self):
        source = (REPO_ROOT / 'osintgpt' / '__init__.py').read_text(
            encoding='utf-8'
        )

        # A fallback literal is fine; a release number here would be a second
        # source of truth alongside pyproject.
        assert "__version__ = '0.0.1'" not in source
        assert 'importlib.metadata' in source

    def test_version_is_readable(self):
        assert osintgpt.__version__

    def test_version_matches_the_project_metadata(self, pyproject):
        # An editable install writes its metadata once, so a local version bump
        # needs a reinstall before this agrees.
        assert osintgpt.__version__ == pyproject['project']['version']


class TestMetadata:
    def test_requires_a_modern_python(self, pyproject):
        assert pyproject['project']['requires-python'] == '>=3.10'

    def test_classifiers_do_not_advertise_unsupported_versions(self, pyproject):
        unsupported = {'3.7', '3.8', '3.9'}
        advertised = {
            classifier.rsplit(' ', 1)[-1]
            for classifier in pyproject['project']['classifiers']
            if classifier.startswith('Programming Language :: Python :: 3.')
        }

        assert not advertised & unsupported

    def test_declares_its_runtime_dependencies(self, pyproject):
        names = {
            dependency.split('>')[0].split('=')[0].split('<')[0].strip()
            for dependency in pyproject['project']['dependencies']
        }

        assert {
            'openai', 'python-dotenv', 'rich', 'tiktoken', 'typer'
        } <= names

    def test_installs_the_console_script(self, pyproject):
        assert pyproject['project']['scripts']['osintgpt'] == (
            'osintgpt.cli.main:main'
        )

    def test_pins_openai_below_the_next_major(self, pyproject):
        openai = next(
            dependency
            for dependency in pyproject['project']['dependencies']
            if dependency.startswith('openai')
        )

        assert '>=1.0' in openai and '<3' in openai


class TestPublicSurface:
    def test_settings_is_exported(self):
        assert 'Settings' in osintgpt.__all__
        assert osintgpt.Settings is not None
