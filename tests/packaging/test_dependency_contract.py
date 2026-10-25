'''Direct imports must be declared by their owner, including optional backends.'''

import ast
from pathlib import Path
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
CORE = {
    'PIL': 'pillow', 'anthropic': 'anthropic', 'docx': 'python-docx',
    'dotenv': 'python-dotenv', 'google': 'google-genai', 'httpx': 'httpx',
    'jinja2': 'jinja2', 'markitdown': 'markitdown', 'numpy': 'numpy',
    'openai': 'openai', 'openpyxl': 'openpyxl', 'pandas': 'pandas',
    'pypdf': 'pypdf', 'pypdfium2': 'pypdfium2', 'rich': 'rich',
    'streamlit': 'streamlit', 'tiktoken': 'tiktoken', 'tomli': 'tomli',
    'tomli_w': 'tomli-w', 'typer': 'typer',
}
EXTRAS = {
    'psycopg': ('postgres', 'psycopg'), 'pgvector': ('postgres', 'pgvector'),
    'qdrant_client': ('qdrant', 'qdrant-client'),
    'sentence_transformers': ('local', 'sentence-transformers'),
}


def dependency_names(requirements):
    return {re.split(r'[\[<>=!~; ]', item, maxsplit=1)[0].lower() for item in requirements}


def test_every_third_party_import_has_an_explicit_owner(monkeypatch):
    # Exercise the older interpreter's stdlib inventory on newer Python too.
    monkeypatch.setattr(sys, 'stdlib_module_names', sys.stdlib_module_names - {'tomllib'})
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    imported = set()
    for path in (ROOT / 'osintgpt').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                imported.add(node.module.split('.')[0])
    # tomllib is guarded by a tomli fallback and is not in Python 3.10's
    # stdlib_module_names, even though it is a standard-library import.
    third_party = imported - sys.stdlib_module_names - {'osintgpt', 'tomllib'}
    # httpx2 is conditionally inspected only when an SDK installs it. The
    # classifier must also load with older SDKs that use httpx exclusively.
    assert third_party <= CORE.keys() | EXTRAS.keys() | {'httpx2'}
    assert set(CORE.values()) <= dependency_names(project['dependencies'])
    for extra, distribution in EXTRAS.values():
        assert distribution in dependency_names(project['optional-dependencies'][extra])
    assert not {distribution for _, distribution in EXTRAS.values()} & dependency_names(project['dependencies'])


def test_package_syntax_parses_on_the_supported_python_floor():
    # Grammar checking is not execution on Python 3.10; that remains a CI job.
    for path in (ROOT / 'osintgpt').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 10))
