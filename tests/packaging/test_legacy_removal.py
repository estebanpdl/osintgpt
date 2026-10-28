"""Retired modules and exports must not leak back into the public package."""

import ast
import importlib
from pathlib import Path

import pytest

import osintgpt


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ('llms', 'embeddings', 'semantic_operations', 'databases')
MODULES = (
    'llms', 'llms.openai_gpt', 'llms.search', 'llms.completions', 'llms.conversation',
    'embeddings', 'embeddings.openai_embeddings', 'semantic_operations',
    'semantic_operations.operations', 'databases', 'databases.sql_manager',
)
EXPORTS = (
    'OpenAIGPT', 'OpenAIEmbeddingGenerator', 'SemanticOperations', 'SQLDatabaseManager',
    'SearchMixin', 'ConversationLogMixin', 'CompletionsMixin',
)


@pytest.mark.parametrize('name', MODULES)
def test_removed_module_cannot_be_imported(name):
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(f'osintgpt.{name}')
    assert caught.value.name == f'osintgpt.{name.split(".")[0]}'


@pytest.mark.parametrize('name', EXPORTS)
def test_removed_name_is_not_exported(name):
    assert name not in osintgpt.__all__
    assert not hasattr(osintgpt, name)
    assert not hasattr(importlib.import_module('osintgpt.llm'), name)


def test_surviving_source_never_imports_removed_packages():
    removed = {f'osintgpt.{name}' for name in PACKAGES}
    for directory in ('osintgpt', 'examples'):
        for path in (ROOT / directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or '']
                    if node.module == 'osintgpt':
                        modules.extend(f'osintgpt.{alias.name}' for alias in node.names)
                else:
                    continue
                assert not any(module == name or module.startswith(name + '.')
                               for module in modules for name in removed), path
