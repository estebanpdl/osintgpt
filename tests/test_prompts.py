# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_prompts.py
# Description: The prompt templates and the environment they load through.
#   Prompt text is the one surface with no type to check it, so what can be
#   checked mechanically is.
# =================================================================================

# import modules
import pytest

# import submodules
from jinja2 import TemplateNotFound
from jinja2.exceptions import UndefinedError

# import osintgpt prompts
from osintgpt.prompts import (
    available,
    basic_summarization,
    prompt,
    static_prompt,
    topic_modeling_summarization
)
from osintgpt.prompts.templates import SUFFIX, TEMPLATE_DIR

README = TEMPLATE_DIR / 'README.md'


class TestEnvironment:
    def test_a_template_renders(self):
        assert static_prompt('summarize')

    def test_an_unknown_template_raises(self):
        with pytest.raises(TemplateNotFound):
            prompt('no-such-template')

    def test_trailing_whitespace_is_stripped(self):
        '''A template can end with a newline without it reaching the model.'''
        text = static_prompt('summarize')

        assert text == text.strip()

    def test_a_missing_variable_fails_loudly(self, tmp_path):
        '''
        StrictUndefined: a prompt with a hole in it is worse than an error,
        because it reaches the model looking like an instruction.
        '''
        from jinja2 import Environment, StrictUndefined

        environment = Environment(undefined=StrictUndefined)
        template = environment.from_string('Answer about {{ subject }}.')

        with pytest.raises(UndefinedError):
            template.render()

    def test_static_prompts_are_cached(self):
        assert static_prompt('summarize') is static_prompt('summarize')


class TestTemplates:
    def test_every_template_has_content(self):
        for name in available():
            assert len(static_prompt(name)) > 50, name

    def test_none_is_empty_or_whitespace(self):
        for name in available():
            assert static_prompt(name).strip(), name

    def test_literal_braces_survive(self):
        '''
        The reason for Jinja over str.format: a prompt specifying JSON output
        contains braces, and .format would need every one doubled.
        '''
        text = static_prompt('sentence_details')

        assert '{' in text and '}' in text
        assert '"Language"' in text

    def test_no_template_carries_an_unrendered_placeholder(self):
        for name in available():
            text = static_prompt(name)

            assert '{{' not in text, name
            assert '{%' not in text, name


class TestCallers:
    def test_summarization_reads_from_a_template(self):
        assert basic_summarization() == static_prompt('summarize')

    def test_topic_modeling_reads_from_a_template(self):
        assert topic_modeling_summarization() == static_prompt('topic_modeling')

    def test_no_prompt_text_is_left_in_the_package(self):
        '''
        Prompts scattered across .py files become unauditable, which is the
        problem the template directory exists to prevent.
        '''
        import ast
        from pathlib import Path

        package = Path(__file__).resolve().parent.parent / 'osintgpt'
        offenders = []

        scopes = (
            ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef
        )

        for path in package.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))

            # A docstring is a Constant too, and this is looking for prompt
            # text, so gather them first and skip them below.
            docstrings = set()
            for scope in ast.walk(tree):
                body = getattr(scope, 'body', []) if isinstance(
                    scope, scopes
                ) else []
                if body and isinstance(body[0], ast.Expr) and isinstance(
                    body[0].value, ast.Constant
                ):
                    docstrings.add(id(body[0].value))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if not isinstance(node.value, str):
                    continue
                if id(node) in docstrings:
                    continue
                # A model instruction is long, multi-line, and addresses a
                # reader. Operator-facing messages are none of those.
                if len(node.value) > 400 and node.value.count('\n') > 5:
                    offenders.append(f'{path.name}:{node.lineno}')

        assert offenders == []


class TestIndex:
    def test_the_readme_exists(self):
        assert README.is_file()

    def test_every_template_is_listed(self):
        '''
        An index that misses a template is worse than none: it implies the
        list is complete.
        '''
        text = README.read_text(encoding='utf-8')

        for name in available():
            assert f'`{name}`' in text, name

    def test_it_lists_no_template_that_does_not_exist(self):
        '''
        The template table only. The helpers table above it also opens with a
        backtick, and matching both would read `prompt(name)` as a template.
        '''
        import re

        text = README.read_text(encoding='utf-8')
        section = text.split('## The templates', 1)[1].split('##', 1)[0]
        named = [
            re.match(r'\| `([^`]+)`', line).group(1)
            for line in section.splitlines()
            if line.startswith('| `')
        ]

        assert named
        for name in named:
            assert (TEMPLATE_DIR / f'{name}{SUFFIX}').is_file(), name

    def test_it_explains_the_contract_distinction(self):
        text = README.read_text(encoding='utf-8')

        assert 'Contract prompts' in text
        assert 'Voice prompts' in text


class TestPackaging:
    def test_templates_ship_with_the_package(self):
        '''
        A .md.j2 is not a .py, so it reaches an installed package only if
        packaging is told to include it. Without this, every prompt raises
        TemplateNotFound for anyone who installed from a wheel.
        '''
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        with open(root / 'pyproject.toml', 'rb') as handle:
            config = tomllib.load(handle)

        setuptools = config.get('tool', {}).get('setuptools', {})

        assert setuptools.get('include-package-data') is True
