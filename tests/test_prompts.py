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
    topic_modeling_summarization,
    variables_of
)
from osintgpt.prompts.templates import SUFFIX, TEMPLATE_DIR

README = TEMPLATE_DIR / 'README.md'


def render(name):
    '''
    Render any template, with placeholders for whatever it asks for.

    Variables are discovered from the template rather than listed here, so a
    new prompt is covered by these checks the moment it exists.
    '''
    needed = variables_of(name)
    if not needed:
        return static_prompt(name)

    return prompt(name, **{
        # A list satisfies both a loop and a truth test; a string would break
        # the first, and a scalar the second.
        variable: [{'citation': 'a.md', 'text': 'placeholder'}]
        for variable in needed
    })


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
            assert len(render(name)) > 50, name

    def test_none_is_empty_or_whitespace(self):
        for name in available():
            assert render(name).strip(), name

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
            text = render(name)

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
                if len(node.value) <= 400 or node.value.count('\n') <= 5:
                    continue
                # Long and multi-line is not enough: a SQL schema is code a
                # database reads, not an instruction a model follows. What
                # makes a string a prompt is that it addresses a reader.
                lowered = node.value.lower()
                if any(
                    marker in lowered for marker in (
                        'you are', 'your task', 'you specialize', 'respond',
                        'analyze the', 'when the user', 'when presented'
                    )
                ):
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
        from pathlib import Path

        # read_toml carries the tomllib/tomli fallback: tomllib is 3.11+ and
        # 3.10 is a supported floor.
        from osintgpt.projects.toml_io import read_toml

        root = Path(__file__).resolve().parent.parent
        config = read_toml(root / 'pyproject.toml')

        setuptools = config.get('tool', {}).get('setuptools', {})

        assert setuptools.get('include-package-data') is True
