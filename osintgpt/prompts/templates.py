# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: templates.py
# Description: The Jinja environment prompts are loaded through. Prompt text
#   lives in templates/ rather than inline, so the whole instruction surface
#   can be read and audited in one place.
# =================================================================================

# import submodules
from functools import lru_cache
from pathlib import Path

# type hints
from typing import List

# Jinja rather than str.format: a prompt that specifies JSON output contains
# literal braces, and .format would require doubling every one of them.
# {{ }} sidesteps that permanently.
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).resolve().parent / 'templates'
SUFFIX = '.md.j2'

# StrictUndefined: a template referencing a variable the caller forgot should
# fail at render rather than ship a sentence with a hole in it.
_environment = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined
)


# render a prompt
def prompt(name: str, **context) -> str:
    '''
    Render `templates/<name>.md.j2`.

    Args:
        name (str): Template name, without the suffix.
        **context: Variables the template expects.

    Raises:
        jinja2.TemplateNotFound: If no such template exists.
        jinja2.UndefinedError: If the template needs a variable not given.

    Returns:
        str: The rendered prompt, trailing whitespace stripped so a template \
            can end with a newline without it reaching the model.
    '''
    return _environment.get_template(f'{name}{SUFFIX}').render(**context).strip()


# render a prompt that takes no variables
@lru_cache(maxsize=None)
def static_prompt(name: str) -> str:
    '''
    `prompt()` for templates with no variables — cached, since the result
    cannot differ between calls.

    Args:
        name (str): Template name, without the suffix.

    Returns:
        str: The rendered prompt.
    '''
    return prompt(name)


# what a template expects
def variables_of(name: str) -> List[str]:
    '''
    The variables a template references.

    Asked of the template rather than tracked in a list, so a test can render
    every prompt without a catalogue that drifts from the files.

    Args:
        name (str): Template name, without the suffix.

    Returns:
        List[str]: Variable names, sorted.
    '''
    from jinja2 import meta

    source = _environment.loader.get_source(_environment, f'{name}{SUFFIX}')[0]

    return sorted(meta.find_undeclared_variables(_environment.parse(source)))


# every template that exists
def available() -> List[str]:
    '''
    Returns:
        List[str]: Template names, sorted. Used by the test that keeps the \
            index README honest.
    '''
    return sorted(
        path.name[:-len(SUFFIX)]
        for path in TEMPLATE_DIR.glob(f'*{SUFFIX}')
    )
