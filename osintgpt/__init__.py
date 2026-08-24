# -*- coding: utf-8 -*-

# =================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: __init__.py
# Description: Package initialization file.
# =================================================

# import submodules
from importlib.metadata import PackageNotFoundError, version

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt projects
from osintgpt.projects import Project, ProjectSettings

# import osintgpt evaluation
from osintgpt.evaluation import (
    EvaluationReport,
    Question,
    evaluate,
    load_questions,
    save_questions
)

# import osintgpt indexing and search
from osintgpt.indexing import IndexReport, index_project
from osintgpt.search import search_across_projects, search_project

# define package-level variables and constants
# The version lives in pyproject.toml; reading it back from the installed
# metadata keeps one source of truth. Running from a source tree that was
# never installed has no metadata to read.
try:
    __version__ = version('osintgpt')
except PackageNotFoundError:
    __version__ = '0.0.0+unknown'

__name__ = 'osintgpt'
__all__ = [
    'EvaluationReport',
    'IndexReport',
    'Question',
    'Project',
    'ProjectSettings',
    'Settings',
    'evaluate',
    'index_project',
    'load_questions',
    'save_questions',
    'search_across_projects',
    'search_project'
]

# describition variables
__author__ = 'Esteban Ponce de Leon'
__doc__ = 'A Python OSINT tool using Large Language Models (LLMs)'
