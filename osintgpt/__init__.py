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

# import osintgpt answering
from osintgpt.answering import Answer, answer_question

# import osintgpt indexing, lexical search and semantic search
from osintgpt.indexing import IndexReport, index_project
from osintgpt.lexical import derive_search_terms, lexical_search
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
    'Answer',
    'EvaluationReport',
    'IndexReport',
    'Question',
    'Project',
    'ProjectSettings',
    'Settings',
    'answer_question',
    'derive_search_terms',
    'evaluate',
    'index_project',
    'lexical_search',
    'load_questions',
    'save_questions',
    'search_across_projects',
    'search_project'
]

# describition variables
__author__ = 'Esteban Ponce de Leon'
__doc__ = 'A Python OSINT tool using Large Language Models (LLMs)'
