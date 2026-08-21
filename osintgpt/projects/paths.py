# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: paths.py
# Description: Where a project lives on disk. One place defines the layout, so
#   nothing else has to hardcode a filename.
# =================================================================================

# import submodules
from dataclasses import dataclass
from pathlib import Path

# file and directory names inside a project
CONFIG_FILE = 'project.toml'
STORE_FILE = 'store.sqlite'
SOURCES_FILE = 'sources.toml'
EXTRACTS_DIR = 'extracts'
CANON_DIR = 'canon'

# names under the osintgpt home
HOME_DIR = '.osintgpt'
PROJECTS_DIR = 'projects'

# resolve the default osintgpt home
def default_home() -> Path:
    '''
    The conventional osintgpt home, `~/.osintgpt`.

    Nothing in the library discovers this on its own — callers pass a home in.
    It is offered so the CLI and the docs agree on one default.

    Returns:
        Path: The default home directory. Not created.
    '''
    return Path.home() / HOME_DIR

# ProjectPaths class
@dataclass(frozen=True)
class ProjectPaths:
    '''
    The files and directories that make up one project, derived from its root.
    '''
    root: Path

    @classmethod
    def under_home(cls, home: Path, slug: str):
        '''
        Paths for a project stored in the conventional place under `home`.

        Args:
            home (Path): The osintgpt home.
            slug (str): The project's slug.

        Returns:
            ProjectPaths: Paths rooted at `<home>/projects/<slug>`.
        '''
        return cls(Path(home) / PROJECTS_DIR / slug)

    @property
    def config(self) -> Path:
        return self.root / CONFIG_FILE

    @property
    def store(self) -> Path:
        return self.root / STORE_FILE

    @property
    def sources(self) -> Path:
        return self.root / SOURCES_FILE

    @property
    def extracts(self) -> Path:
        return self.root / EXTRACTS_DIR

    @property
    def canon(self) -> Path:
        return self.root / CANON_DIR

    # create the directory structure
    def create_directories(self) -> None:
        '''
        Create the project root and its subdirectories, if absent.
        '''
        for directory in (self.root, self.extracts, self.canon):
            directory.mkdir(parents=True, exist_ok=True)
