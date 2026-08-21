# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: registry.py
# Description: An index of the projects under one home, so listing them and
#   searching across them does not walk the filesystem.
# =================================================================================

# import submodules
from dataclasses import dataclass
from pathlib import Path

# type hints
from typing import Iterator, List, Optional, Union

from .paths import PROJECTS_DIR, ProjectPaths
from .project import Project
from .toml_io import read_toml, write_toml

REGISTRY_FILE = 'registry.toml'

REGISTRY_HEADER = '''\
# osintgpt project registry
#
# An index, not a source of truth. Each project.toml owns its own settings; if
# this file disagrees with one, the project wins. Rebuild it at any time by
# rescanning the projects directory.

'''


# RegistryEntry class
@dataclass(frozen=True)
class RegistryEntry:
    '''
    What the index knows about one project without opening it.
    '''
    id: str
    slug: str
    name: str
    path: str
    embedding_model: str = ''

    # build from a project
    @classmethod
    def of(cls, project: Project):
        '''
        Args:
            project (Project): Project to index.

        Returns:
            RegistryEntry: The entry describing it.
        '''
        return cls(
            id=project.id,
            slug=project.slug,
            name=project.name,
            path=str(project.paths.root),
            embedding_model=project.settings.embedding_model
        )

    # load the project this entry points at
    def open(self) -> Project:
        '''
        Read the project itself, which is authoritative over this entry.

        Returns:
            Project: The project on disk.
        '''
        return Project.load(self.path)


# Registry class
@dataclass
class Registry:
    '''
    The projects under one home. Rebuildable from the directory it indexes, so
    a corrupted registry costs a command rather than a corpus.
    '''
    home: Path
    entries: List[RegistryEntry]

    # the registry file for a home
    @staticmethod
    def file_for(home: Union[str, Path]) -> Path:
        return Path(home) / REGISTRY_FILE

    # read the registry
    @classmethod
    def load(cls, home: Union[str, Path]):
        '''
        Read the index for a home.

        Args:
            home (Union[str, Path]): The osintgpt home.

        Returns:
            Registry: The index, empty when no registry file exists.
        '''
        document = read_toml(cls.file_for(home))
        entries = [
            RegistryEntry(
                id=row.get('id', ''),
                slug=row.get('slug', ''),
                name=row.get('name', ''),
                path=row.get('path', ''),
                embedding_model=row.get('embedding_model', '')
            )
            for row in document.get('project', [])
        ]

        return cls(home=Path(home), entries=entries)

    # rebuild by rescanning
    @classmethod
    def rebuild(cls, home: Union[str, Path]):
        '''
        Rescan the projects directory and replace the index with what is there.

        Projects created outside the conventional location are not found by a
        scan; re-register those explicitly.

        Args:
            home (Union[str, Path]): The osintgpt home.

        Returns:
            Registry: The rebuilt index, already written to disk.
        '''
        home = Path(home)
        entries = []
        projects_root = home / PROJECTS_DIR
        if projects_root.is_dir():
            for candidate in sorted(projects_root.iterdir()):
                if not ProjectPaths(candidate).config.is_file():
                    continue
                entries.append(RegistryEntry.of(Project.load(candidate)))

        registry = cls(home=home, entries=entries)
        registry.save()

        return registry

    # write the registry
    def save(self) -> None:
        '''
        Write the index to disk.
        '''
        write_toml(
            self.file_for(self.home),
            {'project': [
                {
                    'id': e.id, 'slug': e.slug, 'name': e.name,
                    'path': e.path, 'embedding_model': e.embedding_model
                }
                for e in self.entries
            ]},
            header=REGISTRY_HEADER
        )

    # index a project and persist
    def register(self, project: Project) -> None:
        '''
        Add or refresh a project's entry and write the index.

        Args:
            project (Project): Project to index.
        '''
        entry = RegistryEntry.of(project)
        self.entries = [
            e for e in self.entries if e.id != entry.id
        ] + [entry]
        self.save()

    # drop a project from the index
    def unregister(self, key: str) -> bool:
        '''
        Remove an entry by id or slug and write the index. Does not touch the
        project on disk.

        Args:
            key (str): Project id or slug.

        Returns:
            bool: True when an entry was removed.
        '''
        remaining = [e for e in self.entries if key not in (e.id, e.slug)]
        removed = len(remaining) != len(self.entries)
        if removed:
            self.entries = remaining
            self.save()

        return removed

    # find an entry without opening the project
    def find(self, key: str) -> Optional[RegistryEntry]:
        '''
        Args:
            key (str): Project id or slug.

        Returns:
            Optional[RegistryEntry]: The entry, or None.
        '''
        for entry in self.entries:
            if key in (entry.id, entry.slug):
                return entry

        return None

    # load a project by id or slug
    def open(self, key: str) -> Project:
        '''
        Resolve a project through the index and read it from disk, so callers
        get the authoritative settings rather than the indexed summary.

        Args:
            key (str): Project id or slug.

        Raises:
            KeyError: If nothing in the index matches.

        Returns:
            Project: The project.
        '''
        entry = self.find(key)
        if entry is None:
            raise KeyError(f'no project {key!r} in {self.file_for(self.home)}')

        return entry.open()

    def __iter__(self) -> Iterator[RegistryEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
