# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: sources.py
# Description: What a project has been told to index. Registration is the only
#   gate: a file nobody registered is not corpus, wherever it sits on disk.
# =================================================================================

# import submodules
from dataclasses import dataclass, field, replace
from pathlib import Path

# type hints
from typing import Iterator, List, Optional, Union

# import osintgpt ingestion
from .documents import FieldMapping
from .loaders import READABLE_SUFFIXES

# import osintgpt projects
from osintgpt.projects.toml_io import read_toml, write_toml

# A folder registration should not be able to swallow a home directory by
# accident. Hit the ceiling and the source says so rather than indexing an
# arbitrary prefix of what it found.
MAX_FOLDER_FILES = 5_000

SOURCES_HEADER = '''\
# osintgpt corpus
#
# What this project has been told to index. Nothing is indexed because it
# happened to be on disk — a file or folder appears here or it is not corpus.
# Structured formats must name which of their fields carry content.

'''


# Source class
@dataclass(frozen=True)
class Source:
    '''
    One registered file or folder, and how to read it.
    '''
    # Relative to the project root where possible, so a project directory can
    # be moved or handed to someone else without rewriting its corpus.
    path: str
    # Field roles, required for structured formats and ignored for prose.
    mapping: FieldMapping = field(default_factory=FieldMapping)
    # Kept so a source can be described without re-reading the file.
    note: str = ''

    @property
    def is_folder_hint(self) -> bool:
        return not Path(self.path).suffix

    # what this source covers on disk
    def resolve(self, root: Union[str, Path]) -> List[Path]:
        '''
        Files this source covers, relative to a project root.

        A folder tracks what arrives beneath it, which is what lets a
        collection grow without re-registering every file. Unreadable
        extensions are skipped rather than reported: a folder of mixed
        material is normal.

        Args:
            root (Union[str, Path]): Project root, or the directory the \
                relative path is measured from.

        Returns:
            List[Path]: Existing files, sorted, capped at MAX_FOLDER_FILES.
        '''
        target = Path(root) / self.path

        if target.is_file():
            return [target]

        if not target.is_dir():
            return []

        found: List[Path] = []
        for candidate in sorted(target.rglob('*')):
            if len(found) >= MAX_FOLDER_FILES:
                break
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() in READABLE_SUFFIXES:
                found.append(candidate)

        return found

    def to_dict(self) -> dict:
        recorded = {'path': self.path}
        if self.note:
            recorded['note'] = self.note
        fields = self.mapping.to_dict()
        if fields:
            recorded['fields'] = fields

        return recorded

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            path=str(data.get('path', '')),
            mapping=FieldMapping.from_dict(data.get('fields')),
            note=str(data.get('note', ''))
        )


# Corpus class
@dataclass
class Corpus:
    '''
    A project's registered sources. Mutable and written to disk, because
    registering is something an operator does repeatedly.
    '''
    path: Path
    sources: List[Source] = field(default_factory=list)

    # read a project's corpus
    @classmethod
    def load(cls, path: Union[str, Path]):
        '''
        Args:
            path (Union[str, Path]): The project's sources file.

        Returns:
            Corpus: The registered sources, empty when the file is absent.
        '''
        document = read_toml(path)

        return cls(
            path=Path(path),
            sources=[Source.from_dict(row) for row in document.get('source', [])]
        )

    def save(self) -> None:
        write_toml(
            self.path,
            {'source': [source.to_dict() for source in self.sources]},
            header=SOURCES_HEADER
        )

    def _key_for(self, path: Union[str, Path]) -> str:
        root = self.path.parent.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        try:
            stored = resolved.relative_to(root)
        except ValueError:
            stored = resolved

        return stored.as_posix()

    # register a file or folder
    def register(
        self,
        path: Union[str, Path],
        mapping: Optional[FieldMapping] = None,
        note: str = ''
    ) -> Source:
        '''
        Add a source, or replace what was registered at the same path.

        Args:
            path (Union[str, Path]): File or folder to register. Project-local \
                paths are stored relative to the project root.
            mapping (FieldMapping, optional): Field roles for structured \
                formats.
            note (str): A line describing where the material came from.

        Returns:
            Source: The registered source.
        '''
        key = self._key_for(path)
        source = Source(
            path=key, mapping=mapping or FieldMapping(), note=note
        )
        self.sources = [
            registered
            for registered in self.sources
            if self._key_for(registered.path) != key
        ] + [source]
        self.save()

        return source

    # drop a source
    def unregister(self, path: Union[str, Path]) -> bool:
        '''
        Remove a source. The files stay; only the registration goes.

        Args:
            path (Union[str, Path]): The registered path.

        Returns:
            bool: True when something was removed.
        '''
        key = self._key_for(path)
        remaining = [
            source
            for source in self.sources
            if self._key_for(source.path) != key
        ]
        removed = len(remaining) != len(self.sources)
        if removed:
            self.sources = remaining
            self.save()

        return removed

    def find(self, path: Union[str, Path]) -> Optional[Source]:
        key = self._key_for(path)
        for source in self.sources:
            if self._key_for(source.path) == key:
                return source

        return None

    # every file the corpus covers
    def files(self, root: Union[str, Path]) -> List[Path]:
        '''
        Files across every source, deduplicated in registration order.

        Overlapping sources are ordinary — a folder and a file inside it — and
        the first registration wins, so a file is read once with one mapping.

        Args:
            root (Union[str, Path]): Project root.

        Returns:
            List[Path]: Existing files, in registration order.
        '''
        seen = set()
        found: List[Path] = []
        for source in self.sources:
            for candidate in source.resolve(root):
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(candidate)

        return found

    # which source governs a file
    def mapping_for(
        self, path: Union[str, Path], root: Union[str, Path]
    ) -> FieldMapping:
        '''
        The field roles that apply to one file.

        A file registered directly beats the folder containing it, so a
        spreadsheet inside a registered folder can name its own fields.

        Args:
            path (Union[str, Path]): File to look up.
            root (Union[str, Path]): Project root.

        Returns:
            FieldMapping: The mapping, empty when nothing set one.
        '''
        target = Path(path).resolve()

        direct = None
        folder = None
        for source in self.sources:
            covered = {p.resolve() for p in source.resolve(root)}
            if target not in covered:
                continue
            if Path(root, source.path).resolve() == target:
                direct = source
            elif folder is None:
                folder = source

        chosen = direct or folder

        return chosen.mapping if chosen else FieldMapping()

    def __iter__(self) -> Iterator[Source]:
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)
