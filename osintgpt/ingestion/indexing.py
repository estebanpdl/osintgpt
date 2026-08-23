# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: indexing.py
# Description: Deciding what has changed since last time. An unchanged document
#   costs a hash and nothing else; embedding is the expensive part, so the
#   whole point is to not do it twice for the same bytes.
# =================================================================================

# import modules
import hashlib

# import submodules
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# type hints
from typing import Dict, Iterable, List, Optional, Union

# import osintgpt projects
from osintgpt.projects.toml_io import read_toml, write_toml

# What a document did to the index on the last pass.
ADDED = 'added'
CHANGED = 'changed'
UNCHANGED = 'unchanged'
REMOVED = 'removed'

STATE_HEADER = '''\
# osintgpt index state
#
# A content hash per indexed document, so an unchanged document costs a hash
# check rather than an embedding call. Delete this file to force a full
# re-index; nothing here is a source of truth, only a record of what was seen.

'''


# hash a document's bytes
def content_hash(data: Union[bytes, str]) -> str:
    '''
    Args:
        data (Union[bytes, str]): Document content.

    Returns:
        str: SHA-256 hex digest. Text is hashed as UTF-8 so the same content \
            hashes the same regardless of how it was read.
    '''
    if isinstance(data, str):
        data = data.encode('utf-8')

    return hashlib.sha256(data).hexdigest()


# IndexedDocument class
@dataclass(frozen=True)
class IndexedDocument:
    '''
    What the index remembers about one document.
    '''
    ref: str
    hash: str
    # How many chunks it produced, so a re-index knows what it is replacing
    # without re-reading the document.
    chunks: int = 0
    indexed_at: str = ''


# IndexPlan class
@dataclass(frozen=True)
class IndexPlan:
    '''
    What a pass would do, before it does any of it.
    '''
    added: List[Path] = field(default_factory=list)
    changed: List[Path] = field(default_factory=list)
    unchanged: List[Path] = field(default_factory=list)
    # Refs the index holds that no source covers any more. Their vectors are
    # dead weight: invisible to a corpus walk, still returned by a search.
    removed: List[str] = field(default_factory=list)

    @property
    def work(self) -> List[Path]:
        '''Documents that would be read and embedded.'''
        return self.added + self.changed

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.removed)

    @property
    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f'{len(self.added)} new')
        if self.changed:
            parts.append(f'{len(self.changed)} changed')
        if self.removed:
            parts.append(f'{len(self.removed)} removed')
        if self.unchanged:
            parts.append(f'{len(self.unchanged)} unchanged')

        return ', '.join(parts) if parts else 'nothing to index'


# IndexState class
@dataclass
class IndexState:
    '''
    A hash per indexed document. Rebuildable by deleting it, so a corrupted
    state costs a re-index rather than a corpus.
    '''
    path: Path
    documents: Dict[str, IndexedDocument] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Union[str, Path]):
        '''
        Args:
            path (Union[str, Path]): The project's index state file.

        Returns:
            IndexState: What was indexed, empty when the file is absent.
        '''
        document = read_toml(path)
        documents = {}
        for row in document.get('document', []):
            ref = str(row.get('ref', ''))
            if not ref:
                continue
            documents[ref] = IndexedDocument(
                ref=ref,
                hash=str(row.get('hash', '')),
                chunks=int(row.get('chunks', 0) or 0),
                indexed_at=str(row.get('indexed_at', ''))
            )

        return cls(path=Path(path), documents=documents)

    def save(self) -> None:
        write_toml(
            self.path,
            {'document': [
                {
                    'ref': d.ref, 'hash': d.hash, 'chunks': d.chunks,
                    'indexed_at': d.indexed_at
                }
                for d in sorted(self.documents.values(), key=lambda d: d.ref)
            ]},
            header=STATE_HEADER
        )

    # what a pass would do
    def plan(
        self, files: Iterable[Path], root: Union[str, Path], force: bool = False
    ) -> IndexPlan:
        '''
        Compare the corpus against what was indexed.

        Reading a file to hash it is cheap; embedding it is not, which is why
        the comparison happens on bytes rather than on modification times — a
        touched file with unchanged content should cost nothing.

        Args:
            files (Iterable[Path]): Files the corpus currently covers.
            root (Union[str, Path]): Project root, for relative refs.
            force (bool): Treat every document as changed. The escape hatch \
                for a chunker change, which alters output without altering \
                any document.

        Returns:
            IndexPlan: Added, changed, unchanged and removed.
        '''
        root = Path(root)
        added, changed, unchanged = [], [], []
        seen = set()

        for path in files:
            ref = _ref_for(path, root)
            seen.add(ref)
            known = self.documents.get(ref)

            if known is None:
                added.append(path)
            elif force or known.hash != _hash_file(path):
                changed.append(path)
            else:
                unchanged.append(path)

        return IndexPlan(
            added=added,
            changed=changed,
            unchanged=unchanged,
            removed=sorted(set(self.documents) - seen)
        )

    # record what a pass did
    def record(
        self, path: Union[str, Path], root: Union[str, Path], chunks: int
    ) -> IndexedDocument:
        '''
        Args:
            path (Union[str, Path]): Document that was indexed.
            root (Union[str, Path]): Project root.
            chunks (int): How many chunks it produced.

        Returns:
            IndexedDocument: The entry now held for it.
        '''
        ref = _ref_for(Path(path), Path(root))
        entry = IndexedDocument(
            ref=ref,
            hash=_hash_file(Path(path)),
            chunks=chunks,
            indexed_at=datetime.now(timezone.utc).isoformat(timespec='seconds')
        )
        self.documents[ref] = entry

        return entry

    def forget(self, refs: Iterable[str]) -> int:
        '''
        Drop entries for documents no source covers any more.

        Args:
            refs (Iterable[str]): Refs to remove.

        Returns:
            int: How many entries were dropped.
        '''
        dropped = 0
        for ref in list(refs):
            if self.documents.pop(ref, None) is not None:
                dropped += 1

        return dropped

    @property
    def chunks(self) -> int:
        return sum(d.chunks for d in self.documents.values())

    def __len__(self) -> int:
        return len(self.documents)


def _hash_file(path: Path) -> str:
    return content_hash(path.read_bytes())


def _ref_for(path: Path, root: Path) -> str:
    '''
    A ref relative to the project where possible.

    Relative refs are what let a project directory be moved, archived or
    handed to a colleague without every document looking new.
    '''
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
