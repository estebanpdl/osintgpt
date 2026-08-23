# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: preview.py
# Description: What osintgpt would index, without indexing it. Reads and
#   chunks; embeds nothing, so it costs nothing and can be run repeatedly.
# =================================================================================

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Dict, List, Optional, Union

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL

# import osintgpt pricing
from osintgpt.pricing import estimate_cost

# import utils
from osintgpt.utils import count_tokens

from .chunking import MAX_CHARS, chunk_text
from .documents import FieldMapping
from .loaders import READABLE_SUFFIXES, load_documents, needs_mapping
from .pdf import extract_pdf
from .tabular import UnmappedSourceError, describe_fields

# Directories that hold tooling rather than corpus.
IGNORED_DIRECTORIES = {
    '.git', '.venv', 'venv', '__pycache__', 'node_modules', '.idea',
    '.vscode', '.pytest_cache'
}


# FilePreview class
@dataclass(frozen=True)
class FilePreview:
    '''
    What one file would contribute.
    '''
    path: Path
    documents: int = 0
    chunks: int = 0
    characters: int = 0
    tokens: int = 0
    # Fields awaiting a role, for a structured file with no mapping yet.
    fields: Dict[str, dict] = field(default_factory=dict)
    # PDF pages a vision model would have to read. Counted separately because
    # this is the expensive half of ingestion, and nothing else in a dry run
    # costs a generation call.
    vision_pages: int = 0
    # Why this file would contribute nothing, if it would not.
    problem: str = ''

    @property
    def needs_configuration(self) -> bool:
        return bool(self.fields)

    @property
    def is_readable(self) -> bool:
        return not self.problem and not self.fields


# DryRun class
@dataclass(frozen=True)
class DryRun:
    '''
    What a folder would become. Nothing here has been embedded or stored.
    '''
    root: Path
    embedding_model: str
    files: List[FilePreview] = field(default_factory=list)
    unsupported: List[Path] = field(default_factory=list)

    @property
    def readable(self) -> List[FilePreview]:
        return [f for f in self.files if f.is_readable]

    @property
    def unconfigured(self) -> List[FilePreview]:
        return [f for f in self.files if f.needs_configuration]

    @property
    def failed(self) -> List[FilePreview]:
        return [f for f in self.files if f.problem]

    @property
    def documents(self) -> int:
        return sum(f.documents for f in self.readable)

    @property
    def chunks(self) -> int:
        return sum(f.chunks for f in self.readable)

    @property
    def tokens(self) -> int:
        return sum(f.tokens for f in self.readable)

    @property
    def vision_pages(self) -> int:
        '''
        Pages that would each cost one generation call to read.

        Reported apart from the embedding estimate because it is a different
        order of expense: embedding a corpus is fractions of a cent, and
        transcribing a scanned document is a call per page.
        '''
        return sum(f.vision_pages for f in self.files)

    # what embedding this would cost
    @property
    def estimated_cost(self) -> Optional[float]:
        '''
        Returns:
            Optional[float]: Estimated USD to embed once, or None when the \
                model carries no price. Covers embedding only — a graph pass \
                costs one generation call per document on top.
        '''
        return estimate_cost(self.embedding_model, self.tokens)

    # operator-facing summary
    @property
    def summary(self) -> str:
        '''
        Returns:
            str: One line, leading with what would be indexed and naming what \
                still needs a decision.
        '''
        cost = self.estimated_cost
        parts = [
            f'{len(self.readable)} files',
            f'{self.documents:,} documents',
            f'{self.chunks:,} chunks',
            f'{self.tokens:,} tokens',
            f'~${cost:.4f} to embed' if cost is not None else 'cost unknown'
        ]
        if self.vision_pages:
            parts.append(
                f'{self.vision_pages} pages need a vision model, at one '
                'generation call each'
            )
        if self.unconfigured:
            parts.append(f'{len(self.unconfigured)} need field mapping')
        if self.failed:
            parts.append(f'{len(self.failed)} unreadable')
        if self.unsupported:
            parts.append(f'{len(self.unsupported)} unsupported')

        return ', '.join(parts)


# preview one file
def preview_file(
    path: Union[str, Path],
    mapping: Optional[FieldMapping] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_chars: int = MAX_CHARS
) -> FilePreview:
    '''
    Read and chunk one file, counting what it would contribute.

    Args:
        path (Union[str, Path]): File to preview.
        mapping (FieldMapping, optional): Field roles, for structured files.
        embedding_model (str): Model whose tokenizer counts the tokens. \
            Encodings differ, so counting for the wrong model is wrong.
        max_chars (int): Chunk ceiling.

    Returns:
        FilePreview: Counts, or the fields awaiting a role, or the problem.
    '''
    path = Path(path)

    if needs_mapping(path) and (mapping is None or not mapping.is_set):
        try:
            return FilePreview(path=path, fields=describe_fields(path))
        except Exception as error:  # noqa: BLE001 — one bad file, not a stop
            return FilePreview(path=path, problem=str(error))

    try:
        documents = load_documents(path, mapping)
    except UnmappedSourceError:
        return FilePreview(path=path, fields=describe_fields(path))
    except Exception as error:  # noqa: BLE001 — one bad file, not a stop
        return FilePreview(path=path, problem=str(error))

    chunks = [
        chunk for document in documents
        for chunk in chunk_text(document.text, max_chars=max_chars)
    ]
    characters = sum(len(chunk) for chunk in chunks)

    vision_pages = 0
    if path.suffix.lower() == '.pdf':
        # Counted without transcribing: the point is to know the cost before
        # committing to it.
        try:
            vision_pages = extract_pdf(path).needs_vision
        except Exception:  # noqa: BLE001 — a count, not the document
            vision_pages = 0

    return FilePreview(
        path=path,
        documents=len(documents),
        chunks=len(chunks),
        characters=characters,
        tokens=sum(count_tokens(chunk, embedding_model) for chunk in chunks),
        vision_pages=vision_pages
    )


# preview a folder
def dry_run(
    root: Union[str, Path],
    mappings: Optional[Dict[str, FieldMapping]] = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    max_chars: int = MAX_CHARS
) -> DryRun:
    '''
    Walk a folder and report what indexing it would produce.

    One unreadable file is reported rather than raised, so a corpus with a
    corrupt document still yields a picture of the rest.

    Args:
        root (Union[str, Path]): Folder or single file to preview.
        mappings (Dict[str, FieldMapping], optional): Field roles keyed by \
            path relative to `root`, as a project file records them.
        embedding_model (str): Model whose tokenizer counts the tokens.
        max_chars (int): Chunk ceiling.

    Returns:
        DryRun: Per-file previews plus totals.
    '''
    root = Path(root)
    mappings = mappings or {}
    files: List[FilePreview] = []
    unsupported: List[Path] = []

    for path in _walk(root):
        if path.suffix.lower() not in READABLE_SUFFIXES:
            unsupported.append(path)
            continue

        key = _key_for(path, root)
        files.append(preview_file(
            path,
            mapping=mappings.get(key) or mappings.get(path.as_posix()),
            embedding_model=embedding_model,
            max_chars=max_chars
        ))

    return DryRun(
        root=root,
        embedding_model=embedding_model,
        files=files,
        unsupported=unsupported
    )


def _walk(root: Path) -> List[Path]:
    if root.is_file():
        return [root]

    found: List[Path] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if IGNORED_DIRECTORIES & set(path.relative_to(root).parts):
            continue
        found.append(path)

    return found


def _key_for(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name

    return path.relative_to(root).as_posix()
