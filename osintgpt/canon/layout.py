'''Canon directory layout and deterministic page addressing.'''

import hashlib
import unicodedata

from pathlib import Path
from typing import Optional, Union

from osintgpt.projects.paths import CANON_DIR

SECTIONS = ('entities', 'narratives', 'sources', 'decisions')

_WINDOWS_RESERVED = {
    'aux', 'clock$', 'con', 'nul', 'prn',
    *(f'com{number}' for number in range(1, 10)),
    *(f'lpt{number}' for number in range(1, 10))
}

INDEX_TEXT = '''\
# Project canon

This directory holds the project's curated knowledge and links its pages.
It is maintained by osintgpt.
'''

LOG_TEXT = '''\
# Canon log

This append-only log records changes to the project's curated knowledge.
It is maintained by osintgpt.
'''


def page_slug(name: str) -> str:
    '''
    Make a stable, Unicode-preserving filename stem for a page name.

    Args:
        name (str): Human-readable page name.

    Returns:
        str: A lowercase stem, never empty.
    '''
    normalized = unicodedata.normalize('NFC', str(name).strip()).casefold()
    characters = []
    separator = False
    for character in normalized:
        if character.isalnum():
            if separator and characters:
                characters.append('-')
            characters.append(character)
            separator = False
        else:
            separator = True

    slug = ''.join(characters)
    digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]
    if slug in _WINDOWS_RESERVED:
        return f'page-{digest}'
    if len(slug) > 120:
        return f'{slug[:107]}-{digest}'
    if slug:
        return slug

    return f'page-{digest}'


def create_skeleton(canon: Union[str, Path]) -> Path:
    '''
    Create the canon files and sections without replacing existing content.

    Args:
        canon (Union[str, Path]): Project canon directory.

    Returns:
        Path: The canon directory.
    '''
    root = Path(canon)
    root.mkdir(parents=True, exist_ok=True)
    for section in SECTIONS:
        (root / section).mkdir(exist_ok=True)

    initial = {'index.md': INDEX_TEXT, 'log.md': LOG_TEXT}
    for filename, content in initial.items():
        path = root / filename
        if not path.exists():
            path.write_text(content, encoding='utf-8')

    return root


def page_path(canon: Union[str, Path], section: str, name: str) -> Path:
    '''
    Resolve a section and page name to its canonical filesystem path.

    Args:
        canon (Union[str, Path]): Project canon directory.
        section (str): One of the supported content sections.
        name (str): Human-readable page name.

    Raises:
        ValueError: If the section is not part of the canon layout.

    Returns:
        Path: Destination for the page.
    '''
    if section not in SECTIONS:
        raise ValueError(
            f'canon section must be one of: {", ".join(SECTIONS)}'
        )

    return Path(canon) / section / f'{page_slug(name)}.md'


def resolve_page(canon: Union[str, Path], target: str) -> Optional[Path]:
    '''
    Find the canon page named by a bare or section-qualified wiki target.

    Args:
        canon (Union[str, Path]): Project canon directory.
        target (str): Text inside a wiki link.

    Returns:
        Optional[Path]: Existing page path, or None when the link is broken.
    '''
    root = Path(canon)
    cleaned = unicodedata.normalize('NFC', str(target).strip())
    if not cleaned:
        return None

    parts = cleaned.replace('\\', '/').split('/', 1)
    if len(parts) == 2 and parts[0] in SECTIONS:
        candidate = page_path(root, parts[0], parts[1])

        return candidate if candidate.is_file() else None

    stem = page_slug(cleaned)
    for candidate in (
        root / f'{stem}.md',
        *(root / section / f'{stem}.md' for section in SECTIONS)
    ):
        if candidate.is_file():
            return candidate

    return None


def is_canon_ref(ref: str) -> bool:
    '''Return whether an indexed document ref names canon synthesis.'''
    parts = str(ref).replace('\\', '/').split('/')

    return bool(parts) and parts[0] == CANON_DIR
