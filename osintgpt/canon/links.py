'''Wiki-link parsing and non-mutating backlink and broken-link reports.'''

import re

from pathlib import Path
from typing import Dict, List, Union

from .layout import resolve_page

_WIKI_LINK = re.compile(r'\[\[([^\[\]]+)\]\]')


def links_in(text: str) -> List[str]:
    '''
    Extract wiki-link targets in their written order.

    Args:
        text (str): Markdown or other text containing wiki links.

    Returns:
        List[str]: Non-empty targets, with surrounding space removed.
    '''
    return [
        target.strip()
        for target in _WIKI_LINK.findall(str(text))
        if target.strip()
    ]


def _pages(canon: Path) -> List[Path]:
    if not canon.is_dir():
        return []

    return sorted(path for path in canon.rglob('*.md') if path.is_file())


def _relative(path: Path, canon: Path) -> str:
    return path.relative_to(canon).as_posix()


def backlinks(canon: Union[str, Path]) -> Dict[str, List[str]]:
    '''
    Map each existing target page to pages that link to it.

    Args:
        canon (Union[str, Path]): Project canon directory.

    Returns:
        Dict[str, List[str]]: Relative target paths to relative source paths.
    '''
    root = Path(canon)
    found: Dict[str, List[str]] = {}
    for source in _pages(root):
        source_name = _relative(source, root)
        for link in links_in(source.read_text(encoding='utf-8')):
            target = resolve_page(root, link)
            if target is None:
                continue
            target_name = _relative(target, root)
            sources = found.setdefault(target_name, [])
            if source_name not in sources:
                sources.append(source_name)

    return found


def broken_links(canon: Union[str, Path]) -> Dict[str, List[str]]:
    '''
    Map missing wiki targets to the pages that reference them.

    Args:
        canon (Union[str, Path]): Project canon directory.

    Returns:
        Dict[str, List[str]]: Missing target names to relative source paths.
    '''
    root = Path(canon)
    missing: Dict[str, List[str]] = {}
    for source in _pages(root):
        source_name = _relative(source, root)
        for link in links_in(source.read_text(encoding='utf-8')):
            if resolve_page(root, link) is not None:
                continue
            sources = missing.setdefault(link, [])
            if source_name not in sources:
                sources.append(source_name)

    return missing
