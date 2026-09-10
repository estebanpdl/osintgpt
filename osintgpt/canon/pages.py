'''Reading and writing canon pages through one curator-facing boundary.'''

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

from .layout import create_skeleton, page_path


def write_page(
    canon: Union[str, Path],
    section: str,
    name: str,
    body: str,
    links: Optional[Iterable[str]] = None
) -> Path:
    '''
    Create or replace one canon page and return its path.

    Args:
        canon (Union[str, Path]): Project canon directory.
        section (str): Page section.
        name (str): Human-readable page name.
        body (str): Complete page body.
        links (Iterable[str], optional): Wiki targets to append.

    Returns:
        Path: The page written.
    '''
    root = create_skeleton(canon)
    path = page_path(root, section, name)
    content = str(body).rstrip()
    targets = []
    for link in links or []:
        target = str(link).strip()
        if target:
            targets.append(target)
    if targets:
        linked = '\n'.join(f'[[{target}]]' for target in targets)
        content = f'{content}\n\n{linked}' if content else linked

    path.write_text(f'{content}\n' if content else '', encoding='utf-8')

    return path


def read_page(
    canon: Union[str, Path], section: str, name: str
) -> Optional[str]:
    '''
    Read one canon page without creating the skeleton or a missing page.

    Args:
        canon (Union[str, Path]): Project canon directory.
        section (str): Page section.
        name (str): Human-readable page name.

    Returns:
        Optional[str]: Page text, or None when absent.
    '''
    path = page_path(canon, section, name)

    return path.read_text(encoding='utf-8') if path.is_file() else None


def append_log(canon: Union[str, Path], line: str) -> Path:
    '''
    Append one UTC-dated audit line to the canon log.

    Args:
        canon (Union[str, Path]): Project canon directory.
        line (str): Audit event, collapsed to one line.

    Returns:
        Path: The log file.
    '''
    root = create_skeleton(canon)
    path = root / 'log.md'
    event = ' '.join(str(line).split())
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(f'- [{timestamp}] {event}\n')

    return path
