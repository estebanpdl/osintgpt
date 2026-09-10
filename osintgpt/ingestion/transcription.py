# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: transcription.py
# Description: Reading a page that has no extractable text, and keeping what a
#   vision model said about it so those bytes are never paid for twice.
# =================================================================================

# import modules
import hashlib
import logging

# import submodules
from pathlib import Path

# type hints
from typing import Callable, Optional, Union

# import osintgpt prompts
from osintgpt.prompts import prompt

log = logging.getLogger('osintgpt.ingestion')

SUFFIX = '.md'

# The transcription is the only record of what a scanned page said, so the
# system prompt asks for a transcriber rather than an assistant. Voice is in
# the template; this is the role.
SYSTEM = (
    'You transcribe pages of documents exactly as they are written.'
)


# where one page's transcription is kept
def cache_path(cache_dir: Union[str, Path], image: bytes) -> Path:
    '''
    Keyed on the bytes rather than the document and page number, so the same
    page re-registered under a different name, or a document re-added after a
    move, costs nothing the second time.

    Args:
        cache_dir (Union[str, Path]): Directory holding transcriptions.
        image (bytes): The rendered page.

    Returns:
        Path: Where its transcription belongs, whether or not it exists.
    '''
    digest = hashlib.sha256(image).hexdigest()

    return Path(cache_dir) / f'{digest}{SUFFIX}'


# read a page, remembering what was read
def transcriber_for(
    generator,
    cache_dir: Union[str, Path]
) -> Callable[[bytes, int], str]:
    '''
    A transcriber that reads from the cache first and writes to it after.

    Args:
        generator: A generation provider whose model can accept an image.
        cache_dir (Union[str, Path]): Where transcriptions are kept.

    Returns:
        Callable[[bytes, int], str]: Takes rendered page bytes and the page \
            number, returns markdown.
    '''
    directory = Path(cache_dir)

    def transcribe(image: bytes, page: int) -> str:
        path = cache_path(directory, image)
        cached = _read(path)
        if cached is not None:
            return cached

        text = generator.describe_image(
            SYSTEM, prompt('transcription', page=page), image
        ).strip()

        _write(path, text)

        return text

    return transcribe


# a transcriber for one project, or none when nothing needs it
def transcriber_for_project(
    project, build_generator: Callable[[], object]
) -> Callable[[bytes, int], str]:
    '''
    A transcriber caching into the project's own `extracts/`.

    The generator is built on first use rather than up front: a corpus of
    born-digital PDFs never renders a page, and demanding a generation
    credential to index one would be charging for a model that is never asked
    anything.

    Args:
        project (Project): The project being indexed.
        build_generator (Callable): Returns a generation provider when called.

    Returns:
        Callable[[bytes, int], str]: The transcriber.
    '''
    built = {}

    def generator():
        if 'it' not in built:
            built['it'] = build_generator()

        return built['it']

    class Lazy:
        def describe_image(self, system, user, image, media_type='image/png'):
            return generator().describe_image(system, user, image, media_type)

    return transcriber_for(Lazy(), project.paths.extracts)


def _read(path: Path) -> Optional[str]:
    '''
    An unreadable cache entry is a miss, not a failure: the page can always be
    transcribed again, and refusing to index because a cache file is corrupt
    would be the cache costing more than it saves.
    '''
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None


def _write(path: Path, text: str) -> None:
    '''
    A transcription that cannot be cached is still a transcription. The page
    was already read and paid for; losing the cache write must not lose it.
    '''
    if not text:
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    except OSError as error:
        log.warning('could not cache a transcription at %s: %s', path, error)
