# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: images.py
# Description: Standalone images. There is no text to chunk, so the only
#   question is whether a vector can be stored for one — and if not, saying so.
# =================================================================================

# import submodules
from pathlib import Path

# type hints
from typing import Union

# Raster formats an analyst is likely to be handed. SVG is deliberately absent:
# it is markup, and treating it as a picture would embed a rendering of text
# that the lexical leg could otherwise search directly.
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif',
                  '.tiff'}

# What a stored image chunk carries as its text. A marker rather than a
# caption: nothing was extracted, and inventing a description here would put
# words into the index that no model produced.
MARKER = '[Image: {name}]'

# Refused files are reported with this, and it names no vendor on purpose —
# several providers offer a multimodal embedding model, and the choice is the
# operator's.
NO_IMAGE_SUPPORT = (
    '{name}: the configured embedding model ({model}) embeds text only. '
    'Use a multimodal embedding model to index images.'
)


# is this a standalone image
def is_image(path: Union[str, Path]) -> bool:
    '''
    Args:
        path (Union[str, Path]): File to check.

    Returns:
        bool: True when the file is an image rather than a document.
    '''
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


# read an image for embedding
def read_image(path: Union[str, Path]) -> bytes:
    '''
    Args:
        path (Union[str, Path]): Image to read.

    Returns:
        bytes: The file as stored. No decoding here — which formats a \
            provider accepts is the provider's business, and re-encoding \
            would lose whatever it was given.
    '''
    return Path(path).read_bytes()


# how an image names itself in the index
def marker_for(path: Union[str, Path]) -> str:
    '''
    Args:
        path (Union[str, Path]): The image.

    Returns:
        str: The text stored alongside its vector.
    '''
    return MARKER.format(name=Path(path).name)
