# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: fallback.py
# Description: A last resort for formats osintgpt has no reader for. Optional,
#   opt-in, and never used for a format that already has one.
# =================================================================================

# import submodules
from pathlib import Path

# type hints
from typing import Optional, Union

# Formats osintgpt does not read and markitdown does. Deliberately a list
# rather than "anything it accepts": where a reader exists, that reader is
# better, and a converter that also handles the format would quietly displace
# decisions taken for a reason — a CSV becomes one enormous markdown table,
# which is the opposite of the field-role model.
FALLBACK_SUFFIXES = {'.pptx', '.epub', '.msg', '.xml', '.rtf', '.odt'}


# is this a format the fallback can attempt
def can_convert(path: Union[str, Path]) -> bool:
    '''
    Args:
        path (Union[str, Path]): File to check.

    Returns:
        bool: True when osintgpt has no reader and the fallback might.
    '''
    return Path(path).suffix.lower() in FALLBACK_SUFFIXES


# convert a file osintgpt cannot read
def convert(path: Union[str, Path]) -> Optional[str]:
    '''
    Convert an unsupported file to markdown, if the converter is installed.

    Args:
        path (Union[str, Path]): File to convert.

    Raises:
        ImportError: If markitdown is not installed.

    Returns:
        Optional[str]: The markdown, or None when the converter produced \
            nothing usable.
    '''
    try:
        from markitdown import MarkItDown
    except ImportError as error:
        raise ImportError(
            f'{Path(path).suffix} needs the markitdown package, which '
            'osintgpt requires: reinstall with pip install --force-reinstall '
            'osintgpt'
        ) from error

    # Plugins stay off: a converter reaching for third-party code on a file an
    # analyst was handed is a wider surface than reading it warrants.
    converted = MarkItDown(enable_plugins=False).convert(str(path))
    text = (converted.text_content or '').strip()

    return text or None
