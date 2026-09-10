# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: toml_io.py
# Description: Reading and writing the TOML files a project and its home are
#   made of, including the parser fallback older Pythons need.
# =================================================================================

# import submodules
import tomli_w

from pathlib import Path

# type hints
from typing import Union

# tomllib landed in 3.11; 3.10 is still a supported floor.
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# read a TOML document
def read_toml(path: Union[str, Path]) -> dict:
    '''
    Parse a TOML file.

    Args:
        path (Union[str, Path]): File to read.

    Returns:
        dict: The parsed document, or {} when the file is absent.
    '''
    path = Path(path)
    if not path.is_file():
        return {}

    with open(path, 'rb') as handle:
        return tomllib.load(handle)


# write a TOML document
def write_toml(
    path: Union[str, Path], document: dict, header: str = ''
) -> None:
    '''
    Serialize a document to TOML, creating parent directories as needed.

    Writing through tomli_w rather than string formatting keeps Windows paths
    and quoted names correctly escaped.

    Args:
        path (Union[str, Path]): File to write.
        document (dict): Values to serialize.
        header (str): Comment block placed above the document. TOML parsers \
            ignore it, so it survives a read/write cycle only if the caller \
            supplies it again.
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + tomli_w.dumps(document), encoding='utf-8')
