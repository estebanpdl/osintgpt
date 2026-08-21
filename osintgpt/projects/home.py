# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: home.py
# Description: Defaults an operator sets once for every project they create.
#   Nothing here is discovered: the home is always an argument.
# =================================================================================

# import submodules
from pathlib import Path

# type hints
from typing import Union

from .settings import ProjectSettings
from .toml_io import read_toml, write_toml

CONFIG_FILE = 'config.toml'

CONFIG_HEADER = '''\
# osintgpt user defaults
#
# Settings a new project starts from, and the fallback when a project leaves a
# choice unset. Do not put API keys here; secrets belong in your environment.

'''


# the user config file for a home
def config_file(home: Union[str, Path]) -> Path:
    '''
    Args:
        home (Union[str, Path]): The osintgpt home.

    Returns:
        Path: Where user defaults live for that home.
    '''
    return Path(home) / CONFIG_FILE


# read user defaults
def load_user_defaults(home: Union[str, Path]) -> ProjectSettings:
    '''
    Read the defaults an operator set for this home.

    The home is passed in rather than looked up, so importing osintgpt never
    reaches into anyone's filesystem on its own.

    Args:
        home (Union[str, Path]): The osintgpt home.

    Returns:
        ProjectSettings: The defaults, or an unconfigured set when absent.
    '''
    document = read_toml(config_file(home))

    return ProjectSettings.from_dict(document.get('settings', {}))


# write user defaults
def save_user_defaults(
    home: Union[str, Path], settings: ProjectSettings
) -> None:
    '''
    Replace the defaults for this home.

    Args:
        home (Union[str, Path]): The osintgpt home.
        settings (ProjectSettings): Defaults to store.
    '''
    write_toml(
        config_file(home),
        {'settings': settings.to_dict()},
        header=CONFIG_HEADER
    )
