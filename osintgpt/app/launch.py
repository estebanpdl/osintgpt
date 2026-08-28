# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: launch.py
# Description: The `osintgpt-app` entry point. Streamlit runs a script rather
#   than importing one, so this hands it the script and gets out of the way.
# =================================================================================

# import modules
import sys

# import submodules
from pathlib import Path

# type hints
from typing import List, Optional

MISSING = (
    'the app needs Streamlit: pip install osintgpt[app]'
)


# where the Streamlit script lives
def script_path() -> Path:
    '''
    Returns:
        Path: The app script, resolved against this package so it is found \
            from an installed wheel as well as a source tree.
    '''
    return Path(__file__).resolve().parent / 'main.py'


# launch the app
def main(argv: Optional[List[str]] = None) -> int:
    '''
    Run the Streamlit app.

    Args:
        argv (List[str], optional): Arguments passed through to Streamlit.

    Returns:
        int: Process exit code.
    '''
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(MISSING, file=sys.stderr)

        return 1

    sys.argv = ['streamlit', 'run', str(script_path()), *(argv or sys.argv[1:])]

    return streamlit_cli.main()


if __name__ == '__main__':
    raise SystemExit(main())
