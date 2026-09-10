# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: browse.py
# Description: A native directory picker for the machine the app runs on.
#   Typing a path always works; this is the convenience on top.
# =================================================================================

# import modules
import logging

# type hints
from typing import Optional

log = logging.getLogger('osintgpt.app')


# whether a picker can be opened at all
def can_browse() -> bool:
    '''
    Whether this machine can show a directory dialog.

    tkinter ships with CPython but is genuinely absent on minimal builds and
    on a server with no display. The button is hidden rather than shown and
    broken, because a control that does nothing is worse than no control.

    Returns:
        bool: True when a dialog can be opened.
    '''
    try:
        import tkinter  # noqa: F401
    except Exception:  # noqa: BLE001 — absent, or present and unusable
        return False

    return True


# ask the operator for a directory
def select_directory(initial: Optional[str] = None) -> Optional[str]:
    '''
    Open a native directory dialog and return what was chosen.

    The dialog opens on the machine running the app, not in the browser —
    which is right for a tool an analyst runs locally, and wrong for one
    served to someone else. osintgpt is the first; if that ever changes, this
    is the thing that has to change with it.

    Args:
        initial (str, optional): Directory to open at.

    Returns:
        Optional[str]: The chosen path, or None when cancelled or \
            unavailable. Cancelling is an ordinary outcome and must leave \
            whatever the operator had typed untouched.
    '''
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:  # noqa: BLE001 — no display, no tkinter, no dialog
        return None

    root = None
    try:
        root = tkinter.Tk()
        root.withdraw()
        # Without this the dialog can open behind the browser, where it looks
        # like the button did nothing and the app has hung.
        root.wm_attributes('-topmost', 1)
        chosen = filedialog.askdirectory(initialdir=initial or None)
    except Exception as error:  # noqa: BLE001 — typing a path still works
        log.warning('the directory picker could not open: %s', error)

        return None
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:  # noqa: BLE001 — already gone
                pass

    return chosen or None


# a path input with a picker beside it
def directory_input(
    st,
    label: str,
    key: str,
    state,
    help_text: str = '',
    placeholder: str = ''
) -> str:
    '''
    A text field for a directory, and a button that opens a dialog.

    The field is the interface and the button is the shortcut: an operator
    can always type or paste, which is what keeps this usable where no dialog
    can open at all.

    Args:
        st: The Streamlit module.
        label (str): Field label.
        key (str): Session-state key holding the value.
        state: Session state.
        help_text (str): Help shown against the field.
        placeholder (str): Shown when empty.

    Returns:
        str: The current path, stripped.
    '''
    field = f'{key}-text'
    pending = f'{key}-chosen'

    # Applied before the field is created, because a widget with a key reads
    # its value from session state and ignores anything written to it later
    # in the same run — which is how a chosen directory used to vanish
    # between the dialog closing and the field being drawn.
    if pending in state:
        state[field] = state.pop(pending)

    columns = st.columns([6, 1])

    with columns[0]:
        typed = st.text_input(
            label, key=field, help=help_text, placeholder=placeholder
        )

    with columns[1]:
        # The spacer lines the button up with the field rather than with the
        # field's caption; Streamlit has no vertical alignment for this.
        st.markdown('<div style="height: 1.8rem"></div>',
                    unsafe_allow_html=True)
        if can_browse() and st.button(
            '📁', key=f'{key}-browse', help='Browse for a folder'
        ):
            chosen = select_directory(typed or None)
            if chosen:
                state[pending] = chosen
                st.rerun()

    state[key] = (typed or '').strip()

    return state[key]
