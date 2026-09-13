# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: conversations.py
# Description: The sidebar list of a project's threads. Listing reads a file
#   per conversation, so it is cached on what the directory looks like rather
#   than rebuilt on every keystroke.
# =================================================================================

# import modules
import os
import streamlit as st

# import submodules
from pathlib import Path

# type hints
from typing import Any, Dict, List, Tuple

# import osintgpt projects
from osintgpt.projects import (
    delete_conversation,
    list_in,
    open_conversation,
    rename_conversation
)
from osintgpt.projects.conversations import conversations_dir

from ..session import ASK, CONVERSATION, VIEW, select_conversation

# Threads listed before a filter is offered instead of a longer list. A
# sidebar that scrolls past the fold stops being a way to find anything.
MAX_LISTED = 20

# Widget-key prefixes, which reach the DOM as `st-key-<key>` and are how the
# stylesheet finds these buttons. Streamlit renders each element in its own
# container, so a wrapper div cannot enclose the widgets written after it.
# The open thread takes its own prefix to be picked out the same way.
THREAD_KEY = 'conv-'
OPEN_KEY = 'conv-open-'


# what the conversations directory looks like right now
def fingerprint(directory) -> Tuple:
    '''
    A cheap signature of a directory, for cache invalidation.

    Name, size and modification time per file, read from the directory entry
    without opening anything. Appending a turn changes size and time, a
    rename appends, and a delete removes the entry — so every change the
    listing can show is a change this signature sees.

    Args:
        directory (Union[str, Path]): A conversations directory.

    Returns:
        Tuple: The signature, empty when the directory does not exist.
    '''
    directory = Path(directory)
    if not directory.is_dir():
        return ()

    try:
        return tuple(sorted(
            (entry.name, entry.stat().st_size, entry.stat().st_mtime)
            for entry in os.scandir(directory) if entry.is_file()
        ))
    except OSError:
        return ()


# the conversations to show, cached on the directory's signature
@st.cache_data(show_spinner=False)
def _listed(directory: str, signature: Tuple) -> List[Tuple[str, str]]:
    '''
    `signature` is not read. It is the cache key: the directory's own contents
    decide when this is stale, so nothing has to remember to clear it.
    '''
    return [(item.id, item.name) for item in list_in(directory)]


# render the sidebar section
def render(st_module, project, state: Dict[str, Any]) -> None:
    '''
    List the project's conversations, and offer a new one.

    Args:
        st_module: The Streamlit module.
        project (Project): The selected project.
        state: Session state.
    '''
    sidebar = st_module.sidebar
    directory = conversations_dir(project)
    threads = _listed(str(directory), fingerprint(directory))

    sidebar.markdown(
        '<div class="sidebar-label">Conversations</div>',
        unsafe_allow_html=True
    )
    sidebar.button(
        'New conversation', key='conversation-new',
        on_click=_start, args=(state,), width='stretch'
    )

    if not threads:
        sidebar.caption('Ask something to start one.')

        return

    current = state.get(CONVERSATION)

    for conversation_id, name in _filtered(sidebar, threads):
        prefix = OPEN_KEY if conversation_id == current else THREAD_KEY
        sidebar.button(
            name, key=f'{prefix}{conversation_id}',
            on_click=_open, args=(state, conversation_id), width='stretch'
        )

    if current:
        _manage(st_module, project, state, current)


def _filtered(sidebar, threads) -> List[Tuple[str, str]]:
    '''
    The threads to show, narrowed by a filter once there are enough of them
    that scrolling stops being a way to find one.
    '''
    if len(threads) <= MAX_LISTED:
        return threads

    needle = sidebar.text_input(
        'Filter conversations', key='conversation-filter',
        placeholder='Filter by title', label_visibility='collapsed'
    ).strip().lower()

    if needle:
        # Every match, however many. A filter that also truncates is one an
        # analyst cannot trust to have found the thread they remember.
        return [t for t in threads if needle in t[1].lower()]

    sidebar.caption(
        f'{len(threads)} conversations. Showing the {MAX_LISTED} most '
        'recent; filter to reach the rest.'
    )

    return threads[:MAX_LISTED]


def _manage(st_module, project, state, conversation_id: str) -> None:
    '''
    Rename or delete the open conversation.

    Behind an expander because neither is reached for often, and a delete
    control sitting open beside a list of threads is one that eventually gets
    clicked by accident.
    '''
    conversation = open_conversation(project, conversation_id)
    if conversation is None:
        return

    panel = st_module.sidebar.expander('Rename or delete')

    title = panel.text_input(
        'Title', value=conversation.name,
        # Keyed on the conversation, so switching threads does not carry the
        # previous one's half-typed name across.
        key=f'conversation-title-{conversation_id}'
    )
    if panel.button('Save title', key='conversation-rename'):
        rename_conversation(conversation, title)
        st_module.rerun()

    # Losing a transcript is not recoverable, so the button stays inert
    # until the box above it is ticked.
    confirmed = panel.checkbox(
        'Yes, delete this conversation', key='conversation-confirm'
    )
    if panel.button(
        'Delete', key='conversation-delete', disabled=not confirmed
    ):
        delete_conversation(conversation)
        select_conversation(state, None)
        st_module.rerun()


def _start(state) -> None:
    select_conversation(state, None)
    state[VIEW] = ASK


def _open(state, conversation_id: str) -> None:
    select_conversation(state, conversation_id)
    state[VIEW] = ASK
