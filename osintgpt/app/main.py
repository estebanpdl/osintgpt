# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: main.py
# Description: The Streamlit script. Every widget interaction reruns this file
#   top to bottom, so nothing here does work that is not asked for.
# =================================================================================

# import modules
import streamlit as st

# import submodules
from pathlib import Path

# import osintgpt projects
from osintgpt.projects import default_home

# Absolute, not relative: Streamlit executes this file as a script rather
# than importing it as a module, so it has no parent package to resolve
# against and a relative import fails at load.
from osintgpt.app.session import (
    Runtime,
    cache_key,
    runtime_for,
    selected_project
)
from osintgpt.app.styles import load_css
from osintgpt.app.views import chat, ingest, projects, settings


# providers for a project, built once per project
@st.cache_resource(show_spinner=False)
def _cached_runtime(project_id: str, project_path: str, home: str) -> Runtime:
    '''
    Keyed on the project id rather than the object, because Streamlit caches
    on argument values and a Project is not a stable key. A cached client from
    another project would answer, and the answer would be from the wrong
    corpus.
    '''
    from osintgpt import Project

    return runtime_for(Project.load(project_path), Path(home))


def main() -> None:
    st.set_page_config(page_title='osintgpt', layout='wide')
    load_css(st)
    home = default_home()

    st.sidebar.markdown(
        '<div class="osintgpt-title">osintgpt</div>'
        '<div class="osintgpt-subtitle">Ask your own documents</div>',
        unsafe_allow_html=True
    )
    view = st.sidebar.radio(
        'View', ['Projects', 'Material', 'Ask', 'Settings']
    )

    project = selected_project(st.session_state, home)

    if view == 'Projects' or project is None:
        if project is None and view != 'Projects':
            st.info('Select a project first.')
        projects.render(st, home, st.session_state)

        return

    try:
        runtime = _cached_runtime(
            cache_key(project), str(project.paths.root), str(home)
        )
    except Exception as error:  # noqa: BLE001 — the operator's to fix
        st.error(str(error))
        st.caption(
            'Set the provider credentials this project needs, then reload.'
        )

        return

    if view == 'Material':
        ingest.render(st, runtime, st.session_state)
    elif view == 'Settings':
        settings.render(st, runtime, home, st.session_state)
    else:
        chat.render(st, runtime, st.session_state)


main()
