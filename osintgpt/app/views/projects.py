# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: projects.py
# Description: Creating, choosing and inspecting projects. What a project
#   holds, before anyone asks it anything.
# =================================================================================

# type hints
from typing import Any, Dict

# import osintgpt
from osintgpt import Project

# import osintgpt vector store
from osintgpt.vector_store import store_for

from ..session import list_projects, select_project
from ..styles import badge


# what a project currently holds
def describe(project) -> Dict[str, Any]:
    '''
    Counts and configuration, without opening a provider.

    Reads the store rather than the registry, because the registry records
    what a project was configured as and this answers what it actually
    contains — which is the question an operator is asking when they look.

    Args:
        project (Project): The project.

    Returns:
        Dict[str, Any]: Documents, chunks, models, and which legs are on. \
            A store that cannot be opened is reported, not raised: a \
            misconfigured backend should still let the operator see the \
            project and fix it.
    '''
    settings = project.settings
    described = {
        'name': project.name,
        'slug': project.slug,
        'root': str(project.paths.root),
        'embedding_model': settings.embedding_model or '(default)',
        'backend': settings.storage_backend,
        'legs': [
            name for name, on in (
                ('semantic', settings.semantic_enabled),
                ('lexical', settings.lexical_enabled),
                ('graph', settings.graph_enabled)
            ) if on
        ],
        'documents': None,
        'chunks': None,
        'problem': ''
    }

    try:
        with store_for(project) as store:
            described['documents'] = len(store.refs())
            described['chunks'] = store.count()
            described['stored_models'] = store.models()
    except Exception as error:  # noqa: BLE001 — a view, not a pass
        described['problem'] = str(error)

    return described


# create a project, wherever the operator wants it
def _create(name: str, location: str, home):
    '''
    A project under the home, or at a path the operator chose.

    A project kept elsewhere is registered explicitly, because a scan of the
    home will never find it — and without that entry it would not appear in
    any listing, which looks exactly like the creation having failed.

    Args:
        name (str): Project name.
        location (str): A directory, or empty for the home.
        home (Path): The osintgpt home.

    Returns:
        Project: The created project.
    '''
    from pathlib import Path

    from osintgpt.projects import Registry

    if not location:
        return Project.create(name, home=home)

    project = Project.create(name, path=Path(location).expanduser())
    Registry.load(home).register(project)

    return project


# render the projects view
def render(st, home, state) -> None:
    '''
    Args:
        st: The Streamlit module.
        home (Path): The osintgpt home.
        state: Session state.
    '''
    st.subheader('Projects')

    with st.form('create-project'):
        name = st.text_input('Name a new project')
        location = st.text_input(
            'Location (optional)',
            help='A directory to keep this project in. Leave empty to use '
                 f'{home}. Useful for keeping a case beside its material, or '
                 'on another drive.'
        )
        if st.form_submit_button('Create') and name.strip():
            try:
                project = _create(name.strip(), location.strip(), home)
            except Exception as error:  # noqa: BLE001 — the operator's to fix
                st.error(str(error))
            else:
                select_project(state, project.slug)
                st.rerun()

    entries = list_projects(home)
    if not entries:
        st.info('No projects yet. Create one above.')

        return

    current = state.get('selected_project')
    slugs = [entry.slug for entry in entries]
    chosen = st.radio(
        'Selected project',
        slugs,
        index=slugs.index(current) if current in slugs else 0,
        format_func=lambda slug: next(
            e.name for e in entries if e.slug == slug
        )
    )
    if chosen != current:
        select_project(state, chosen)
        st.rerun()

    project = next(e for e in entries if e.slug == chosen).open()
    facts = describe(project)

    if facts['problem']:
        st.warning(f'The store could not be opened: {facts["problem"]}')

    columns = st.columns(3)
    columns[0].metric('Documents', facts['documents'] if facts['documents'] is not None else '—')
    columns[1].metric('Chunks', facts['chunks'] if facts['chunks'] is not None else '—')
    columns[2].metric('Backend', facts['backend'])

    st.caption(
        f'Embedding model: {facts["embedding_model"]} · '
        f'Legs on: {", ".join(facts["legs"]) or "none"}'
    )
    st.markdown(
        f'<span class="citation-chip">{facts["root"]}</span>',
        unsafe_allow_html=True
    )

    stored = facts.get('stored_models') or []
    if len(stored) > 1:
        # Two models' vectors in one store means a switch left the old ones
        # behind. Search filters by model, so they are invisible and still
        # occupying the store.
        st.markdown(
            badge('model mismatch', 'problem')
            + ' This store holds vectors from more than one embedding model: '
            + f'{", ".join(stored)}. Only the configured one is searched.',
            unsafe_allow_html=True
        )
