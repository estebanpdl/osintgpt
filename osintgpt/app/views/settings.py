# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: settings.py
# Description: Provider, model and leg configuration for the selected project,
#   and an honest account of what leaves the machine.
# =================================================================================

# type hints
from typing import Any, Dict

# import osintgpt credentials
from osintgpt.credentials import credential_status, resolve_credentials

# import osintgpt llm
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    audit_locality
)

# import osintgpt vector store
from osintgpt.vector_store import BACKENDS as STORAGE_BACKENDS

from ..styles import badge
from .embedding_rates import controls as embedding_rate_controls
from .index_progress import model_notice
from osintgpt.index_status import saved_index_status
from ..session import _embedding_model
from dataclasses import replace

# One id per provider and role, shown to convey the shape of the string a
# field wants rather than to recommend a model. Keyed on the role too, because
# a provider's embedding ids look nothing like its generation ids and showing
# the wrong one is worse than showing none.
#
# Deliberately not a list to pick from: a selectable list is stale the day a
# provider ships a new model, and where a backend can be asked what it serves,
# `doctor --check-providers` asks it.
MODEL_SHAPES = {
    ('embedding', 'openai'): 'text-embedding-3-small',
    ('embedding', 'gemini'): 'gemini-embedding-001',
    ('embedding', 'voyage'): 'voyage-3.5',
    ('embedding', 'ollama'): 'nomic-embed-text',
    ('embedding', 'sentence-transformers'): 'all-MiniLM-L6-v2',
    ('generation', 'openai'): 'gpt-5.6-terra',
    ('generation', 'gemini'): 'gemini-2.5-flash',
    ('generation', 'anthropic'): 'claude-sonnet-5',
    ('generation', 'ollama'): 'llama3.1'
}

# The legs, and what each one is for in an analyst's words rather than the
# implementation's. A toggle whose meaning is only clear from the code is a
# toggle nobody touches.
LEGS = [
    ('semantic_enabled', 'Semantic search',
     'Finds passages that mean the same thing in different words.'),
    ('lexical_enabled', 'Exact search',
     'Finds the exact characters — handles, hashes, URLs, case numbers.'),
    ('graph_enabled', 'Relationship graph',
     'Answers how two things are connected. Costs one generation call per '
     'document to build, and must be built explicitly.')
]


# what to type into a model field
def model_help(role: str, provider: str, backends: Dict[str, Any]) -> str:
    '''
    Say what a model field wants, for the provider actually selected.

    Everything here is read from the registry rather than asserted. Whether a
    default exists, and whether the backend can be asked what it serves, are
    both properties of the backend — and "leave empty for the provider
    default" is false for every backend that has none, which is most of them.

    Args:
        role (str): 'embedding' or 'generation'.
        provider (str): The selected provider id.
        backends (Dict[str, Any]): The registry it was selected from.

    Returns:
        str: Help text for the field.
    '''
    spec = backends.get(provider)
    default = getattr(spec, 'default_model', None) if spec else None
    ending = (
        f'Leave empty to use {default}.' if default
        else f'{provider} publishes no default, so this must be filled in.'
    )

    shape = MODEL_SHAPES.get((role, provider), '')
    example = f' It looks like {shape}.' if shape else ''

    discovery = (
        ' Run "osintgpt doctor --check-providers" to list what it offers.'
        if getattr(spec, 'discovers_models', False) else ''
    )

    return (
        f'The exact id {provider} uses, not a display name.'
        f'{example}{discovery} {ending}'
    )


# render the settings view
def render(st, runtime, home, state) -> None:
    '''
    Args:
        st: The Streamlit module.
        runtime (Runtime): Project and providers.
        home (Path): The osintgpt home.
        state: Session state.
    '''
    from osintgpt.projects import load_user_defaults

    project = runtime.project
    st.subheader(f'Settings — {project.name}')
    st.caption('Everything here applies to this project only.')

    defaults = load_user_defaults(home)
    effective = project.effective_settings(defaults)
    config = project.settings_for(resolve_credentials(home), defaults)

    changes = {}
    changes.update(_providers(st, effective))
    selected = replace(effective, **changes)
    proposed = project.with_settings(**changes).settings_for(resolve_credentials(home), defaults)
    model_notice(st, saved_index_status(project), selected.embedding_provider, _embedding_model(selected, proposed))
    changes.update(embedding_rate_controls(st, effective, changes['embedding_provider']))
    changes.update(_legs(st, effective))
    changes.update(_storage(st, effective))

    if st.button('Save', type='primary'):
        project.with_settings(**changes).save()
        st.success('Saved. Reload to use the new configuration.')
        # The providers were built from the old settings, so anything cached
        # for this project is now describing a configuration that no longer
        # exists.
        st.cache_resource.clear()

    st.divider()
    _credentials(st, home)
    _locality(st, config, effective)


def _providers(st, effective) -> Dict[str, Any]:
    st.markdown('#### Models')
    columns = st.columns(2)

    embedding = sorted(EMBEDDING_BACKENDS)
    generation = sorted(GENERATION_BACKENDS)

    with columns[0]:
        provider = st.selectbox(
            'Embedding provider', embedding,
            index=embedding.index(effective.embedding_provider)
            if effective.embedding_provider in embedding else 0,
            help='Turns documents into vectors. Changing it means '
                 're-indexing: vectors from different models are not '
                 'comparable.'
        )
        model = st.text_input(
            'Embedding model', effective.embedding_model,
            help=model_help('embedding', provider, EMBEDDING_BACKENDS)
        )

    with columns[1]:
        gen_provider = st.selectbox(
            'Generation provider', generation,
            index=generation.index(effective.generation_provider)
            if effective.generation_provider in generation else 0,
            help='Reads the retrieved passages and writes the answer.'
        )
        gen_model = st.text_input(
            'Generation model', effective.generation_model,
            help=model_help('generation', gen_provider, GENERATION_BACKENDS)
        )

    ingestion = st.text_input(
        'Vision model for scanned pages', effective.ingestion_model,
        help='Reads PDF pages that hold no extractable text. Leave empty to '
             'use the generation model.'
    )

    return {
        'embedding_provider': provider,
        'embedding_model': model.strip(),
        'generation_provider': gen_provider,
        'generation_model': gen_model.strip(),
        'ingestion_model': ingestion.strip()
    }


def _legs(st, effective) -> Dict[str, Any]:
    st.markdown('#### Retrieval')
    changes = {}

    for field, label, explanation in LEGS:
        changes[field] = st.checkbox(
            label, value=getattr(effective, field), help=explanation
        )

    changes['suggest_followups'] = st.checkbox(
        'Suggest follow-up questions',
        value=effective.suggest_followups,
        help='One extra generation call after each answer, proposing what '
             'the retrieved material could answer next.'
    )

    return changes


def _storage(st, effective) -> Dict[str, Any]:
    st.markdown('#### Storage')
    backends = list(STORAGE_BACKENDS)
    backend = st.selectbox(
        'Where vectors are kept', backends,
        index=backends.index(effective.storage_backend)
        if effective.storage_backend in backends else 0,
        help='SQLite needs no server. Qdrant and Postgres need one running '
             'and configured below.'
    )

    if backend != 'sqlite':
        st.caption(
            f'Switching to {backend} does not move existing vectors. '
            'Re-index after changing this.'
        )

    return {'storage_backend': backend}


def _credentials(st, home) -> None:
    st.markdown('#### Credentials')
    st.caption(
        'Read from the environment, then from credentials stored with '
        '"osintgpt auth set". Never written into the project, and never '
        'shown here.'
    )

    for row in credential_status(home):
        state = 'good' if row.is_set else 'partial'
        label = row.source if row.is_set else 'not set'
        st.markdown(
            f'{badge(label, state)} <code>{row.variable}</code>',
            unsafe_allow_html=True
        )
        if row.shadowed:
            st.caption(
                f'{row.variable} in the environment is being used instead of '
                f'the stored {row.provider} credential.'
            )


def _locality(st, config, effective) -> None:
    st.markdown('#### What leaves this machine')

    try:
        audit = audit_locality(
            config, effective.embedding_provider, effective.generation_provider
        )
    except Exception as error:  # noqa: BLE001 — a view, not a pass
        st.caption(f'Could not determine: {error}')

        return

    status = 'good' if audit.is_local else 'partial'
    st.markdown(badge(audit.summary, status), unsafe_allow_html=True)

    for entry in audit.remote:
        st.caption(
            f'{entry.role} ({entry.provider}) — {entry.reason}'
        )

    for note in audit.setup:
        # Local after setup is still local. A model downloaded on first use
        # needs a network once, and saying so is the difference between an
        # honest claim and a marketing one.
        st.caption(f'Needs a network during setup: {note}')
