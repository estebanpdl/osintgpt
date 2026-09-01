# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: session.py
# Description: What the app remembers between reruns, and the providers it
#   reuses. Everything here is testable without Streamlit, on purpose.
# =================================================================================

# import submodules
from dataclasses import dataclass
from pathlib import Path

# type hints
from typing import Any, Callable, Dict, Optional, Tuple

# import osintgpt credentials
from osintgpt.credentials import resolve_credentials

# import osintgpt llm
from osintgpt.llm import build_embedding_provider, build_generation_provider

# import osintgpt projects
from osintgpt.projects import Project, Registry, load_user_defaults

# Session keys. Named rather than inlined because a typo in one silently
# creates a second piece of state that nothing ever reads.
SELECTED = 'selected_project'
HISTORY = 'chat_history'
PENDING = 'pending_question'


# Runtime class
@dataclass(frozen=True)
class Runtime:
    '''
    A project and the providers configured for it.
    '''
    project: Project
    embedder: Any
    generator: Any

    @property
    def key(self) -> str:
        return self.project.id


# every project under a home
def list_projects(home: Path):
    '''
    Args:
        home (Path): The osintgpt home.

    Returns:
        List[RegistryEntry]: What the registry knows, rebuilt from disk so a \
            project created outside the app still appears.
    '''
    return list(Registry.rebuild(home))


# the project the operator has selected
def selected_project(state: Dict[str, Any], home: Path) -> Optional[Project]:
    '''
    Read the selection out of session state.

    Selection is the app's concept and never the library's: every library call
    takes a project explicitly, so nothing outside this module can silently
    act on "the current one".

    Args:
        state (Dict[str, Any]): Streamlit session state, or any mapping.
        home (Path): The osintgpt home.

    Returns:
        Optional[Project]: The project, or None when none is selected or the \
            selected one has gone.
    '''
    slug = state.get(SELECTED)
    if not slug:
        return None

    try:
        registry = Registry.load(home)
        if registry.find(slug) is None:
            # The index has not seen this project — created with the CLI, or
            # never written. Rescanning is the registry's own answer to that,
            # and it costs a directory listing once rather than every rerun.
            registry = Registry.rebuild(home)

        return registry.open(slug)
    except (KeyError, OSError, ValueError):
        # Selected, then deleted or moved. Forgetting is better than raising
        # on every rerun until the operator notices.
        state.pop(SELECTED, None)

        return None


# choose a project
def select_project(state: Dict[str, Any], slug: Optional[str]) -> None:
    '''
    Set the selection and clear everything that belonged to the old one.

    Chat history is per project. Carrying it across a switch would show an
    analyst answers from a corpus they are no longer looking at, which is the
    worst kind of wrong: it looks right.

    Args:
        state (Dict[str, Any]): Session state.
        slug (str, optional): Project to select, or None to clear.
    '''
    if state.get(SELECTED) == slug:
        return

    if slug:
        state[SELECTED] = slug
    else:
        state.pop(SELECTED, None)

    state.pop(HISTORY, None)
    state.pop(PENDING, None)


# the settings a project runs with
def runtime_for(
    project: Project,
    home: Path,
    builder: Optional[Callable[..., Tuple[Any, Any]]] = None
) -> Runtime:
    '''
    Build the providers a project is configured for.

    Args:
        project (Project): The project.
        home (Path): The osintgpt home, for user defaults.
        builder (Callable, optional): Builds (embedder, generator), for tests.

    Raises:
        Exception: Whatever a provider raises when it cannot be built — a \
            missing key is the operator's to fix, and the view says so.

    Returns:
        Runtime: The project and its providers.
    '''
    defaults = load_user_defaults(home)
    effective = project.effective_settings(defaults)
    config = project.settings_for(resolve_credentials(home), defaults)

    if builder is not None:
        embedder, generator = builder(effective, config)
    else:
        embedder = build_embedding_provider(
            effective.embedding_provider, config,
            model=effective.embedding_model or None
        )
        generator = build_generation_provider(
            effective.generation_provider, config,
            model=effective.generation_model or None
        )

    return Runtime(project=project, embedder=embedder, generator=generator)


# what a cached resource is keyed on
def cache_key(project: Project) -> str:
    '''
    The identity a cached client belongs to.

    Keyed on the project id rather than its slug or path: a slug can be reused
    after a delete, and a cached client answering from the wrong corpus is the
    worst bug available here — it answers, and the answer looks right.

    Args:
        project (Project): The project.

    Returns:
        str: Its cache key.
    '''
    return project.id


# remember a question and its answer
def remember(state: Dict[str, Any], question: str, answer: Any) -> None:
    '''
    Append to this project's chat history.

    Args:
        state (Dict[str, Any]): Session state.
        question (str): What was asked.
        answer: What came back.
    '''
    state.setdefault(HISTORY, []).append({
        'question': question, 'answer': answer
    })


# take a queued question
def take_pending(state: Dict[str, Any]) -> Optional[str]:
    '''
    Read and clear a question queued by a follow-up button.

    A button sets it and the next rerun consumes it. Clearing on read is what
    stops the same question being asked again on every subsequent rerun.

    Args:
        state (Dict[str, Any]): Session state.

    Returns:
        Optional[str]: The queued question, or None.
    '''
    return state.pop(PENDING, None)


# queue a question for the next rerun
def queue_question(state: Dict[str, Any], question: str) -> None:
    '''
    Args:
        state (Dict[str, Any]): Session state.
        question (str): The question a button submitted.
    '''
    cleaned = (question or '').strip()
    if cleaned:
        state[PENDING] = cleaned
