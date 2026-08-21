# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: registry.py
# Description: Which backends exist, what each needs, and where it connects.
#   Adding a provider that speaks an OpenAI-compatible API is an entry here.
# =================================================================================

# import submodules
from dataclasses import dataclass

# type hints
from typing import Dict, Optional, Tuple

# import osintgpt config
from osintgpt.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    ENV_VARS,
    Settings
)

from .local import DEFAULT_LOCAL_EMBEDDING_MODEL

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

GEMINI_COMPAT_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'
VOYAGE_COMPAT_URL = 'https://api.voyageai.com/v1'

# Stands in where a backend needs no credential but a client still demands a
# string — Ollama ignores it, in-process backends never see it.
KEYLESS_PLACEHOLDER = 'not-required'

# BackendSpec class
@dataclass(frozen=True)
class BackendSpec:
    '''
    What one backend needs in order to be built: which client speaks to it,
    where it lives, and which setting carries its key.
    '''
    # Which implementation serves this backend. Everything sharing a kind is
    # served by one class; a new kind means a genuinely different protocol.
    kind: str
    # Settings field holding the API key; None for a keyless backend.
    settings_field: Optional[str] = None
    base_url: Optional[str] = None
    # Base URL comes from settings rather than being fixed here.
    ollama: bool = False
    # pip extra that installs this backend's SDK; None when it ships with
    # osintgpt. Named in the error when the import fails.
    extra: Optional[str] = None
    # Model to use when the caller names none. Set only where the right answer
    # is known and stable; elsewhere the caller must choose, because a guess
    # here goes stale the way a hardcoded model always has.
    default_model: Optional[str] = None
    # True where the endpoint is known to answer a list-models request. Left
    # false rather than assumed: claiming it and getting a 404 is worse than
    # not offering it.
    discovers_models: bool = False
    # True when content need not leave the machine. For Ollama this depends on
    # where its base URL points, so the flag alone is not the whole answer —
    # see locality.audit_locality.
    local: bool = False


OPENAI_COMPAT = 'openai-compat'
ANTHROPIC = 'anthropic'
SENTENCE_TRANSFORMERS = 'sentence-transformers'

EMBEDDING_BACKENDS: Dict[str, BackendSpec] = {
    'openai': BackendSpec(
        OPENAI_COMPAT, 'openai_api_key',
        default_model=DEFAULT_EMBEDDING_MODEL, discovers_models=True
    ),
    'gemini': BackendSpec(OPENAI_COMPAT, 'gemini_api_key', GEMINI_COMPAT_URL),
    'voyage': BackendSpec(OPENAI_COMPAT, 'voyage_api_key', VOYAGE_COMPAT_URL),
    # Discovery matters most here: it reports the models actually pulled onto
    # this machine, which no static list could know.
    'ollama': BackendSpec(
        OPENAI_COMPAT, None, ollama=True, discovers_models=True, local=True
    ),
    # Runs in this process: no endpoint, no key, nothing leaving the machine.
    'sentence-transformers': BackendSpec(
        SENTENCE_TRANSFORMERS, None, extra='local',
        default_model=DEFAULT_LOCAL_EMBEDDING_MODEL, local=True
    )
}

GENERATION_BACKENDS: Dict[str, BackendSpec] = {
    'openai': BackendSpec(
        OPENAI_COMPAT, 'openai_api_key', discovers_models=True
    ),
    'gemini': BackendSpec(OPENAI_COMPAT, 'gemini_api_key', GEMINI_COMPAT_URL),
    'ollama': BackendSpec(
        OPENAI_COMPAT, None, ollama=True, discovers_models=True, local=True
    ),
    # Native SDK because Anthropic publishes no compatible endpoint, not
    # because it is preferred.
    'anthropic': BackendSpec(
        ANTHROPIC, 'anthropic_api_key', extra='anthropic',
        discovers_models=True
    )
}


# look up a backend by id
def backend_spec(
    provider: str, backends: Dict[str, BackendSpec], role: str
) -> BackendSpec:
    '''
    Resolve a provider id, or say what the valid ones are.

    Args:
        provider (str): Provider id.
        backends (Dict[str, BackendSpec]): The registry to look in.
        role (str): 'embedding' or 'generation', for the error message.

    Raises:
        ValueError: If the id is not registered.

    Returns:
        BackendSpec: The backend's requirements.
    '''
    spec = backends.get(provider)
    if spec is None:
        valid = ', '.join(sorted(backends))
        raise ValueError(
            f'unknown {role} provider {provider!r}; choose one of: {valid}'
        )

    return spec


# where a backend connects
def resolve_base_url(spec: BackendSpec, settings: Settings) -> Optional[str]:
    '''
    Args:
        spec (BackendSpec): The backend.
        settings (Settings): Configuration, consulted for the Ollama host.

    Returns:
        Optional[str]: Base URL, or None to use the client's own default.
    '''
    if not spec.ollama:
        return spec.base_url

    base = settings.ollama_base_url or DEFAULT_OLLAMA_BASE_URL

    return f"{base.rstrip('/')}/v1"


# the key a backend authenticates with
def resolve_api_key(
    spec: BackendSpec, settings: Settings, provider: str
) -> str:
    '''
    Args:
        spec (BackendSpec): The backend.
        settings (Settings): Configuration carrying the credentials.
        provider (str): Provider id, for the error message.

    Raises:
        MissingEnvironmentVariableError: If a required key has no value.

    Returns:
        str: The key, or a placeholder for keyless backends.
    '''
    if spec.settings_field is None:
        return KEYLESS_PLACEHOLDER

    key = getattr(settings, spec.settings_field)
    if not key:
        raise MissingEnvironmentVariableError(
            ENV_VARS[spec.settings_field],
            hint=f'the {provider} provider needs it'
        )

    return key


# everything needed to construct a client
def connection_for(
    provider: str, backends: Dict[str, BackendSpec], role: str,
    settings: Settings
) -> Tuple[BackendSpec, Optional[str], str]:
    '''
    Validate a provider id and resolve how to reach it.

    Args:
        provider (str): Provider id.
        backends (Dict[str, BackendSpec]): The registry to look in.
        role (str): 'embedding' or 'generation'.
        settings (Settings): Configuration carrying credentials.

    Returns:
        Tuple[BackendSpec, Optional[str], str]: The spec, base URL and key.
    '''
    spec = backend_spec(provider, backends, role)

    return spec, resolve_base_url(spec, settings), resolve_api_key(
        spec, settings, provider
    )
