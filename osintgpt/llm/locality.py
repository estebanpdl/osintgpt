# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: locality.py
# Description: Answers whether a chosen pair of providers keeps the corpus on
#   this machine, and names anything that does not.
# =================================================================================

# import submodules
from dataclasses import dataclass, field
from urllib.parse import urlparse

# type hints
from typing import List, Optional

# import osintgpt config
from osintgpt.config import DEFAULT_OLLAMA_BASE_URL, Settings

from .registry import EMBEDDING_BACKENDS, GENERATION_BACKENDS, backend_spec

# Addresses that cannot leave the machine. A hostname outside this set is
# somebody's server even when the operator owns it, so it is reported rather
# than assumed.
LOOPBACK = {'localhost', '127.0.0.1', '::1', '0.0.0.0', ''}


# is a URL served by this machine
def is_loopback(url: str) -> bool:
    '''
    Args:
        url (str): A base URL.

    Returns:
        bool: True when the host cannot be off this machine.
    '''
    host = urlparse(url if '//' in url else f'//{url}').hostname

    return (host or '') in LOOPBACK


# ProviderLocality class
@dataclass(frozen=True)
class ProviderLocality:
    '''
    One role's verdict: whether using it sends content off the machine.
    '''
    role: str
    provider: str
    is_local: bool
    reason: str


# LocalityReport class
@dataclass(frozen=True)
class LocalityReport:
    '''
    Whether a configuration is local, and what has to be in place first.
    '''
    providers: List[ProviderLocality] = field(default_factory=list)
    # One-time fetches that must happen before disconnecting. Being local is
    # not the same as never having needed a network.
    setup: List[str] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return bool(self.providers) and all(p.is_local for p in self.providers)

    @property
    def remote(self) -> List[ProviderLocality]:
        return [p for p in self.providers if not p.is_local]

    # operator-facing summary
    @property
    def summary(self) -> str:
        '''
        Returns:
            str: One line stating the verdict and naming what breaks it.
        '''
        if self.is_local:
            return 'local: nothing leaves this machine at query time'

        named = '; '.join(f'{p.role} ({p.provider}) {p.reason}' for p in self.remote)

        return f'not local: {named}'


# judge one role
def _judge(role: str, provider: str, backends, settings: Settings) -> ProviderLocality:
    spec = backend_spec(provider, backends, role)

    if not spec.local:
        return ProviderLocality(
            role=role, provider=provider, is_local=False,
            reason='sends content to a hosted API'
        )

    if not spec.ollama:
        return ProviderLocality(
            role=role, provider=provider, is_local=True,
            reason='runs in this process'
        )

    url = settings.ollama_base_url or DEFAULT_OLLAMA_BASE_URL
    if is_loopback(url):
        return ProviderLocality(
            role=role, provider=provider, is_local=True,
            reason='served from this machine'
        )

    return ProviderLocality(
        role=role, provider=provider, is_local=False,
        reason=f'points at {url}, which is not this machine'
    )


# what must be present before going offline
def _setup_for(provider: str, model: Optional[str], backends) -> List[str]:
    spec = backends.get(provider)
    if spec is None or not spec.local:
        return []

    name = model or spec.default_model or 'the chosen model'
    if spec.ollama:
        return [f'pull {name} into Ollama before disconnecting']

    # A bare name is fetched from the model hub on first use; a path is not.
    looks_like_a_path = bool(model) and ('/' in model or '\\' in model)
    if looks_like_a_path:
        return []

    return [
        f'download {name} once, or pass a path to a copy already on disk'
    ]


# audit a configuration
def audit_locality(
    settings: Settings,
    embedding_provider: str,
    generation_provider: str,
    embedding_model: Optional[str] = None,
    generation_model: Optional[str] = None
) -> LocalityReport:
    '''
    Report whether this pair of providers keeps content on the machine.

    Args:
        settings (Settings): Configuration, consulted for the Ollama host.
        embedding_provider (str): Embedding provider id.
        generation_provider (str): Generation provider id.
        embedding_model (str, optional): Chosen embedding model.
        generation_model (str, optional): Chosen generation model.

    Raises:
        ValueError: If either provider id is not registered.

    Returns:
        LocalityReport: The verdict, plus one-time setup requirements.
    '''
    providers = [
        _judge('embedding', embedding_provider, EMBEDDING_BACKENDS, settings),
        _judge('generation', generation_provider, GENERATION_BACKENDS, settings)
    ]
    setup = (
        _setup_for(embedding_provider, embedding_model, EMBEDDING_BACKENDS)
        + _setup_for(generation_provider, generation_model, GENERATION_BACKENDS)
    )

    return LocalityReport(providers=providers, setup=setup)
