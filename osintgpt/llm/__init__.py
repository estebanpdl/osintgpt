# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: __init__.py
# Description: The two factories that turn a provider id plus configuration
#   into something implementing one of the interfaces.
# =================================================================================

# type hints
from typing import Optional

# import osintgpt config
from osintgpt.config import Settings

from .anthropic_native import AnthropicGeneration
from .base import EmbeddingProvider, GenerationProvider
from .calling import (
    Exchange,
    ModelTurn,
    ToolCall,
    ToolCallingUnsupported,
    ToolSpec,
    tool_spec
)
from .local import SentenceTransformerEmbedding
from .locality import LocalityReport, ProviderLocality, audit_locality
from .usage import Usage, UsageRecorder
from .openai_compat import OpenAICompatEmbedding, OpenAICompatGeneration
from .registry import (
    ANTHROPIC,
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    BackendSpec,
    OPENAI_COMPAT,
    SENTENCE_TRANSFORMERS,
    backend_spec,
    connection_for
)

__all__ = [
    'BackendSpec',
    'LocalityReport',
    'ProviderLocality',
    'audit_locality',
    'EMBEDDING_BACKENDS',
    'EmbeddingProvider',
    'GENERATION_BACKENDS',
    'GenerationProvider',
    'Usage',
    'UsageRecorder',
    'build_embedding_provider',
    'build_generation_provider'
]


# build an embedding provider
def build_embedding_provider(
    provider: str,
    settings: Settings,
    model: Optional[str] = None,
    recorder: Optional[UsageRecorder] = None
) -> EmbeddingProvider:
    '''
    Construct the embedding backend named by `provider`.

    Args:
        provider (str): Provider id from EMBEDDING_BACKENDS.
        settings (Settings): Configuration carrying credentials.
        model (str, optional): Model name. Defaults to the configured \
            embedding model, then the backend's own default — which differs \
            per backend, since a local model name is not an OpenAI one.

    Raises:
        ValueError: If the provider id is not registered.
        MissingEnvironmentVariableError: If its credential is missing.

    Returns:
        EmbeddingProvider: A ready client.
    '''
    spec, base_url, api_key = connection_for(
        provider, EMBEDDING_BACKENDS, 'embedding', settings
    )
    model = model or settings.openai_embedding_model or spec.default_model
    if not model:
        raise ValueError(
            f'no model given for the {provider} embedding provider; pass '
            'model= or set one in Settings'
        )

    if spec.kind == SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedding(model=model, recorder=recorder)

    return OpenAICompatEmbedding(
        model=model, api_key=api_key, base_url=base_url,
        discovers_models=spec.discovers_models,
        billable=not spec.local, provider=provider, recorder=recorder
    )


# build a generation provider
def build_generation_provider(
    provider: str,
    settings: Settings,
    model: Optional[str] = None,
    recorder: Optional[UsageRecorder] = None
) -> GenerationProvider:
    '''
    Construct the generation backend named by `provider`.

    Args:
        provider (str): Provider id from GENERATION_BACKENDS.
        settings (Settings): Configuration carrying credentials.
        model (str, optional): Model name. Defaults to the configured chat \
            model.

    Raises:
        ValueError: If the provider id is not registered, or no model is set.
        MissingEnvironmentVariableError: If its credential is missing.

    Returns:
        GenerationProvider: A ready client.
    '''
    spec, base_url, api_key = connection_for(
        provider, GENERATION_BACKENDS, 'generation', settings
    )
    model = model or settings.openai_gpt_model
    if not model:
        raise ValueError(
            f'no model given for the {provider} generation provider; pass '
            'model= or set one in Settings'
        )

    if spec.kind == ANTHROPIC:
        return AnthropicGeneration(
            model=model, api_key=api_key, recorder=recorder
        )

    return OpenAICompatGeneration(
        model=model, api_key=api_key, base_url=base_url,
        discovers_models=spec.discovers_models,
        billable=not spec.local, provider=provider, recorder=recorder
    )
