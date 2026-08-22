# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: openai_compat.py
# Description: One client for every backend speaking the OpenAI API — OpenAI,
#   Gemini's compatibility endpoint, Voyage and Ollama differ only by base URL.
# =================================================================================

# import submodules
from openai import OpenAI

# type hints
from typing import List, Optional

from .base import EmbeddingProvider, GenerationProvider
from .usage import Usage, UsageRecorder


# list the models an OpenAI-compatible endpoint reports
def _list_models(client) -> List[str]:
    '''
    Args:
        client: An OpenAI-compatible client.

    Returns:
        List[str]: Model ids, sorted. Against Ollama these are the
            models pulled onto this machine.
    '''
    return sorted(model.id for model in client.models.list())


# read a prompt-token count that some gateways omit
def _prompt_tokens(response) -> int:
    usage = getattr(response, 'usage', None)

    return getattr(usage, 'prompt_tokens', 0) or 0


# Gemini's compatibility endpoint rejects batches over 100 inputs. Other
# backends allow more, so 100 is the safe floor rather than a tuning knob.
MAX_BATCH = 100


# OpenAICompatEmbedding class
class OpenAICompatEmbedding(EmbeddingProvider):
    '''
    Embeddings over any OpenAI-compatible endpoint.
    '''
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        batch_size: int = MAX_BATCH,
        discovers_models: bool = False,
        billable: bool = True,
        provider: str = '',
        recorder: Optional[UsageRecorder] = None
    ) -> None:
        '''
        Args:
            model (str): Embedding model name.
            api_key (str): Credential; keyless backends pass a placeholder.
            base_url (str, optional): Endpoint, or None for OpenAI's own.
            batch_size (int): Inputs per request.
            discovers_models (bool): Whether this endpoint answers a
                list-models request.
        '''
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size
        self.supports_model_discovery = discovers_models
        self.billable = billable
        self.provider = provider
        self.recorder = recorder

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=texts[start:start + self.batch_size]
            )
            # Providers are not required to return the batch in order.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
            self._record(Usage(
                provider=self.provider,
                model=self.model,
                input_tokens=_prompt_tokens(response),
                billable=self.billable,
                counted=getattr(response, 'usage', None) is not None
            ))

        return vectors

    def list_models(self) -> List[str]:
        return _list_models(self.client)


# OpenAICompatGeneration class
class OpenAICompatGeneration(GenerationProvider):
    '''
    Chat completions over any OpenAI-compatible endpoint.
    '''
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        discovers_models: bool = False,
        billable: bool = True,
        provider: str = '',
        recorder: Optional[UsageRecorder] = None
    ) -> None:
        '''
        Args:
            model (str): Chat model name.
            api_key (str): Credential; keyless backends pass a placeholder.
            base_url (str, optional): Endpoint, or None for OpenAI's own.
            discovers_models (bool): Whether this endpoint answers a
                list-models request.
        '''
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.supports_model_discovery = discovers_models
        self.billable = billable
        self.provider = provider
        self.recorder = recorder

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ]
        )

        usage = getattr(response, 'usage', None)
        self._record(Usage(
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            output_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            billable=self.billable,
            counted=usage is not None
        ))

        return response.choices[0].message.content or ''

    def list_models(self) -> List[str]:
        return _list_models(self.client)
