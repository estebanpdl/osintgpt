# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: base.py
# Description: The two interfaces every provider implements. Nothing outside
#   this package imports a vendor SDK; callers hold a provider and call it.
# =================================================================================

# import submodules
from abc import ABC, abstractmethod

# type hints
from typing import List, Optional

from .usage import Usage, UsageRecorder

# EmbeddingProvider class
class EmbeddingProvider(ABC):
    '''
    Turns text into vectors. `model` names the model that produced them, which
    is what stops vectors from different models being compared.
    '''
    model: str

    # True when this provider can embed images into the same vector space as
    # text. A property of the model, not the vendor.
    supports_images: bool = False

    # True when the backend can be asked what models it has. Set from the
    # registry, because it depends on the endpoint rather than the client.
    supports_model_discovery: bool = False

    # Collects what each call consumed. Optional: a provider works without one
    # and simply reports nothing.
    recorder: Optional[UsageRecorder] = None

    def _record(self, usage: Usage) -> None:
        if self.recorder is not None:
            self.recorder.record(usage)

    def list_models(self) -> List[str]:
        '''
        Ask the backend which models it offers.

        Returns whatever the endpoint reports, unclassified — an endpoint that
        serves both chat and embedding models lists both, and nothing in the
        response reliably separates them.

        Raises:
            NotImplementedError: If this backend cannot be asked.

        Returns:
            List[str]: Model names.
        '''
        raise NotImplementedError(
            f'the {type(self).__name__} backend cannot list its models'
        )

    def embed_images(self, images: List[bytes]) -> List[List[float]]:
        '''
        Embed images into the same vector space as text.

        Not abstract: most embedding models are text-only, and requiring every
        provider to implement a refusal would be ceremony. `supports_images`
        is what a caller checks; this is what it calls afterwards.

        Args:
            images (List[bytes]): Image files as stored.

        Raises:
            NotImplementedError: If the configured model embeds text only.

        Returns:
            List[List[float]]: One vector per image, in the order given.
        '''
        raise NotImplementedError(
            f'{self.model} embeds text only; use a multimodal embedding '
            'model to index images'
        )

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        '''
        Embed a batch of texts.

        Args:
            texts (List[str]): Texts to embed.

        Returns:
            List[List[float]]: One vector per input, in the same order.
        '''


# GenerationProvider class
class GenerationProvider(ABC):
    '''
    Sends a prompt to a chat model and returns its text.
    '''
    model: str

    # See EmbeddingProvider.supports_model_discovery.
    supports_model_discovery: bool = False

    # See EmbeddingProvider.recorder.
    recorder: Optional[UsageRecorder] = None

    def _record(self, usage: Usage) -> None:
        if self.recorder is not None:
            self.recorder.record(usage)

    def list_models(self) -> List[str]:
        '''
        Ask the backend which models it offers.

        Raises:
            NotImplementedError: If this backend cannot be asked.

        Returns:
            List[str]: Model names.
        '''
        raise NotImplementedError(
            f'the {type(self).__name__} backend cannot list its models'
        )

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        '''
        Single-turn completion.

        Args:
            system (str): System instruction.
            user (str): User message.

        Returns:
            str: The model's reply, empty when it produced none.
        '''
