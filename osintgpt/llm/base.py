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
from enum import Enum
from dataclasses import asdict
from osintgpt.index_events import emit_index

# type hints
from typing import Callable, List, Optional, Tuple

from .calling import Exchange, ModelTurn, ToolCallingUnsupported, ToolSpec
from .usage import Usage, UsageRecorder

BatchSink = Callable[[List[List[float]]], None]


# EmbeddingPurpose class
class EmbeddingPurpose(Enum):
    '''
    Which side of a retrieval pair a vector is for. Some models embed a
    question and the passage answering it differently; those that do not
    ignore this.
    '''
    DOCUMENT = 'document'
    QUERY = 'query'


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

    def indexing_options(self) -> dict:
        '''Vector-affecting options for index/checkpoint compatibility.

        Custom providers must override this when options beyond their model
        affect vectors. Do not include credentials, batching or usage limits.
        '''
        return {}

    def can_adopt_legacy_index(self) -> bool:
        '''Legacy stores do not record provider options; default to unknown.'''
        return False

    def _record(self, usage: Usage) -> None:
        try:
            if self.recorder is not None:
                self.recorder.record(usage)
        finally:
            emit_index('usage', **asdict(usage))

    def embed_checkpointed(
        self, texts: List[str], *, on_batch: BatchSink,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> None:
        '''
        Deliver successful vectors to a durable sink. Providers making several
        requests override this to deliver each response before the next call.
        Existing custom providers remain compatible at whole-call granularity.
        '''
        on_batch(self.embed(texts, purpose=purpose))

    def _deliver_batch(
        self, vectors: List[List[float]], usage: Usage, on_batch: BatchSink
    ) -> None:
        # Persist a successful response before accounting can stop the run.
        # Even a failed disk write must account for the call already made.
        try:
            on_batch(vectors)
        finally:
            self._record(usage)

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

    def embed_images(
        self,
        images: List[bytes],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> List[List[float]]:
        '''
        Embed images into the same vector space as text.

        Not abstract: most embedding models are text-only, and requiring every
        provider to implement a refusal would be ceremony. `supports_images`
        is what a caller checks; this is what it calls afterwards.

        Args:
            images (List[bytes]): Image files as stored.
            purpose (EmbeddingPurpose): Which side of a retrieval pair these \
                vectors are for.

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
    def embed(
        self,
        texts: List[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> List[List[float]]:
        '''
        Embed a batch of texts.

        Args:
            texts (List[str]): Texts to embed.
            purpose (EmbeddingPurpose): Which side of a retrieval pair these \
                vectors are for. Defaulting to DOCUMENT keeps a caller that \
                passes nothing embedding both sides alike, which is what \
                every model without the distinction does anyway.

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
        try:
            if self.recorder is not None:
                self.recorder.record(usage)
        finally:
            emit_index('usage', **asdict(usage))

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

    # True when this backend can be given tools and asked to call them. A
    # property of the model as much as the vendor: a small local model served
    # by an endpoint that supports tools may still not use them well.
    supports_tools: bool = False

    # True when this backend accepts an image alongside a prompt. Like
    # supports_tools, it is a property of the model as much as the vendor:
    # an endpoint that accepts images will still refuse for a text-only model.
    supports_vision: bool = False

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

    def describe_image(
        self, system: str, user: str, image: bytes, media_type: str = 'image/png'
    ) -> str:
        '''
        Single-turn completion over an image.

        Not abstract, for the same reason `embed_images` is not: most models
        cannot do this, and making every backend implement a refusal would be
        ceremony. `supports_vision` is the flag to check first.

        Args:
            system (str): System instruction.
            user (str): What to do with the image.
            image (bytes): The image, encoded as `media_type` says.
            media_type (str): Its MIME type.

        Raises:
            NotImplementedError: If this backend cannot be given an image.

        Returns:
            str: The model's reply.
        '''
        raise NotImplementedError(
            f'the {type(self).__name__} backend cannot be given an image'
        )

    def generate_with_tools(
        self,
        system: str,
        user: str,
        tools: List['ToolSpec'],
        history: Optional[List['Exchange']] = None,
        conversation: Optional[List[Tuple[str, str]]] = None
    ) -> 'ModelTurn':
        '''
        One round of a tool-calling conversation.

        Not abstract: a backend that cannot call tools should say so by
        raising, and requiring every provider to implement a refusal would be
        ceremony. `supports_tools` is what a caller checks first.

        Passing an empty `tools` list is how the caller says *answer now* —
        the model is given no way to ask for more, so it must reply from what
        it already has.

        `history` and `conversation` are different things and the names are
        not interchangeable: `history` is the tool calls made while answering
        *this* question, and `conversation` is what was asked and answered
        *before* it. Confusing them sends a model the wrong material.

        Args:
            system (str): System instruction.
            user (str): The question, unchanged across rounds.
            tools (List[ToolSpec]): Tools the model may call this round.
            history (List[Exchange], optional): Completed rounds, in order.
            conversation (List[Tuple[str, str]], optional): Earlier questions \
                and the answers they got, oldest first, placed ahead of this \
                question as prior turns. Carries no passages: the tools can \
                retrieve those again, and re-sending them would crowd out the \
                corpus material this answer has to be grounded in.

        Raises:
            ToolCallingUnsupported: If this backend cannot call tools.

        Returns:
            ModelTurn: What it said, what it wants to run, or both.
        '''
        raise ToolCallingUnsupported(
            f'{self.model} cannot call tools; the static pipeline answers '
            'instead'
        )
