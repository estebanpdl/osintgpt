# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: gemini.py
# Description: Native Gemini embeddings. The compatibility endpoint cannot say
#   what a vector is for, and a retrieval pair embedded without that is the
#   thing this module exists to avoid.
# =================================================================================

# import submodules
from google import genai
from google.genai import types

# type hints
from typing import List, Optional, Sequence

from .base import EmbeddingProvider, EmbeddingPurpose
from .usage import Usage, UsageRecorder

# Gemini rejects batches over 100 inputs.
MAX_BATCH = 100

TASK_TYPE_STYLE = 'task_type'
INSTRUCTION_STYLE = 'instruction'

# Models that take the task as a request parameter. Everything else is sent it
# as an instruction, which is where Google has moved: a model released after
# this set was written inherits the current behaviour instead of an error.
TASK_TYPE_MODELS = {
    'gemini-embedding-001',
    'text-embedding-004',
    'embedding-001'
}

TASK_TYPES = {
    EmbeddingPurpose.DOCUMENT: 'RETRIEVAL_DOCUMENT',
    EmbeddingPurpose.QUERY: 'RETRIEVAL_QUERY'
}

# Google's templates for the models carrying no task_type. The document side
# names no title because what is embedded already renders its section path,
# and repeating it would only weight the heading.
INSTRUCTIONS = {
    EmbeddingPurpose.DOCUMENT: 'title: none | text: {text}',
    EmbeddingPurpose.QUERY: 'task: search result | query: {text}'
}


# GeminiEmbedding class
class GeminiEmbedding(EmbeddingProvider):
    '''
    Embeddings from the Gemini API, told on every call which side of a
    retrieval pair they are for.
    '''
    def __init__(
        self,
        model: str,
        api_key: str,
        batch_size: int = MAX_BATCH,
        task_style: Optional[str] = None,
        recorder: Optional[UsageRecorder] = None
    ) -> None:
        '''
        Args:
            model (str): Embedding model name.
            api_key (str): Gemini API key.
            batch_size (int): Inputs per request.
            task_style (str, optional): 'task_type' or 'instruction'. \
                Defaults to whichever the model takes — the escape hatch for \
                a model whose answer changes after this was written.

        Raises:
            ValueError: If `task_style` names neither style.
        '''
        self.model = model
        self.client = genai.Client(api_key=api_key)
        self.batch_size = batch_size
        self.recorder = recorder

        if task_style is None:
            task_style = (
                TASK_TYPE_STYLE if model in TASK_TYPE_MODELS
                else INSTRUCTION_STYLE
            )

        if task_style not in (TASK_TYPE_STYLE, INSTRUCTION_STYLE):
            raise ValueError(
                f'unknown task style {task_style!r}; choose one of: '
                f'{INSTRUCTION_STYLE}, {TASK_TYPE_STYLE}'
            )

        self.task_style = task_style

    def embed(
        self,
        texts: List[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._embed(texts[start:start + self.batch_size], purpose)
            )

        return vectors

    def _embed(
        self, batch: Sequence[str], purpose: EmbeddingPurpose
    ) -> List[List[float]]:
        if self.task_style == TASK_TYPE_STYLE:
            contents = list(batch)
            config = types.EmbedContentConfig(task_type=TASK_TYPES[purpose])
        else:
            template = INSTRUCTIONS[purpose]
            # One Content per text. Given bare strings these models return a
            # single aggregated vector for the whole batch instead of one per
            # input, which no caller here could detect from the numbers.
            contents = [
                types.Content(
                    parts=[types.Part.from_text(text=template.format(text=t))]
                )
                for t in batch
            ]
            config = None

        response = self.client.models.embed_content(
            model=self.model, contents=contents, config=config
        )

        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f'{self.model} returned {len(embeddings)} vectors for '
                f'{len(batch)} inputs'
            )

        tokens = _token_count(embeddings)
        self._record(Usage(
            provider='gemini',
            model=self.model,
            input_tokens=tokens or 0,
            counted=tokens is not None
        ))

        return [list(embedding.values) for embedding in embeddings]


def _token_count(embeddings) -> Optional[int]:
    '''
    Only the Vertex surface reports per-input token counts. The developer API
    reports billable characters instead, which is not a token count and is not
    converted into one: an absent count says so and stops a cost ceiling
    rather than being guessed at.
    '''
    counts = [
        getattr(getattr(embedding, 'statistics', None), 'token_count', None)
        for embedding in embeddings
    ]
    if not counts or any(count is None for count in counts):
        return None

    return int(sum(counts))
