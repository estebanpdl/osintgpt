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

from .base import BatchSink, EmbeddingProvider, EmbeddingPurpose
from .usage import Usage, UsageRecorder
from .embedding_pacing import EmbeddingPacer
from .embedding_retry import ATTEMPT_TIMEOUT_SECONDS
from osintgpt.index_events import emit_index

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
        recorder: Optional[UsageRecorder] = None,
        rate_limits=None,
        rate_state_path: str = '',
        retry_policy=None
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
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.model = model
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=ATTEMPT_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=1)
            )
        )
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
        endpoint = self.indexing_options()['endpoint'] or 'https://generativelanguage.googleapis.com'
        self.pacer = EmbeddingPacer('gemini', model, endpoint, api_key, rate_limits, rate_state_path, retry_policy=retry_policy)

    def indexing_options(self) -> dict:
        # Include the resolved template/task, not just its selector: changing
        # our template also changes the vector space for an unchanged model.
        api = getattr(self.client, '_api_client', None)
        http = getattr(api, '_http_options', None)
        return {
            'task_style': self.task_style,
            'document_task': TASK_TYPES[EmbeddingPurpose.DOCUMENT]
                if self.task_style == TASK_TYPE_STYLE
                else INSTRUCTIONS[EmbeddingPurpose.DOCUMENT],
            'endpoint': str(getattr(http, 'base_url', '') or ''),
            'api_version': str(getattr(http, 'api_version', '') or ''),
            'request_version': 1
        }

    def embed(
        self,
        texts: List[str],
        *,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> List[List[float]]:
        vectors: List[List[float]] = []
        self.embed_checkpointed(texts, purpose=purpose, on_batch=vectors.extend)
        return vectors

    def embed_checkpointed(
        self, texts: List[str], *, on_batch: BatchSink,
        purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT
    ) -> None:
        pacer = getattr(self, 'pacer', None)
        render = (
            (lambda text: INSTRUCTIONS[purpose].format(text=text))
            if self.task_style == INSTRUCTION_STYLE else (lambda text: text)
        )
        def request(batch):
            if not pacer:
                emit_index('request', attempt=1, maximum=1, inputs=len(batch))
            return self._request(batch, purpose)

        if pacer:
            responses = pacer.responses(texts, self.batch_size, request, render=render)
        else:
            batches = (texts[start:start + self.batch_size] for start in range(0, len(texts), self.batch_size))
            responses = ((batch, request(batch), None) for batch in batches)
        for batch, response, ticket in responses:
            self._deliver_response(batch, response, on_batch, ticket)

    def _request(self, batch: Sequence[str], purpose: EmbeddingPurpose):
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

        return self.client.models.embed_content(model=self.model, contents=contents, config=config)

    def _deliver_response(self, batch, response, on_batch, ticket):
        pacer = getattr(self, 'pacer', None)
        embeddings = response.embeddings or []
        tokens = _token_count(embeddings)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f'{self.model} returned {len(embeddings)} vectors for '
                f'{len(batch)} inputs'
            )

        def deliver(vectors):
            on_batch(vectors)
            if pacer:
                http_response = getattr(response, 'sdk_http_response', None)
                pacer.observe(ticket, tokens=tokens, headers=getattr(http_response, 'headers', None))

        self._deliver_batch([list(e.values) for e in embeddings], Usage(
            provider='gemini',
            model=self.model,
            input_tokens=tokens or 0,
            counted=tokens is not None
        ), deliver)


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
