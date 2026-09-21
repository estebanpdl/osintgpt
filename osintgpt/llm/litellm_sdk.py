# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: litellm_sdk.py
# Description: Every provider LiteLLM can route to (Bedrock, Vertex AI, Azure,
#   Mistral, Cohere, ...) through its SDK, or through a LiteLLM Proxy when a
#   base URL is configured. The route is part of the model name.
# =================================================================================

# import modules
import base64

# type hints
from typing import List, Optional

# import osintgpt config
from osintgpt.config import Settings
from osintgpt.index_events import emit_index

from .base import BatchSink, EmbeddingProvider, EmbeddingPurpose, GenerationProvider
from .openai_compat import (
    MAX_BATCH,
    chat_model_turn,
    chat_tool_messages,
    chat_tools
)
from .usage import Usage, UsageRecorder

# Module named litellm_sdk rather than litellm, for the reason
# anthropic_native.py gives: a module shadowing the SDK it imports is a trap.

PROXY_ROUTE = 'litellm_proxy/'

# Routes whose key osintgpt already stores, so `osintgpt auth set anthropic`
# also serves `anthropic/...` through LiteLLM. Every other route reads its
# provider's own environment variables (AWS_*, AZURE_*, MISTRAL_API_KEY, ...),
# which is LiteLLM's convention rather than osintgpt's.
ROUTE_KEYS = {
    'openai': 'openai_api_key',
    'anthropic': 'anthropic_api_key',
    'gemini': 'gemini_api_key',
    'voyage': 'voyage_api_key'
}


# import the SDK only when the backend is chosen
def _load_litellm():
    try:
        import litellm
    except ImportError as error:
        raise ImportError(
            "the litellm provider needs the 'litellm' package: install it "
            "with pip install 'osintgpt[litellm]'"
        ) from error

    return litellm


# the key a model's route authenticates with
def resolve_litellm_key(model: str, settings: Settings) -> Optional[str]:
    '''
    Args:
        model (str): LiteLLM model name, e.g. `anthropic/claude-sonnet-5`.
        settings (Settings): Configuration carrying the credentials.

    Returns:
        Optional[str]: The proxy's virtual key when a proxy is configured, \
            else the stored key for the model's route, else None so LiteLLM \
            reads that provider's environment variables itself.
    '''
    if settings.litellm_base_url or settings.litellm_api_key:
        return settings.litellm_api_key or None

    route = model.split('/', 1)[0] if '/' in model else 'openai'
    field = ROUTE_KEYS.get(route)

    return (getattr(settings, field) or None) if field else None


# the model name a request carries
def _routed(model: str, base_url: Optional[str]) -> str:
    # Behind a proxy the name is the proxy's alias. Without the prefix LiteLLM
    # would read `gemini-2.5-flash` as a Gemini route and skip the proxy.
    if base_url and not model.startswith(PROXY_ROUTE):
        return PROXY_ROUTE + model

    return model


# the proxy's OpenAI-compatible root
def _proxy_v1(base_url: str) -> str:
    root = base_url.rstrip('/')

    return root if root.endswith('/v1') else f'{root}/v1'


# list what a LiteLLM Proxy serves
def _list_proxy_models(base_url: Optional[str], api_key: Optional[str]) -> List[str]:
    if not base_url:
        raise NotImplementedError(
            'the litellm backend lists models only through a LiteLLM Proxy; '
            'set LITELLM_BASE_URL, or name a route such as anthropic/<model>'
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key or 'not-required', base_url=_proxy_v1(base_url))

    return sorted(model.id for model in client.models.list())


# LiteLLMGeneration class
class LiteLLMGeneration(GenerationProvider):
    '''
    Chat completions through the LiteLLM SDK.
    '''
    supports_tools = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        recorder: Optional[UsageRecorder] = None,
        client: Optional[object] = None
    ) -> None:
        '''
        Args:
            model (str): `<route>/<model>`, e.g. `bedrock/...` or \
                `anthropic/claude-sonnet-5`; behind a proxy, its alias.
            api_key (str, optional): Key for the route or the proxy. None \
                lets LiteLLM read the provider's environment variables.
            base_url (str, optional): LiteLLM Proxy URL.
            client (object, optional): Stands in for the litellm module, \
                for tests.

        Raises:
            ImportError: If litellm is not installed.
        '''
        self.model = model
        self.api_key = api_key or None
        self.base_url = base_url or None
        self.recorder = recorder
        self.client = client if client is not None else _load_litellm()
        self.supports_model_discovery = bool(self.base_url)
        self.supports_vision = self._model_supports_vision()

    def _model_supports_vision(self) -> bool:
        # LiteLLM's model map knows which models accept images. A proxy alias
        # is not in it, and the proxy decides, so it is not refused up front.
        if self.base_url:
            return True
        check = getattr(self.client, 'supports_vision', None)
        if check is None:
            return True
        try:
            return bool(check(model=self.model))
        except Exception:
            return True

    def _complete(self, messages: list, **extra):
        request = {
            'model': _routed(self.model, self.base_url),
            'messages': messages,
            # Drop what a route cannot accept rather than fail the call on it.
            'drop_params': True,
            **extra
        }
        if self.api_key:
            request['api_key'] = self.api_key
        if self.base_url:
            request['api_base'] = self.base_url

        response = self.client.completion(**request)

        usage = getattr(response, 'usage', None)
        self._record(Usage(
            provider='litellm',
            model=self.model,
            input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            output_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            counted=usage is not None
        ))

        return response.choices[0].message

    def generate(self, system: str, user: str) -> str:
        message = self._complete([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ])

        return getattr(message, 'content', None) or ''

    def describe_image(
        self, system: str, user: str, image: bytes,
        media_type: str = 'image/png'
    ) -> str:
        # The OpenAI data-URI shape; LiteLLM rewrites it into each route's own
        # image block, so this one request serves Claude and Gemini alike.
        encoded = base64.b64encode(image).decode('ascii')
        message = self._complete([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': user},
                {'type': 'image_url', 'image_url': {
                    'url': f'data:{media_type};base64,{encoded}'
                }}
            ]}
        ])

        return getattr(message, 'content', None) or ''

    def generate_with_tools(
        self, system, user, tools, history=None, conversation=None
    ):
        extra = {'tools': chat_tools(tools)} if tools else {}
        message = self._complete(
            chat_tool_messages(system, user, history, conversation), **extra
        )

        return chat_model_turn(message)

    def list_models(self) -> List[str]:
        return _list_proxy_models(self.base_url, self.api_key)


# LiteLLMEmbedding class
class LiteLLMEmbedding(EmbeddingProvider):
    '''
    Embeddings through the LiteLLM SDK.
    '''
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = MAX_BATCH,
        recorder: Optional[UsageRecorder] = None,
        client: Optional[object] = None
    ) -> None:
        '''
        Args:
            model (str): `<route>/<model>`, e.g. `cohere/embed-english-v3.0` \
                or `bedrock/amazon.titan-embed-text-v2:0`.
            api_key (str, optional): Key for the route or the proxy.
            base_url (str, optional): LiteLLM Proxy URL.
            batch_size (int): Inputs per request.
            client (object, optional): Stands in for the litellm module, \
                for tests.

        Raises:
            ImportError: If litellm is not installed.
        '''
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.model = model
        self.api_key = api_key or None
        self.base_url = base_url or None
        self.batch_size = batch_size
        self.recorder = recorder
        self.client = client if client is not None else _load_litellm()
        self.supports_model_discovery = bool(self.base_url)

    def indexing_options(self) -> dict:
        # The route is already in the model name; the proxy is what else can
        # change which model actually answers to that name.
        return {
            'endpoint': self.base_url.rstrip('/') if self.base_url else '',
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
        # Purpose is accepted and ignored, as on the OpenAI-compatible path:
        # the shared request shape has no field for it.
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            emit_index('request', attempt=1, maximum=1, inputs=len(batch))

            request = {
                'model': _routed(self.model, self.base_url),
                'input': batch,
                'drop_params': True
            }
            if self.api_key:
                request['api_key'] = self.api_key
            if self.base_url:
                request['api_base'] = self.base_url

            response = self.client.embedding(**request)

            # Items arrive as objects or plain dicts depending on the route.
            items = [
                (item.get('index'), item.get('embedding'))
                if isinstance(item, dict)
                else (getattr(item, 'index', None), getattr(item, 'embedding', None))
                for item in response.data
            ]
            ordered = sorted(items, key=lambda item: -1 if item[0] is None else item[0])
            if [index for index, _ in ordered] != list(range(len(batch))):
                raise RuntimeError(
                    f'{self.model} returned invalid vector indices for '
                    f'{len(batch)} inputs'
                )

            usage = getattr(response, 'usage', None)
            self._deliver_batch([vector for _, vector in ordered], Usage(
                provider='litellm',
                model=self.model,
                input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
                counted=usage is not None
            ), on_batch)

    def list_models(self) -> List[str]:
        return _list_proxy_models(self.base_url, self.api_key)
