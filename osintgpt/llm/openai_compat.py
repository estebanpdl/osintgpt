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

# import modules
import base64
import json

# import submodules
from openai import OpenAI

# type hints
from typing import List, Optional

from .base import EmbeddingProvider, GenerationProvider
from .calling import ModelTurn, ToolCall
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
    # The endpoint accepts tools. Whether the model behind it uses them well
    # is a different question, and the loop answers it by degrading when a
    # round comes back unusable rather than by trusting this flag.
    supports_tools = True
    supports_vision = True

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

    def describe_image(
        self, system: str, user: str, image: bytes,
        media_type: str = 'image/png'
    ) -> str:
        # A data URI rather than a hosted URL: the image is a page of the
        # operator's own document, and putting it somewhere fetchable to pass
        # a reference would publish the thing they are analysing.
        encoded = base64.b64encode(image).decode('ascii')
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': user},
                    {'type': 'image_url', 'image_url': {
                        'url': f'data:{media_type};base64,{encoded}'
                    }}
                ]}
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

    def generate_with_tools(self, system, user, tools, history=None):
        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ]

        for exchange in history or []:
            # The assistant turn has to carry the calls it made, or the
            # results that follow refer to nothing and the request is refused.
            messages.append({
                'role': 'assistant',
                'content': exchange.turn.text or None,
                'tool_calls': [
                    {
                        'id': call.id,
                        'type': 'function',
                        'function': {
                            'name': call.name,
                            'arguments': json.dumps(call.arguments)
                        }
                    }
                    for call in exchange.turn.calls
                ]
            })
            for call in exchange.turn.calls:
                messages.append({
                    'role': 'tool',
                    'tool_call_id': call.id,
                    'content': exchange.results.get(call.id, '')
                })

        request = {'model': self.model, 'messages': messages}
        if tools:
            request['tools'] = [
                {
                    'type': 'function',
                    'function': {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.schema()
                    }
                }
                for tool in tools
            ]

        response = self.client.chat.completions.create(**request)

        usage = getattr(response, 'usage', None)
        self._record(Usage(
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            output_tokens=getattr(usage, 'completion_tokens', 0) or 0,
            billable=self.billable,
            counted=usage is not None
        ))

        message = response.choices[0].message

        return ModelTurn(
            text=getattr(message, 'content', None) or '',
            calls=[
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_arguments(call.function.arguments)
                )
                for call in (getattr(message, 'tool_calls', None) or [])
            ]
        )

    def list_models(self) -> List[str]:
        return _list_models(self.client)


def _arguments(raw) -> dict:
    '''
    A model's arguments are a JSON string, and a malformed one is the model's
    mistake rather than a reason to fail the round: an empty mapping lets the
    tool report what it needed, which the model can then correct.
    '''
    if isinstance(raw, dict):
        return raw

    try:
        parsed = json.loads(raw or '{}')
    except ValueError:
        return {}

    return parsed if isinstance(parsed, dict) else {}
