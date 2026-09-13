# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: openai_responses.py
# Description: OpenAI's own endpoint. Separate from the compatibility client
#   because /v1/responses is the only place OpenAI serves tools and reasoning
#   together, and no other backend implements it.
# =================================================================================

# import modules
import base64
import json

# import submodules
from dataclasses import dataclass
from openai import OpenAI

# type hints
from typing import Any, Dict, List, Optional, Tuple

from .base import GenerationProvider
from .calling import Exchange, ModelTurn, ToolCall, ToolSpec
from .usage import Usage, UsageRecorder

# Reasoning models emit their chain of thought as items in the response, and a
# later round reads worse without them — the model re-derives what it already
# worked out. They are opaque and provider-specific, so they ride back through
# the loop on the turn itself rather than in anything the loop understands.
@dataclass(frozen=True)
class _ResponsesTurn(ModelTurn):
    '''
    A turn carrying the raw response items needed to replay it.
    '''
    items: Tuple[Any, ...] = ()


# OpenAIResponsesGeneration class
class OpenAIResponsesGeneration(GenerationProvider):
    '''
    Generation over OpenAI's /v1/responses endpoint.
    '''
    supports_tools = True
    supports_vision = True

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        discovers_models: bool = True,
        billable: bool = True,
        provider: str = 'openai',
        recorder: Optional[UsageRecorder] = None
    ) -> None:
        '''
        Args:
            model (str): Chat model name.
            api_key (str): Credential.
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
        response = self._create(
            instructions=system,
            input=[{'role': 'user', 'content': user}]
        )

        return response.output_text or ''

    def describe_image(
        self, system: str, user: str, image: bytes,
        media_type: str = 'image/png'
    ) -> str:
        # A data URI rather than a hosted URL: the image is a page of the
        # operator's own document, and putting it somewhere fetchable to pass
        # a reference would publish the thing they are analysing.
        encoded = base64.b64encode(image).decode('ascii')
        response = self._create(
            instructions=system,
            input=[{'role': 'user', 'content': [
                {'type': 'input_text', 'text': user},
                {'type': 'input_image', 'detail': 'auto',
                 'image_url': f'data:{media_type};base64,{encoded}'}
            ]}]
        )

        return response.output_text or ''

    def generate_with_tools(
        self,
        system: str,
        user: str,
        tools: List[ToolSpec],
        history: Optional[List[Exchange]] = None,
        conversation: Optional[List[Tuple[str, str]]] = None
    ) -> ModelTurn:
        items: List[Any] = []

        for asked, answered in conversation or []:
            items.append({'role': 'user', 'content': asked})
            items.append({'role': 'assistant', 'content': answered})

        items.append({'role': 'user', 'content': user})

        for exchange in history or []:
            # Everything the model produced that round, verbatim: the calls it
            # made and the reasoning behind them. Rebuilding the calls from the
            # neutral form would drop the reasoning, which this endpoint
            # expects back alongside the results it produced.
            items.extend(getattr(exchange.turn, 'items', ()))
            for call in exchange.turn.calls:
                items.append({
                    'type': 'function_call_output',
                    'call_id': call.id,
                    'output': exchange.results.get(call.id, '')
                })

        request: Dict[str, Any] = {
            'instructions': system,
            'input': items
        }
        if tools:
            request['tools'] = [
                {
                    'type': 'function',
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.schema()
                }
                for tool in tools
            ]

        response = self._create(**request)

        return _ResponsesTurn(
            text=response.output_text or '',
            calls=[
                ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=_arguments(item.arguments)
                )
                for item in response.output
                if item.type == 'function_call'
            ],
            items=tuple(
                item.model_dump(exclude_none=True) for item in response.output
            )
        )

    def list_models(self) -> List[str]:
        return sorted(model.id for model in self.client.models.list())

    def _create(self, **request):
        '''
        Send one request and record what it consumed.

        `store=False` on every call: the default leaves the conversation on
        OpenAI's servers, and a tool whose premise is that an analyst controls
        where their material goes cannot quietly retain it elsewhere. The cost
        is that context is replayed each round rather than referenced by id.
        '''
        response = self.client.responses.create(
            model=self.model, store=False, **request
        )

        usage = getattr(response, 'usage', None)
        self._record(Usage(
            provider=self.provider,
            model=self.model,
            input_tokens=getattr(usage, 'input_tokens', 0) or 0,
            output_tokens=getattr(usage, 'output_tokens', 0) or 0,
            billable=self.billable,
            counted=usage is not None
        ))

        return response


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
