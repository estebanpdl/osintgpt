# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: anthropic_native.py
# Description: Claude through Anthropic's own SDK, which is the only way in —
#   there is no OpenAI-compatible endpoint to point the shared client at.
# =================================================================================

# type hints
from typing import List, Optional

from .base import GenerationProvider
from .calling import ModelTurn, ToolCall
from .usage import Usage, UsageRecorder

# The API requires an explicit ceiling. Too low truncates a reply mid-thought
# and costs a retry, so this is generous rather than frugal.
DEFAULT_MAX_TOKENS = 16000

# Module named anthropic_native rather than anthropic: a module shadowing the
# SDK it imports is a trap even where the import rules make it legal.


# AnthropicGeneration class
class AnthropicGeneration(GenerationProvider):
    '''
    Chat completions through the Anthropic SDK.
    '''
    supports_model_discovery = True

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[object] = None,
        recorder: Optional[UsageRecorder] = None
    ) -> None:
        '''
        Args:
            model (str): Claude model id.
            api_key (str): Anthropic API key.
            max_tokens (int): Ceiling on the reply.
            client (object, optional): A prepared client, for tests.

        Raises:
            ImportError: If the anthropic package is not installed.
        '''
        self.model = model
        self.max_tokens = max_tokens
        self.recorder = recorder

        if client is not None:
            self.client = client
            return

        try:
            from anthropic import Anthropic
        except ImportError as error:
            raise ImportError(
                "the anthropic provider needs the 'anthropic' package, "
                'which osintgpt requires: reinstall with pip install '
                '--force-reinstall osintgpt'
            ) from error

        self.client = Anthropic(api_key=api_key)

    supports_tools = True

    def generate_with_tools(self, system, user, tools, history=None):
        messages = [{'role': 'user', 'content': user}]

        for exchange in history or []:
            content = []
            if exchange.turn.text:
                content.append({'type': 'text', 'text': exchange.turn.text})
            content += [
                {
                    'type': 'tool_use',
                    'id': call.id,
                    'name': call.name,
                    'input': call.arguments
                }
                for call in exchange.turn.calls
            ]
            messages.append({'role': 'assistant', 'content': content})

            # Results come back as a user turn here, not a role of their own.
            # This is the shape the two vendors disagree about, and the reason
            # each provider builds its own messages rather than sharing them.
            messages.append({
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': call.id,
                        'content': exchange.results.get(call.id, '')
                    }
                    for call in exchange.turn.calls
                ]
            })

        request = {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'system': system,
            'messages': messages
        }
        if tools:
            request['tools'] = [
                {
                    'name': tool.name,
                    'description': tool.description,
                    'input_schema': tool.schema()
                }
                for tool in tools
            ]

        response = self.client.messages.create(**request)

        text = ''.join(
            block.text for block in response.content
            if getattr(block, 'type', None) == 'text'
        )
        calls = [
            ToolCall(
                id=block.id,
                name=block.name,
                arguments=dict(getattr(block, 'input', None) or {})
            )
            for block in response.content
            if getattr(block, 'type', None) == 'tool_use'
        ]

        return ModelTurn(text=text, calls=calls)

    def generate(self, system: str, user: str) -> str:
        # The system prompt is its own parameter here rather than a message,
        # which is the one shape difference from the OpenAI-compatible path.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{'role': 'user', 'content': user}]
        )

        # Field names differ from the OpenAI shape: input_tokens rather than
        # prompt_tokens.
        usage = getattr(response, 'usage', None)
        self._record(Usage(
            provider='anthropic',
            model=self.model,
            input_tokens=getattr(usage, 'input_tokens', 0) or 0,
            output_tokens=getattr(usage, 'output_tokens', 0) or 0,
            counted=usage is not None
        ))

        # A reply can carry thinking blocks alongside text; only text is asked
        # for here, and a refusal returns no text blocks at all.
        return ''.join(
            block.text for block in response.content if block.type == 'text'
        )

    def list_models(self) -> List[str]:
        return sorted(model.id for model in self.client.models.list())
