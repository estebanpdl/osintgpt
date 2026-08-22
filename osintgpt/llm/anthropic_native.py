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
                "the anthropic provider needs the 'anthropic' package: "
                'pip install osintgpt[anthropic]'
            ) from error

        self.client = Anthropic(api_key=api_key)

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
