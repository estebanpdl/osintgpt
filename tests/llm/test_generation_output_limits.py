'''Graph generation can cap output without changing ordinary generation defaults.'''

from types import SimpleNamespace

import pytest

from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.openai_compat import OpenAICompatGeneration
from osintgpt.llm.openai_responses import OpenAIResponsesGeneration
from tests.conftest import FAKE_KEY, StubOpenAI


@pytest.mark.parametrize('provider_type,endpoint,parameter', [
    (OpenAIResponsesGeneration, 'responses', 'max_output_tokens'),
    (OpenAICompatGeneration, 'chat', 'max_tokens')])
def test_output_limit_reaches_openai_request(provider_type, endpoint, parameter):
    provider = provider_type(model='offline', api_key=FAKE_KEY)
    provider.client = StubOpenAI()
    provider.generate('instructions', 'source', max_output_tokens=1234)
    provider.generate('instructions', 'source')
    requests = provider.client.responses.calls if endpoint == 'responses' else provider.client.chat.completions.calls
    assert requests[0][parameter] == 1234 and parameter not in requests[1]


def test_anthropic_uses_the_smaller_output_limit():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(content=[])

    provider = AnthropicGeneration(model='offline', api_key=FAKE_KEY, max_tokens=2000,
                                   client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    provider.generate('instructions', 'source', max_output_tokens=1000)
    provider.generate('instructions', 'source', max_output_tokens=4000)
    assert [item['max_tokens'] for item in requests] == [1000, 2000]
