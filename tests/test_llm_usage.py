# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_usage.py
# Description: What a run consumed. Tokens are the record and are exact; the
#   money figure is a reading over them and says where it is incomplete.
# =================================================================================

# import modules
import pytest

# import submodules
from types import SimpleNamespace

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llm
from osintgpt.llm import (
    Usage,
    UsageRecorder,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm.usage import CostLimitReached
from osintgpt.llm.anthropic_native import AnthropicGeneration
from osintgpt.llm.local import SentenceTransformerEmbedding

# import osintgpt pricing
from osintgpt.pricing import PRICES

from conftest import FAKE_KEY, StubOpenAI

PRICED = 'gpt-4o'
UNPRICED = 'gpt-99-unreleased'


@pytest.fixture
def keyed():
    return Settings(openai_api_key=FAKE_KEY, openai_gpt_model=PRICED)


class TestUsage:
    def test_totals_both_directions(self):
        usage = Usage('openai', PRICED, input_tokens=100, output_tokens=50)

        assert usage.total_tokens == 150

    def test_a_priced_model_estimates(self):
        usage = Usage('openai', PRICED, input_tokens=1_000_000)

        assert usage.estimated_cost == PRICES[PRICED]['input']

    def test_input_and_output_price_differently(self):
        cheap = Usage('openai', PRICED, input_tokens=1_000_000)
        dear = Usage('openai', PRICED, output_tokens=1_000_000)

        assert dear.estimated_cost > cheap.estimated_cost

    def test_an_unpriced_model_is_none_not_zero(self):
        usage = Usage('openai', UNPRICED, input_tokens=1_000)

        assert usage.estimated_cost is None

    def test_a_local_provider_costs_a_real_zero(self):
        '''
        Zero and unknown are different answers. A model nobody can bill for
        must not read as a model nobody priced.
        '''
        usage = Usage('ollama', UNPRICED, input_tokens=1_000, billable=False)

        assert usage.estimated_cost == 0.0
        assert usage.estimated_cost is not None


class TestRecorder:
    def test_starts_empty(self):
        recorder = UsageRecorder()

        assert recorder.calls == 0
        assert recorder.total_tokens == 0
        assert recorder.summary == 'no provider calls'

    def test_accumulates_across_calls(self):
        recorder = UsageRecorder()
        recorder.record(Usage('openai', PRICED, 100, 50))
        recorder.record(Usage('openai', PRICED, 200, 25))

        assert recorder.calls == 2
        assert recorder.input_tokens == 300
        assert recorder.output_tokens == 75
        assert recorder.total_tokens == 375

    def test_breaks_down_by_model(self):
        recorder = UsageRecorder()
        recorder.record(Usage('openai', PRICED, 100))
        recorder.record(Usage('openai', 'text-embedding-3-small', 400))
        recorder.record(Usage('openai', PRICED, 100))

        assert recorder.by_model == {
            PRICED: 200, 'text-embedding-3-small': 400
        }

    def test_reports_unpriced_calls_beside_the_total(self):
        '''A partial sum must never be readable as a complete one.'''
        recorder = UsageRecorder()
        recorder.record(Usage('openai', PRICED, 1_000_000))
        recorder.record(Usage('openai', UNPRICED, 1_000_000))

        assert recorder.estimated_cost == PRICES[PRICED]['input']
        assert recorder.unpriced_calls == 1

    def test_local_calls_are_not_unpriced(self):
        recorder = UsageRecorder()
        recorder.record(Usage('ollama', UNPRICED, 1_000, billable=False))

        assert recorder.unpriced_calls == 0
        assert recorder.estimated_cost == 0.0

    def test_counts_calls_the_backend_did_not_measure(self):
        recorder = UsageRecorder()
        recorder.record(Usage('sentence-transformers', 'm', counted=False))

        assert recorder.uncounted_calls == 1

    def test_summary_leads_with_tokens(self):
        recorder = UsageRecorder()
        recorder.record(Usage('openai', PRICED, 1_000, 500))
        summary = recorder.summary

        assert '1,500 tokens' in summary
        assert summary.index('tokens') < summary.index('$')

    def test_summary_states_the_gaps(self):
        recorder = UsageRecorder()
        recorder.record(Usage('openai', UNPRICED, 1_000))
        recorder.record(Usage('sentence-transformers', 'm', counted=False))
        summary = recorder.summary

        assert 'unpriced' in summary
        assert 'not counted' in summary

    def test_a_ceiling_stops_after_the_call_that_crosses_it(self):
        recorder = UsageRecorder(cost_ceiling_usd=0.000003)
        recorder.record(Usage(
            'openai', 'text-embedding-3-small', input_tokens=100
        ))

        with pytest.raises(CostLimitReached, match='exceeded after'):
            recorder.record(Usage(
                'openai', 'text-embedding-3-small', input_tokens=100
            ))

        assert recorder.calls == 2

    def test_a_ceiling_refuses_an_uncounted_billable_call(self):
        recorder = UsageRecorder(cost_ceiling_usd=1.0)

        with pytest.raises(CostLimitReached, match='cannot be enforced'):
            recorder.record(Usage(
                'gateway', PRICED, counted=False
            ))

    def test_local_calls_never_consume_the_ceiling(self):
        recorder = UsageRecorder(cost_ceiling_usd=0.0)

        recorder.record(Usage(
            'ollama', UNPRICED, billable=False, counted=False
        ))

        assert recorder.calls == 1
        assert recorder.estimated_billable_cost == 0.0


class TestProviderRecording:
    def test_a_provider_without_a_recorder_still_works(self, keyed):
        provider = build_generation_provider('openai', keyed)
        provider.client = StubOpenAI()

        assert provider.generate('s', 'u') == 'STUB REPLY'
        assert provider.recorder is None

    def test_generation_records_both_directions(self, keyed):
        recorder = UsageRecorder()
        provider = build_generation_provider('openai', keyed, recorder=recorder)
        provider.client = StubOpenAI()
        provider.generate('s', 'u')

        assert recorder.calls == 1
        assert recorder.input_tokens == 11
        assert recorder.output_tokens == 7

    def test_embedding_records_once_per_batch(self, keyed):
        recorder = UsageRecorder()
        provider = build_embedding_provider('openai', keyed, recorder=recorder)
        provider.client = StubOpenAI()
        provider.embed([f'doc {i}' for i in range(250)])

        assert recorder.calls == 3
        assert recorder.total_tokens == 250 * 5

    def test_a_response_without_usage_is_recorded_as_uncounted(self, keyed):
        recorder = UsageRecorder()
        provider = build_generation_provider('openai', keyed, recorder=recorder)
        provider.client = StubOpenAI()
        provider.client.chat.completions.create = lambda **kwargs: (
            SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content='hi'))
            ])
        )
        provider.generate('s', 'u')

        assert recorder.uncounted_calls == 1
        assert recorder.total_tokens == 0

    def test_ollama_is_recorded_as_free(self):
        recorder = UsageRecorder()
        provider = build_generation_provider(
            'ollama', Settings(), model='qwen3:8b', recorder=recorder
        )
        provider.client = StubOpenAI()
        provider.generate('s', 'u')

        assert recorder.estimated_cost == 0.0
        assert recorder.unpriced_calls == 0
        assert recorder.total_tokens == 18

    def test_the_local_encoder_reports_zero_cost_and_no_count(self):
        recorder = UsageRecorder()
        provider = SentenceTransformerEmbedding(
            encoder=SimpleNamespace(encode=lambda texts: [[0.1]] * len(texts)),
            recorder=recorder
        )
        provider.embed(['a', 'b'])

        assert recorder.estimated_cost == 0.0
        assert recorder.uncounted_calls == 1

    def test_anthropic_reads_its_own_field_names(self):
        recorder = UsageRecorder()
        provider = AnthropicGeneration(
            model='claude-opus-5', api_key=FAKE_KEY, recorder=recorder,
            client=SimpleNamespace(messages=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    content=[SimpleNamespace(type='text', text='hi')],
                    usage=SimpleNamespace(input_tokens=31, output_tokens=9)
                )
            ))
        )
        provider.generate('s', 'u')

        assert recorder.input_tokens == 31
        assert recorder.output_tokens == 9

    def test_one_recorder_spans_several_providers(self):
        '''A project session mixes an embedder and a generator.'''
        recorder = UsageRecorder()
        settings = Settings(openai_api_key=FAKE_KEY, openai_gpt_model=PRICED)

        generator = build_generation_provider(
            'openai', settings, recorder=recorder
        )
        generator.client = StubOpenAI()
        embedder = build_embedding_provider(
            'openai', settings, recorder=recorder
        )
        embedder.client = StubOpenAI()

        generator.generate('s', 'u')
        embedder.embed(['a'])

        assert recorder.calls == 2
        assert set(recorder.by_model) == {PRICED, 'text-embedding-3-small'}
