# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_agentic_loop.py
# Description: The loop the model drives. Three properties matter more than
#   the mechanics: an empty answer is impossible, the cap is never disclosed,
#   and a model that cannot call tools still answers.
# =================================================================================

# import modules
import json
import math
import pytest

# import osintgpt
from osintgpt import Project, agentic_answer, index_project
from osintgpt.agentic import MAX_ROUNDS, TOOL_NAMES, TOOL_SPECS, run_tool
from osintgpt.agentic.tools import ToolContext
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.llm.calling import ModelTurn, ToolCall, ToolCallingUnsupported

MODEL = 'test-embedding'


class WordEmbedder(EmbeddingProvider):
    model = MODEL
    VOCABULARY = 'aardvark zebra quokka'.split()

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        low = text.lower()
        counts = [float(low.count(w)) for w in self.VOCABULARY]
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


class ScriptedModel(GenerationProvider):
    '''
    Plays prepared turns in order, and records what it was offered.
    '''

    model = 'test-generation'
    supports_tools = True

    def __init__(self, *turns, error=None):
        self.turns = list(turns)
        self.error = error
        self.offered = []
        self.systems = []
        self.histories = []

    def generate(self, system, user, **kwargs):
        return 'static answer'

    def generate_with_tools(self, system, user, tools, history=None):
        if self.error:
            raise self.error

        self.offered.append([t.name for t in tools])
        self.systems.append(system)
        self.histories.append(list(history or []))

        if not self.turns:
            return ModelTurn(text='ran out of script')

        return self.turns.pop(0)


class ToollessModel(GenerationProvider):
    model = 'small-local-model'
    supports_tools = False

    def generate(self, system, user, **kwargs):
        return 'answered without tools'


def calls(*names):
    return ModelTurn(
        text='',
        calls=[
            ToolCall(id=f'c{i}', name=name, arguments=args)
            for i, (name, args) in enumerate(names)
        ]
    )


@pytest.fixture
def embedder():
    return WordEmbedder()


@pytest.fixture
def project(tmp_path, embedder):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()
    (material / 'alpha.md').write_text(
        '# Alpha\n\nA report about aardvark sightings near @acct_1.',
        encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register('material')
    index_project(instance, embedder)

    return instance


class TestAnEmptyAnswerIsImpossible:
    '''
    Whatever goes wrong, something is returned. An analyst can act on "the
    corpus does not cover this"; they cannot act on silence.
    '''

    def test_a_model_that_cannot_call_tools_still_answers(
        self, project, embedder
    ):
        answer = agentic_answer(project, 'q', embedder, ToollessModel())

        assert answer.text
        assert answer.degraded

    def test_a_provider_refusing_tools_degrades(self, project, embedder):
        model = ScriptedModel(error=ToolCallingUnsupported('no tools here'))

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.text
        assert 'no tools here' in answer.trace.degraded

    def test_a_provider_error_degrades_rather_than_raising(
        self, project, embedder
    ):
        model = ScriptedModel(error=RuntimeError('the endpoint fell over'))

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.text
        assert 'fell over' in answer.trace.degraded

    def test_a_model_that_calls_forever_still_produces_an_answer(
        self, project, embedder
    ):
        forever = [calls(('semantic_search', {'query': 'aardvark'}))] * 20
        model = ScriptedModel(*forever)

        answer = agentic_answer(project, 'q', embedder, model, max_rounds=3)

        assert answer.text

    def test_a_final_round_producing_nothing_falls_back(
        self, project, embedder
    ):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='   ')
        )

        answer = agentic_answer(project, 'q', embedder, model, max_rounds=1)

        assert answer.text
        assert answer.degraded


class TestTheCapIsNotDisclosed:
    '''
    Knowing a budget invites spending it. The model is told what the tools do
    and nothing about how many rounds it has.
    '''

    def test_the_prompt_never_names_a_round_limit(self, project, embedder):
        model = ScriptedModel(ModelTurn(text='done'))
        agentic_answer(project, 'q', embedder, model)
        system = model.systems[0].lower()

        for leak in ('round', 'budget', 'limit of', 'you have', 'attempts'):
            assert leak not in system

    def test_the_cap_is_a_small_number_of_rounds(self):
        assert 2 <= MAX_ROUNDS <= 12


class TestTheFinalAsk:
    def test_the_last_request_offers_no_tools(self, project, embedder):
        '''
        With nothing left to call, the model must answer from what it has
        rather than asking for more it cannot get.
        '''
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='the answer')
        )

        agentic_answer(project, 'q', embedder, model, max_rounds=1)

        assert model.offered[0]
        assert model.offered[-1] == []

    def test_it_answers_from_what_it_gathered(self, project, embedder):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='gathered answer')
        )

        answer = agentic_answer(project, 'q', embedder, model, max_rounds=1)

        assert answer.text == 'gathered answer'
        assert not answer.degraded


class TestStoppingEarly:
    def test_a_model_that_answers_immediately_is_not_pushed_further(
        self, project, embedder
    ):
        model = ScriptedModel(ModelTurn(text='I already know'))

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.text == 'I already know'
        assert len(model.offered) == 1

    def test_there_is_no_minimum_number_of_calls(self, project, embedder):
        model = ScriptedModel(ModelTurn(text='no retrieval needed'))

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.trace.calls == 0


class TestTheTrace:
    def test_every_call_is_recorded(self, project, embedder):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'}),
                  ('list_documents', {})),
            ModelTurn(text='done')
        )

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.trace.calls == 2
        assert answer.trace.tools_used == ['semantic_search', 'list_documents']

    def test_a_call_records_its_arguments(self, project, embedder):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='done')
        )

        answer = agentic_answer(project, 'q', embedder, model)

        assert 'aardvark' in answer.trace.entries[0].label

    def test_the_model_s_narration_is_kept(self, project, embedder):
        speaking = ModelTurn(
            text='Let me survey first.',
            calls=[ToolCall(id='c0', name='list_documents', arguments={})]
        )
        model = ScriptedModel(speaking, ModelTurn(text='done'))

        answer = agentic_answer(project, 'q', embedder, model)

        assert 'Let me survey first.' in answer.trace.narration

    def test_a_failing_call_is_recorded_not_raised(self, project, embedder):
        model = ScriptedModel(
            calls(('fetch_source', {'ref': '../escape.md'})),
            ModelTurn(text='done')
        )

        answer = agentic_answer(project, 'q', embedder, model)

        assert answer.trace.failures
        assert answer.text == 'done'

    def test_the_trace_reads_the_same_shape_whatever_the_provider(
        self, project, embedder
    ):
        '''
        Nothing provider-shaped reaches the trace, so one can be compared
        against another.
        '''
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='done')
        )

        answer = agentic_answer(project, 'q', embedder, model)
        lines = answer.trace.lines()

        assert lines[0] == 'round 1'
        assert 'semantic_search' in lines[1]

    def test_it_says_when_the_static_pipeline_answered(
        self, project, embedder
    ):
        answer = agentic_answer(project, 'q', embedder, ToollessModel())

        assert 'Static pipeline' in ' '.join(answer.trace.reading)


class TestToolResultsReachTheModel:
    def test_a_result_comes_back_as_json(self, project, embedder):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='done')
        )
        agentic_answer(project, 'q', embedder, model)

        exchange = model.histories[-1][0]
        payload = json.loads(list(exchange.results.values())[0])

        assert 'passages' in payload

    def test_an_error_comes_back_as_an_error_the_model_can_read(
        self, project, embedder
    ):
        model = ScriptedModel(
            calls(('no_such_tool', {})),
            ModelTurn(text='done')
        )
        agentic_answer(project, 'q', embedder, model)

        payload = json.loads(
            list(model.histories[-1][0].results.values())[0]
        )

        assert 'no tool named' in payload['error']

    def test_sources_are_gathered_from_what_the_tools_returned(
        self, project, embedder
    ):
        model = ScriptedModel(
            calls(('semantic_search', {'query': 'aardvark'})),
            ModelTurn(text='done')
        )

        answer = agentic_answer(project, 'q', embedder, model)

        assert any('alpha' in ref for ref in answer.sources)


class TestTheRegistry:
    def test_every_tool_has_a_handler(self, project, embedder):
        context = ToolContext(project=project, embedder=embedder)

        for name in TOOL_NAMES:
            assert 'no tool named' not in (
                run_tool(context, name, _minimal(name)).error or ''
            )

    def test_an_unknown_tool_reports_rather_than_raises(
        self, project, embedder
    ):
        context = ToolContext(project=project, embedder=embedder)

        result = run_tool(context, 'invented', {})

        assert 'no tool named' in result.error

    def test_a_wrong_argument_reports_rather_than_raises(
        self, project, embedder
    ):
        '''
        The model can read the problem and correct itself; a raised error
        would end the round.
        '''
        context = ToolContext(project=project, embedder=embedder)

        result = run_tool(context, 'semantic_search', {'nonsense': 1})

        assert result.error

    def test_the_survey_mode_is_offered_in_the_schema(self):
        spec = next(s for s in TOOL_SPECS if s.name == 'exact_search')

        assert 'refs' in spec.parameters['properties']['mode']['enum']

    def test_every_provider_is_offered_the_same_tools(self):
        '''
        Answer quality must not depend on which vendor was picked, so there is
        one table rather than a set per provider.
        '''
        assert len(TOOL_SPECS) == len(TOOL_NAMES) == 6


class TestTheSystemPrompt:
    def test_it_teaches_surveying_before_reading(self, project, embedder):
        model = ScriptedModel(ModelTurn(text='done'))
        agentic_answer(project, 'q', embedder, model)

        assert 'Survey before you read' in model.systems[0]

    def test_it_carries_today_s_date(self, project, embedder):
        from datetime import date

        model = ScriptedModel(ModelTurn(text='done'))
        agentic_answer(project, 'q', embedder, model)

        assert date.today().isoformat() in model.systems[0]

    def test_it_names_no_language(self, project, embedder):
        model = ScriptedModel(ModelTurn(text='done'))
        agentic_answer(project, 'q', embedder, model)
        system = model.systems[0].lower()

        for language in ('english', 'spanish', 'in english'):
            assert language not in system


def _minimal(name):
    return {
        'semantic_search': {'query': 'x'},
        'exact_search': {'terms': ['x']},
        'list_documents': {},
        'snowball': {'query': 'x', 'depth': 1},
        'graph_query': {'entity': 'x'},
        'fetch_source': {'ref': 'material/alpha.md'}
    }[name]
