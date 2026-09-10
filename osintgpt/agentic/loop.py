# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: loop.py
# Description: The model drives retrieval. Which tool to call and when is its
#   decision; this only runs what it asks for and records what happened.
# =================================================================================

# import modules
import json
import logging
import time

# import submodules
from dataclasses import dataclass, field
from datetime import date

# type hints
from typing import Any, Callable, Dict, List, Optional

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider
from osintgpt.llm.calling import (
    Exchange,
    ModelTurn,
    ToolCallingUnsupported,
    ToolSpec
)

# import osintgpt prompts
from osintgpt.prompts import prompt

from .registry import TOOL_SPECS, run_tool
from .tools import ToolContext
from .trace import Trace

log = logging.getLogger('osintgpt.agentic')

# Six rounds was enough in practice. A round may hold several parallel calls,
# so this is not a step budget. The model is never told the number exists:
# knowing a cap invites spending it rather than stopping when the answer is
# ready.
MAX_ROUNDS = 6


# AgenticAnswer class
@dataclass(frozen=True)
class AgenticAnswer:
    '''
    An answer, and everything the model did to reach it.
    '''
    question: str
    text: str
    trace: Trace = field(default_factory=Trace)
    # Refs the tools actually returned, in the order they were first seen.
    # What a reader opens to check the answer.
    sources: List[str] = field(default_factory=list)
    # What to ask next, each self-contained so it can be sent as written.
    followups: List[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.trace.degraded)


# let the model retrieve and answer
def agentic_answer(
    project,
    question: str,
    embedder: EmbeddingProvider,
    generator: GenerationProvider,
    max_rounds: int = MAX_ROUNDS,
    store: Optional[Any] = None,
    on_round: Optional[Callable[[int, ModelTurn], None]] = None
) -> AgenticAnswer:
    '''
    Give the model the retrieval tools and let it decide what to call.

    A backend that cannot call tools falls back to the static pipeline rather
    than failing, and the trace says so. An empty answer is structurally
    impossible: when the rounds run out the model is asked once more with no
    tools offered, so it must reply from what it has.

    Args:
        project (Project): The project to answer from.
        question (str): The question, as asked.
        embedder (EmbeddingProvider): Must be the model the project was \
            indexed with.
        generator (GenerationProvider): Drives the tools and writes the answer.
        max_rounds (int): Most rounds of tool calling before the final ask.
        store (BaseVectorEngine, optional): Defaults to the project's own.
        on_round (Callable, optional): Called with the round number and what \
            the model produced, for progress.

    Returns:
        AgenticAnswer: The answer, its sources, and the trace.
    '''
    from osintgpt.projects.questions import record_question

    record_question(project, question)

    trace = Trace()
    context = ToolContext(
        project=project, embedder=embedder, generator=generator, store=store
    )

    if _nothing_indexed(project, store):
        # No call is made, of any kind. A corpus with nothing in it cannot
        # ground an answer, and a model given tools that can only return
        # nothing would answer from its training instead.
        from osintgpt.answering import NOTHING_RETRIEVED

        trace.degraded = 'the project has nothing indexed'

        return AgenticAnswer(
            question=question, text=NOTHING_RETRIEVED, trace=trace
        )

    if not generator.supports_tools:
        return _static(project, question, embedder, generator, trace, store,
                       'the model does not support tool calling')

    system = prompt('agentic', today=date.today().isoformat())
    history: List[Exchange] = []
    sources: List[str] = []
    # Passages the tools actually returned, so a suggestion is grounded in
    # what the model read rather than in what it might have read.
    gathered: List[dict] = []

    for round_number in range(1, max(int(max_rounds), 1) + 1):
        try:
            turn = generator.generate_with_tools(
                system, question, TOOL_SPECS, history
            )
        except ToolCallingUnsupported as error:
            return _static(project, question, embedder, generator, trace,
                           store, str(error))
        except Exception as error:  # noqa: BLE001 — degrade, do not fail
            log.warning('tool round %d failed: %s', round_number, error)

            return _static(project, question, embedder, generator, trace,
                           store, f'tool calling failed: {error}')

        trace.say(turn.text)
        if on_round:
            on_round(round_number, turn)

        if not turn.wants_tools:
            # It answered. Nothing here decides it should have kept looking.
            return _answered(
                project, generator, question, turn.text.strip(), trace,
                sources, gathered
            )

        results = _run_calls(
            context, turn, trace, round_number, sources, gathered
        )
        history.append(Exchange(turn=turn, results=results))

    # Rounds exhausted. One more request with no tools offered, so the model
    # has no way to ask for more and must answer from what it gathered.
    try:
        final = generator.generate_with_tools(system, question, [], history)
        text = final.text.strip()
    except Exception as error:  # noqa: BLE001 — an answer is still owed
        log.warning('final round failed: %s', error)
        text = ''

    if not text:
        return _static(project, question, embedder, generator, trace, store,
                       'the model gathered material but produced no answer')

    return _answered(
        project, generator, question, text, trace, sources, gathered
    )


def _answered(project, generator, question, text, trace, sources, gathered):
    '''
    Attach suggestions to an answer the model produced itself.
    '''
    from osintgpt.answering import followups_for

    return AgenticAnswer(
        question=question, text=text, trace=trace, sources=sources,
        followups=followups_for(
            project, generator, question, text, gathered
        )
    )


def _nothing_indexed(project, store) -> bool:
    '''
    Whether the project holds any vectors at all.

    Checked before the generator is touched, so an unindexed project costs
    nothing rather than a model call that could only be answered ungrounded.
    '''
    from osintgpt.vector_store import store_for

    owned = store is None
    engine = store or store_for(project)
    try:
        return engine.count() == 0
    except Exception:  # noqa: BLE001 — a guard, not the answer
        return False
    finally:
        if owned and hasattr(engine, 'close'):
            engine.close()


def _run_calls(
    context, turn, trace, round_number, sources, gathered
) -> Dict[str, str]:
    '''
    Run everything the model asked for this round, recording each.
    '''
    results: Dict[str, str] = {}

    for call in turn.calls:
        started = time.perf_counter()
        try:
            result = run_tool(context, call.name, call.arguments)
            error = result.error
            count = result.count
            payload = result.payload
        except Exception as failure:  # noqa: BLE001 — one call, not the loop
            error, count, payload = str(failure), 0, {}

        elapsed = time.perf_counter() - started
        trace.record(
            round_number, call.name, call.arguments,
            count=count, seconds=elapsed, error=error
        )

        for ref in _refs_in(payload):
            if ref not in sources:
                sources.append(ref)

        gathered.extend(payload.get('passages') or [])

        results[call.id] = json.dumps(
            {'error': error} if error else payload,
            ensure_ascii=False, default=str
        )

    return results


def _refs_in(payload: Dict[str, Any]) -> List[str]:
    '''
    Every document a payload mentions, wherever the tool put it.
    '''
    found = []
    for key in ('passages', 'documents', 'hops', 'claims', 'path'):
        for item in payload.get(key, []) or []:
            ref = item.get('ref') if isinstance(item, dict) else None
            if ref:
                found.append(ref)

    if payload.get('ref'):
        found.append(payload['ref'])

    return found


def _static(project, question, embedder, generator, trace, store, reason):
    '''
    The pipeline that answers when the loop cannot.

    Retrieve once, answer from what came back. It is what a model without tool
    calling gets, and what everything falls back to — an empty answer has to
    be structurally impossible, not merely unlikely.
    '''
    from osintgpt.answering import answer_question

    trace.degraded = reason
    log.info('falling back to the static pipeline: %s', reason)

    # record=False: the question was logged before the loop began, and
    # degrading is not asking again.
    answer = answer_question(
        project, question, embedder, generator, store=store, record=False
    )

    return AgenticAnswer(
        question=question, text=answer.text, trace=trace,
        sources=[r.ref for r in answer.passages],
        followups=answer.followups
    )
