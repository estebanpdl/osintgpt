# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: followups.py
# Description: Questions the retrieved material could answer next. The model
#   has just read what the analyst has not, and knows what went unpursued.
# =================================================================================

# import modules
import json
import logging
import re

# type hints
from typing import Any, List, Optional, Sequence

# import osintgpt llm
from osintgpt.llm.base import GenerationProvider

# import osintgpt prompts
from osintgpt.prompts import prompt

log = logging.getLogger('osintgpt.followups')

# Three is what fits under an answer without competing with it. More reads as
# a menu, and an analyst scanning a menu is not reading the answer.
DEFAULT_SUGGESTIONS = 3

# How much retrieved material the suggestion call sees. Enough to notice a
# thread the answer left alone, bounded so this stays the cheap call it is
# meant to be.
MAX_PASSAGES = 12
PASSAGE_CHARS = 500

# Recent questions shown to the model so it does not repeat one.
RECENT_QUESTIONS = 6


# propose what to ask next
def suggest_followups(
    generator: GenerationProvider,
    question: str,
    answer: str,
    passages: Sequence[Any],
    asked: Optional[Sequence[str]] = None,
    n: int = DEFAULT_SUGGESTIONS
) -> List[str]:
    '''
    Ask the model what this material could answer next.

    One extra generation call, made after an answer already succeeded. It
    fails soft in every direction: a provider error, an unparseable reply or
    nothing worth asking all return an empty list. Suggestions must never
    break an answer that worked.

    Args:
        generator (GenerationProvider): Writes the suggestions.
        question (str): What was just asked.
        answer (str): What was answered, so a suggestion does not restate it.
        passages (Sequence): The passages the answer drew on. Suggestions are \
            grounded in these, never in what the model happens to know.
        asked (Sequence[str], optional): Questions already put to this \
            project, so a suggestion does not repeat one.
        n (int): How many to propose.

    Returns:
        List[str]: Self-contained questions, possibly empty. Each is sent as \
            written — a CLI prints them numbered and an interface makes each \
            a button — so none may depend on this conversation.
    '''
    material = _material(passages)
    if not material:
        # Nothing was retrieved, so there is nothing to be curious about. A
        # model asked anyway would invent questions from its training.
        return []

    try:
        reply = generator.generate(
            prompt(
                'followups',
                n=max(int(n), 1),
                question=question,
                answer=(answer or '')[:1500],
                passages=material,
                asked=list(asked or [])[-RECENT_QUESTIONS:]
            ),
            'Propose the follow-up questions.'
        )
    except Exception as error:  # noqa: BLE001 — never break a good answer
        log.warning('follow-up suggestions failed: %s', error)

        return []

    return _parse(reply)[:max(int(n), 1)]


def _material(passages: Sequence[Any]) -> List[dict]:
    '''
    The passages as the prompt wants them, bounded and deduplicated.
    '''
    seen = set()
    rows = []

    for item in list(passages)[:MAX_PASSAGES * 2]:
        citation, text = _describe(item)
        if not text or citation in seen:
            continue
        seen.add(citation)
        rows.append({'citation': citation, 'text': text[:PASSAGE_CHARS]})
        if len(rows) >= MAX_PASSAGES:
            break

    return rows


def _describe(item: Any):
    '''
    Read a passage from whichever shape it arrived in.

    The static path carries SearchResult objects and the agentic path carries
    the payload dictionaries the tools returned, and suggestions should work
    after either.
    '''
    if isinstance(item, dict):
        return (
            str(item.get('citation') or item.get('ref') or ''),
            str(item.get('text') or '')
        )

    chunk = getattr(item, 'chunk', None)
    if chunk is not None:
        return str(chunk.citation), str(getattr(item, 'text', '') or '')

    return '', str(item or '')


def _parse(reply: str) -> List[str]:
    '''
    A JSON array out of a reply, tolerating prose or a code fence around it.
    '''
    match = re.search(r'\[.*\]', (reply or '').strip(), re.DOTALL)
    if not match:
        return []

    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return []

    if not isinstance(parsed, list):
        return []

    seen = []
    for item in parsed:
        text = str(item).strip() if isinstance(item, str) else ''
        if text and text not in seen:
            seen.append(text)

    return seen
