# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: chat.py
# Description: Asking, and seeing how the answer was reached. Every claim
#   traces to a passage the analyst can open.
# =================================================================================

# type hints
from typing import Any, Dict, List

# import osintgpt
from osintgpt import agentic_answer

from ..session import queue_question, remember, take_pending

# Enough of a passage to judge it without the chip becoming the page.
PREVIEW_CHARS = 900


# the passages an answer actually read
def passages_of(answer: Any) -> List[Dict[str, str]]:
    '''
    Pull the passages out of an answer's trace.

    A source chip an analyst cannot open is decoration, and the text is
    already in the trace — the tools returned it — so nothing needs
    retrieving twice to show it.

    Args:
        answer (AgenticAnswer): The answer.

    Returns:
        List[Dict[str, str]]: One entry per passage, deduplicated by citation.
    '''
    found: List[Dict[str, str]] = []
    seen = set()

    for entry in getattr(answer.trace, 'entries', []):
        for item in getattr(entry, 'payload', {}).get('passages', []) or []:
            citation = item.get('citation') or item.get('ref') or ''
            if not citation or citation in seen:
                continue
            seen.add(citation)
            found.append({
                'citation': citation,
                'ref': item.get('ref', citation),
                'text': str(item.get('text', ''))[:PREVIEW_CHARS]
            })

    return found


# render the chat view
def render(st, runtime, state) -> None:
    '''
    Ask a question and show the answer, its sources, and how it was reached.

    Args:
        st: The Streamlit module.
        runtime (Runtime): Project and providers.
        state: Session state.
    '''
    st.subheader(f'Ask — {runtime.project.name}')

    for turn in state.get('chat_history', []):
        _turn(st, turn['question'], turn['answer'], state, replayed=True)

    question = take_pending(state) or st.chat_input('Ask about this project')
    if not question:
        return

    with st.spinner('Searching…'):
        answer = agentic_answer(
            runtime.project, question, runtime.embedder, runtime.generator
        )

    remember(state, question, answer)
    _turn(st, question, answer, state, replayed=False)


def _turn(st, question, answer, state, replayed: bool) -> None:
    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):
        st.write(answer.text)

        if getattr(answer, 'degraded', ''):
            # An answer from the static path is still an answer, and which
            # one the analyst got changes how much weight it carries.
            st.caption(f'Answered without tools: {answer.degraded}')

        _sources(st, answer)
        _trace(st, answer)
        if not replayed:
            _followups(st, answer, state)


def _sources(st, answer) -> None:
    passages = passages_of(answer)
    if not passages:
        if answer.sources:
            st.caption('Sources: ' + ', '.join(answer.sources))

        return

    st.caption(f'{len(passages)} sources')
    for passage in passages:
        with st.expander(passage['citation']):
            st.write(passage['text'])


def _trace(st, answer) -> None:
    lines = answer.trace.lines()
    if not lines:
        return

    with st.expander('How I searched'):
        for line in lines:
            st.text(line)
        for note in answer.trace.reading:
            st.caption(note)


def _followups(st, answer, state) -> None:
    '''
    Each suggestion is a complete question, which is why a click can submit it
    unchanged rather than having to reconstruct what it referred to.
    '''
    if not answer.followups:
        return

    st.caption('Ask next')
    for index, suggestion in enumerate(answer.followups):
        st.button(
            suggestion,
            key=f'followup-{len(state.get("chat_history", []))}-{index}',
            on_click=queue_question,
            args=(state, suggestion)
        )
