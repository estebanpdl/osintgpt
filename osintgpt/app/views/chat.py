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
from ..styles import badge, escape, escape_html

# Enough of a passage to judge it without the chip becoming the page.
PREVIEW_CHARS = 900

# Documents listed under one call before the rest are counted instead. A
# survey tool can return hundreds, and a list that long stops being read.
MAX_TRACE_DOCUMENTS = 40


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
            # one the analyst got changes how much weight it carries. Amber,
            # not red: it worked, just not the way it should have.
            st.markdown(
                badge('answered without tools', 'partial')
                + f' <span class="citation-chip">{escape(answer.degraded)}</span>',
                unsafe_allow_html=True
            )

        _sources(st, answer)
        _trace(st, answer)
        if not replayed:
            _followups(st, answer, state)


def _sources(st, answer) -> None:
    passages = passages_of(answer)
    if not passages:
        if answer.sources:
            st.caption(
                'Sources: '
                + ', '.join(escape(ref) for ref in answer.sources)
            )

        return

    st.caption(f'{len(passages)} sources')
    for passage in passages:
        with st.expander(escape(passage['citation'])):
            st.write(passage['text'])


def _trace(st, answer) -> None:
    trace = answer.trace
    if not (trace.entries or trace.narration):
        return

    with st.expander('How I searched'):
        for number in trace.round_numbers:
            said = trace.said_in(number)
            calls = _calls(trace.calls_in(number))

            # Markdown inside raw HTML is not parsed again, so the model's
            # words cannot share a block with the calls. A round that spoke
            # pays for two more elements; one that did not stays a single
            # block, which is most of them.
            if said:
                st.markdown(_round(number), unsafe_allow_html=True)
                for line in said:
                    st.markdown(_quoted(line.text))
                st.markdown(calls, unsafe_allow_html=True)
            else:
                st.markdown(_round(number) + calls, unsafe_allow_html=True)

        for note in trace.reading:
            st.caption(note)


def _round(number: int) -> str:
    return f'<div class="trace-round">Round {number}</div>'


def _calls(entries) -> str:
    '''
    One row per call: what ran, what it returned, and how long it took —
    opening onto the documents it touched.

    A call that touched nothing stays a plain row. An affordance that opens
    onto an empty list is worse than no affordance: it invites the one click
    that proves there was nothing to see.

    Args:
        entries (List[TraceEntry]): The round's calls, in order.

    Returns:
        str: The rows as HTML.
    '''
    rows = []

    for entry in entries:
        failed = '' if entry.ok else ' trace-failed'
        outcome = entry.counted if entry.ok else entry.error
        arguments = entry.arguments_line

        head = (
            '<div class="trace-head">'
            f'<span class="trace-tool">{escape_html(entry.tool)}</span>'
            f'<span class="trace-count{failed}">{escape_html(outcome)}</span>'
            f'<span class="trace-time">{entry.seconds:.2f}s</span>'
            '</div>'
            + (
                f'<div class="trace-args">{escape_html(arguments)}</div>'
                if arguments else ''
            )
        )

        if not entry.refs:
            rows.append(f'<div class="trace-call trace-flat">{head}</div>')
            continue

        rows.append(
            '<details class="trace-call">'
            f'<summary>{head}</summary>'
            f'<div class="trace-detail">{_documents(entry.refs)}</div>'
            '</details>'
        )

    return ''.join(rows)


def _documents(refs) -> str:
    '''
    The documents a call touched, by the ref the index stores.

    The full ref, not the shortened one the argument line carries: this is the
    part of the trace an analyst checks an answer against, and a path they
    cannot copy is a path they cannot open.

    Args:
        refs (Sequence[str]): Document refs, in the order returned.

    Returns:
        str: The list as HTML.
    '''
    shown = refs[:MAX_TRACE_DOCUMENTS]
    rest = len(refs) - len(shown)

    items = ''.join(
        f'<span class="trace-doc">{escape_html(ref)}</span>' for ref in shown
    )
    more = (
        f'<span class="trace-more">+{rest} more</span>' if rest > 0 else ''
    )
    label = 'document' + ('' if len(refs) == 1 else 's')

    return (
        f'<div class="trace-docs-label">{len(refs)} {label}</div>'
        f'{items}{more}'
    )


def _quoted(text: str) -> str:
    '''
    The model's words as a blockquote, so they read as something said rather
    than as another line of machinery. Blank lines keep the quote unbroken.

    Args:
        text (str): What the model said, in its own markdown.

    Returns:
        str: The same text, every line quoted.
    '''
    return '\n'.join(
        f'> {line}' if line.strip() else '>'
        for line in text.splitlines()
    )


def _followups(st, answer, state) -> None:
    '''
    Each suggestion is a complete question, which is why a click can submit it
    unchanged rather than having to reconstruct what it referred to.
    '''
    if not answer.followups:
        return

    st.caption('Ask next')
    st.markdown('<div class="followup-row">', unsafe_allow_html=True)
    for index, suggestion in enumerate(answer.followups):
        st.button(
            suggestion,
            key=f'followup-{len(state.get("chat_history", []))}-{index}',
            on_click=queue_question,
            args=(state, suggestion)
        )
    st.markdown('</div>', unsafe_allow_html=True)
