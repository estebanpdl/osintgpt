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

# import submodules
from collections import defaultdict

# type hints
from typing import Any, Dict, List

# import osintgpt
from osintgpt import agentic_answer

# import osintgpt projects
from osintgpt.projects import recent_pairs

from ..session import (
    CONVERSATION,
    current_conversation,
    queue_question,
    remember,
    take_pending
)
from ..styles import badge, escape, escape_html

# Streamlit's own defaults are a red and an orange plate, and red is reserved
# here for an answer that cannot be trusted. Named icons render in body text
# colour instead, so the two sides are told apart by glyph.
USER_AVATAR = ':material/person:'
ASSISTANT_AVATAR = ':material/robot_2:'

# Enough of a passage to judge it without the chip becoming the page.
PREVIEW_CHARS = 900

# Documents listed under one call before the rest are counted instead. A
# survey tool can return hundreds, and a list that long stops being read.
MAX_TRACE_DOCUMENTS = 40

# A score is only readable next to what produced it. The semantic tools rank
# by cosine against the query vector; exact search ranks by how many of the
# searched terms a chunk contains, which is a share and not a similarity.
SIMILARITY = 'Best passage similarity in this call using cosine similarity.'
COVERAGE = (
    'Best passage score in this call: the share of the searched terms it '
    'contains.'
)
SCORE_MEANS = defaultdict(lambda: SIMILARITY, {'exact_search': COVERAGE})

# The refs survey has no score to show, so its count is what it ranked on.
# Chunks, not occurrences: forty mentions inside one chunk count once.
MATCHES_MEAN = 'Chunks containing at least one of the searched terms.'


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
    from osintgpt.agentic import answer_from_dict

    project = runtime.project
    conversation = current_conversation(state, project)
    st.subheader(f'Ask — {project.name}')
    if conversation is not None and len(conversation):
        st.caption(conversation.name)

    turns = conversation.turns if conversation else []
    for index, turn in enumerate(turns):
        _turn(
            st, turn.question, answer_from_dict(turn.answer), state,
            index=index, suggest=index == len(turns) - 1
        )

    question = take_pending(state) or st.chat_input('Ask about this project')
    if not question:
        return

    with st.spinner('Searching…'):
        answer = agentic_answer(
            project, question, runtime.embedder, runtime.generator,
            # Read before the turn is recorded, so the window holds what came
            # before this question and never the question itself.
            conversation=recent_pairs(
                conversation, project.settings.conversation_window
            )
        )

    remember(state, project, question, answer)
    # `turns` was read before the answer was appended, so the new turn takes
    # the position after the last one replayed above.
    _turn(st, question, answer, state, index=len(turns), suggest=True)


def _turn(st, question, answer, state, index: int, suggest: bool) -> None:
    with st.chat_message('user', avatar=USER_AVATAR):
        st.write(question)

    with st.chat_message('assistant', avatar=ASSISTANT_AVATAR):
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
        if suggest:
            _followups(st, answer, state, index)


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

        if not entry.documents:
            rows.append(f'<div class="trace-call trace-flat">{head}</div>')
            continue

        rows.append(
            '<details class="trace-call">'
            f'<summary>{head}</summary>'
            '<div class="trace-detail">'
            f'{_documents(entry.documents, entry.tool)}</div>'
            '</details>'
        )

    return ''.join(rows)


def _documents(documents, tool: str = '') -> str:
    '''
    The documents a call touched, each with what the call measured about it.

    The full ref, not the shortened one the argument line carries: this is the
    part of the trace an analyst checks an answer against, and a path they
    cannot copy is a path they cannot open.

    Args:
        documents (Sequence[TouchedDocument]): Documents, in the order \
            returned — which is the order the call ranked them in.
        tool (str): Which tool produced them, because the score means a \
            different thing depending on which one did.

    Returns:
        str: The list as HTML.
    '''
    shown = documents[:MAX_TRACE_DOCUMENTS]
    rest = len(documents) - len(shown)

    items = ''.join(
        '<div class="trace-docfile">'
        f'<span class="trace-doc">{escape_html(document.ref)}</span>'
        f'{_measures(document, tool)}'
        '</div>'
        for document in shown
    )
    more = (
        f'<span class="trace-more">+{rest} more</span>' if rest > 0 else ''
    )
    label = 'document' + ('' if len(documents) == 1 else 's')

    return (
        f'<div class="trace-docs-label">{len(documents)} {label}</div>'
        f'{items}{more}'
    )


def _measures(document, tool: str = '') -> str:
    '''
    What the call learned about one document, as a row beneath its ref.

    A document the call only listed gets no row at all rather than an empty
    one — `list_documents` reports existence, and a blank line under every ref
    would imply it had measured something and found nothing.
    '''
    if not document.scored:
        return ''

    parts = []

    if document.best is not None:
        parts.append(
            f'<span class="trace-meta-score" title="{SCORE_MEANS[tool]}">'
            f'{document.best:.2f}</span>'
        )

    if document.passages:
        unit = 'passage' + ('' if document.passages == 1 else 's')
        parts.append(f'<span>{document.passages} {unit}</span>')

    if document.matches:
        unit = 'match' + ('' if document.matches == 1 else 'es')
        parts.append(
            f'<span class="trace-meta-counted" title="{MATCHES_MEAN}">'
            f'{document.matches} {unit}</span>'
        )

    for value in (document.section, document.timestamp, document.author):
        if value:
            parts.append(f'<span>{escape_html(value)}</span>')

    parts += [
        f'<span class="trace-meta-term">{escape_html(term)}</span>'
        for term in document.terms
    ]

    return f'<span class="trace-doc-meta">{"".join(parts)}</span>'


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


def _followups(st, answer, state, index: int) -> None:
    '''
    Each suggestion is a complete question, which is why a click can submit it
    unchanged rather than having to reconstruct what it referred to.

    Offered on the newest turn, whether it was just answered or read back from
    the transcript — where they were written with the answer. Suggestions
    outlive the rerun that produced them, so leaving the view, refreshing, or
    restarting the server does not take them; an older turn still offers none,
    because those questions have been moved past.
    '''
    if not answer.followups:
        return

    # Keyed on the turn's position rather than on the question: two identical
    # questions in one thread would share a key, and a key reused across turns
    # carries the previous click.
    turn = f'{state.get(CONVERSATION, "new")}-{index}'

    st.caption('Ask next')
    for number, suggestion in enumerate(answer.followups):
        # The key reaches the DOM as `st-key-<key>`, which is how these are
        # styled. Streamlit renders each element in a container of its own, so
        # a wrapper div never encloses the widgets written after it.
        st.button(
            suggestion,
            key=f'followup-{turn}-{number}',
            on_click=queue_question,
            args=(state, suggestion)
        )
