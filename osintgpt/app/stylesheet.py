# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: stylesheet.py
# Description: The CSS the app injects. Separated from the helpers that use it
#   because a stylesheet grows by rule and the helpers do not.
# =================================================================================

STYLESHEET = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@500;600&display=swap');

:root {
    --accent-primary: #a855f7;
    --accent-secondary: #e879f9;
    --accent-cool: #9db4ff;
    --graphite: #0f0c17;
    --graphite-deep: #0a0810;
    --graphite-raised: #191421;
    --glass: rgba(255, 255, 255, 0.09);
    --glass-strong: rgba(255, 255, 255, 0.14);
    --border-faint: rgba(255, 255, 255, 0.18);
    --border-soft: rgba(255, 255, 255, 0.28);
    --text-bright: #f5f2fa;
    --text-secondary: #a39ab0;

    /* Reserved — meaning, not decoration. */
    --status-good: #10b981;
    --status-partial: #f59e0b;
    --status-problem: #ef4444;
}

/* The whole viewport wears the atmosphere: layered aurora glows over a
   graphite ground, in the mark's violet family plus a cool blue. None of
   these colours answer a question an analyst is asking, so none is held
   back. */
html, body, [data-testid="stAppViewContainer"] {
    background: #0c0a12;
    color: var(--text-bright);
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 85% 55% at 12% -12%, rgba(168, 85, 247, 0.17), transparent 55%),
        radial-gradient(ellipse 60% 45% at 102% 12%, rgba(157, 180, 255, 0.10), transparent 55%),
        radial-gradient(ellipse 100% 60% at 50% 118%, rgba(232, 121, 249, 0.09), transparent 60%),
        linear-gradient(180deg, #130f1e 0%, #0a0810 100%);
}

.main .block-container {
    max-width: 1150px;
    padding: 1.4rem 2.4rem 3.5rem;
    color: var(--text-bright);
    font-family: 'Inter', sans-serif;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(168, 85, 247, 0.28); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.45); }
::-webkit-scrollbar-track { background: transparent; }

::selection { background: rgba(168, 85, 247, 0.35); }

/* The mark. Sora is set apart from the working type, and carries the same
   gradient as the primary buttons so the brand and the call to action agree. */
.osintgpt-title {
    font-family: 'Sora', 'Inter', sans-serif;
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin-bottom: 0.2rem;
    background: linear-gradient(100deg, var(--accent-primary) 0%, var(--accent-secondary) 60%, var(--accent-cool) 115%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 16px rgba(168, 85, 247, 0.3));
}

.osintgpt-subtitle {
    color: var(--text-secondary);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-faint);
}

/* View titles draw a hairline under themselves; section labels become quiet
   trackers. Hierarchy comes from size and weight, not from asking every
   heading to shout. */
[data-testid="stHeading"] h2 {
    color: var(--text-bright);
    font-weight: 600;
    letter-spacing: -0.015em;
    border-bottom: 1px solid var(--border-faint);
    padding-bottom: 0.3rem;
    margin-bottom: 1rem;
}

.stMarkdown h4 {
    color: var(--accent-secondary);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.6rem 0 0.6rem;
}

/* One voice per job: only a real call to action earns the gradient, and a
   default button or a follow-up turning orange-edged on every hover would
   drown the one button that matters. */
.stButton > button:not([data-testid="stBaseButton-primary"]):not([kind="primary"]) {
    background: var(--glass);
    color: var(--text-bright);
    border: 1px solid var(--border-faint);
    border-radius: 16px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease, box-shadow 0.25s ease;
}

.stButton > button:not([data-testid="stBaseButton-primary"]):not([kind="primary"]):hover {
    background: var(--glass-strong);
    border-color: var(--border-soft);
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
}

.stButton > button:active {
    transform: translateY(0);
}

[data-testid="stBaseButton-primary"] > button,
button[kind="primary"] {
    background: linear-gradient(120deg, var(--accent-primary), var(--accent-secondary));
    color: #0c0a12;
    border: none;
    font-weight: 600;
    box-shadow:
        0 8px 26px rgba(168, 85, 247, 0.42),
        inset 0 2px 0 rgba(255, 255, 255, 0.45);
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.25s ease, filter 0.2s ease;
}

[data-testid="stBaseButton-primary"] > button:hover,
button[kind="primary"]:hover {
    filter: brightness(1.07);
    transform: translateY(-2px);
    box-shadow:
        0 12px 32px rgba(168, 85, 247, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.45);
}

.stButton > button:focus-visible,
[data-testid="stBaseButton-primary"] > button:focus-visible {
    outline: 2px solid var(--accent-secondary);
    outline-offset: 2px;
}

/* A follow-up is a question, not a call to action. They sit under the answer
   they belong to, against the main page, so they wear the quietest surface.

   Matched on the widget key, which Streamlit puts in the DOM as
   `st-key-<key>`: each element renders in a container of its own, so a
   wrapper div written into a markdown block never encloses what follows it. */
[class*="st-key-followup-"] .stButton > button {
    background: rgba(255, 255, 255, 0.025);
    color: var(--text-secondary);
    border: 1px solid var(--border-faint);
    border-radius: 999px;
    font-weight: 400;
    font-size: 0.85rem;
    justify-content: flex-start;
    text-align: left;
    padding: 0.4rem 1.05rem;
    box-shadow: none;
}

[class*="st-key-followup-"] .stButton > button:hover {
    background: rgba(168, 85, 247, 0.1);
    border-color: var(--accent-primary);
    color: var(--text-bright);
    transform: none;
    box-shadow: none;
}

/* Fields: glass where the analyst types, a violet ring while they are in it. */
.stTextInput > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: 12px;
}

.stTextInput [data-baseweb="input"],
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--glass);
    border: 1px solid var(--border-faint);
    border-radius: 12px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 8px rgba(0,0,0,0.25);
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.stTextInput [data-baseweb="input"]:focus-within {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.22), inset 0 1px 0 rgba(255,255,255,0.1);
    background: rgba(168, 85, 247, 0.07);
}

.stTextInput input,
.stNumberInput input,
[data-testid="stChatInput"] textarea {
    color: var(--text-bright);
    caret-color: var(--accent-secondary);
}

input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.3);
}

[data-testid="stRadio"] label {
    border-radius: 10px;
    padding: 0.3rem 0.55rem;
    margin: 0.05rem 0;
    transition: background 0.15s ease;
}

[data-testid="stRadio"] label:hover {
    background: var(--glass);
}

input[type="radio"], input[type="checkbox"] {
    accent-color: var(--accent-primary);
}

.stProgress > div > div {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 999px;
    overflow: hidden;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 999px;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.5);
}

/* Cards carry a top edge of light and a soft drop, so a raised surface reads
   as raised against the aurora behind it rather than as a flat box. */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--glass) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid var(--border-faint);
    border-radius: 16px;
    padding: 0.7rem 1.1rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 6px 18px rgba(0, 0, 0, 0.28);
}

[data-testid="stMetricValue"] {
    color: var(--text-bright);
    font-size: 1.9rem;
    font-weight: 600;
    background: linear-gradient(180deg, #ffffff 0%, var(--accent-secondary) 130%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

[data-testid="stMetricDelta"] {
    color: var(--accent-cool);
}

.stExpander {
    background: linear-gradient(180deg, var(--glass) 0%, rgba(255, 255, 255, 0.015) 100%);
    border: 1px solid var(--border-faint);
    border-radius: 14px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 4px 18px rgba(0, 0, 0, 0.25);
    overflow: hidden;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] button {
    color: var(--text-bright);
    font-weight: 500;
    background: transparent;
    border: none;
    box-shadow: none;
}

[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] button:hover {
    background: rgba(255, 255, 255, 0.03);
}

hr {
    border-top-width: 1px;
    border-color: var(--border-faint);
    margin: 1.6rem 0;
}

.stMarkdown p,
.stAlert p {
    color: var(--text-bright);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border-faint);
    backdrop-filter: blur(3px);
}

[data-testid="stCaptionContainer"] p,
.stCaptionContent p {
    color: var(--text-secondary);
    font-size: 0.8rem;
    letter-spacing: 0.01em;
}

/* The search trace is machinery, so it reads like machinery: monospace,
   recessed, and tuned down so the prose answer stays the loudest thing.
   Written as one block per round rather than one element per line, because
   Streamlit puts a paragraph gap between elements and that turned a dense
   record into a column of floating sentences. */
.trace-round {
    color: var(--text-secondary);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0.9rem 0 0.45rem;
}

.trace-round:first-child {
    margin-top: 0;
}

.trace-call {
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border-faint);
    border-radius: 10px;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.3rem;
}

/* A call that touched documents opens onto them; one that touched none is
   the same box without the affordance. */
details.trace-call > summary {
    display: block;
    cursor: pointer;
    list-style: none;
}

details.trace-call > summary::-webkit-details-marker {
    display: none;
}

details.trace-call:hover {
    border-color: var(--border-soft);
}

.trace-head::before {
    content: '▸';
    color: var(--text-secondary);
    font-size: 0.62rem;
    line-height: 1;
}

details.trace-call[open] > summary .trace-head::before {
    content: '▾';
}

/* The flat row keeps the marker's width so tool names stay on one left edge
   whether or not a call can be opened. */
.trace-flat .trace-head::before {
    content: '';
    width: 0.62rem;
}

/* Tool, outcome and elapsed time on one row: the three things being compared
   across calls line up, and the arguments drop beneath rather than pushing
   them out of alignment. */
.trace-head {
    display: flex;
    align-items: baseline;
    gap: 0.65rem;
}

.trace-tool {
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--accent-secondary);
}

.trace-count {
    font-size: 0.76rem;
    color: var(--text-bright);
}

.trace-count.trace-failed {
    color: var(--status-problem);
}

.trace-time {
    margin-left: auto;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
}

.trace-args {
    margin-top: 0.22rem;
    padding-left: 1.27rem;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.74rem;
    line-height: 1.5;
    color: var(--text-secondary);
    overflow-wrap: anywhere;
}

/* What the call actually read. The refs are full and selectable, because
   this is the half of the trace an analyst checks an answer against. */
.trace-detail {
    margin-top: 0.5rem;
    padding: 0.5rem 0 0.15rem 1.27rem;
    border-top: 1px solid var(--border-faint);
}

.trace-docs-label {
    color: var(--text-secondary);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.trace-doc {
    display: block;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.74rem;
    line-height: 1.6;
    color: var(--text-bright);
    overflow-wrap: anywhere;
    user-select: text;
}

.trace-more {
    display: block;
    margin-top: 0.25rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

/* The model's own words, quoted rather than boxed — they are prose, and the
   calls around them are not. */
[data-testid="stExpander"] blockquote {
    border-left: 2px solid var(--border-soft);
    margin: 0.1rem 0 0.55rem;
    padding: 0 0 0 0.85rem;
    color: var(--text-secondary);
    font-size: 0.85rem;
}

[data-testid="stChatMessage"] {
    background: linear-gradient(180deg, var(--glass) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid var(--border-faint);
    border-radius: 18px;
    padding: 0.3rem 1rem 0.3rem 0.2rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-bottom: 0.5rem;
}

[data-testid="stChatInput"] {
    background: var(--glass);
    border: 1px solid var(--border-faint);
    border-radius: 18px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 4px rgba(168, 85, 247, 0.22);
    background: rgba(168, 85, 247, 0.07);
}

[data-testid="stSpinner"] div p {
    color: var(--text-secondary);
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse 90% 40% at -10% 0%, rgba(168, 85, 247, 0.13), transparent 60%),
        var(--graphite);
    border-right: 1px solid var(--border-faint);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.4rem 1.1rem 2rem;
}

/* A quiet tracker over a sidebar section, the same voice the settings
   headings use. */
.sidebar-label {
    color: var(--text-secondary);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.5rem;
}

/* A conversation is somewhere to go back to, not an action. They wear the
   quietest surface available and sit flush, so a list of twenty reads as a
   list rather than as twenty buttons. Matched on the widget key. */
[class*="st-key-conv-"] .stButton > button {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    color: var(--text-secondary);
    font-weight: 400;
    font-size: 0.84rem;
    justify-content: flex-start;
    text-align: left;
    padding: 0.35rem 0.6rem;
    box-shadow: none;
}

[class*="st-key-conv-"] .stButton > button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

[class*="st-key-conv-"] .stButton > button:hover {
    background: var(--glass);
    border-color: var(--border-faint);
    color: var(--text-bright);
    transform: none;
    box-shadow: none;
}

/* The open thread keeps the violet edge, so the one on screen is findable in
   the list without reading it. Later than the rule above, so it wins. */
[class*="st-key-conv-open-"] .stButton > button {
    background: rgba(168, 85, 247, 0.12);
    border-color: var(--accent-primary);
    color: var(--text-bright);
}

.status-badge {
    display: inline-block;
    padding: 0.16rem 0.75rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    vertical-align: middle;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}

.status-good {
    background: rgba(16, 185, 129, 0.16);
    color: var(--status-good);
    border: 1px solid rgba(16, 185, 129, 0.38);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}

.status-partial {
    background: rgba(245, 158, 11, 0.16);
    color: var(--status-partial);
    border: 1px solid rgba(245, 158, 11, 0.38);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}

.status-problem {
    background: rgba(239, 68, 68, 0.16);
    color: var(--status-problem);
    border: 1px solid rgba(239, 68, 68, 0.38);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07);
}

/* A citation is something to click, so it should look like it. */
.citation-chip {
    display: inline-block;
    padding: 0.15rem 0.65rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border-radius: 999px;
    background: var(--glass);
    border: 1px solid var(--border-faint);
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    cursor: pointer;
    transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease;
}

.citation-chip:hover {
    color: var(--text-bright);
    border-color: var(--accent-primary);
    background: rgba(168, 85, 247, 0.08);
}
</style>
'''
