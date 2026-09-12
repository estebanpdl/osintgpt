# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: styles.py
# Description: The app's stylesheet. Plum and violet, taken from the mark —
#   and three colours held back, because they carry meaning here.
# =================================================================================

# import modules
import html

# type hints
from typing import Dict

# Escaped wherever data reaches a markdown renderer. Backslash leads because
# it escapes the rest. `&`, `<` and `>` are absent on purpose: HTML-escaping
# turns them into entities first, and backslashing those would print the
# entity rather than the character.
MARKDOWN_SPECIAL = '\\`*_[]'

# Reserved. Nothing decorative may use these, because each one is the answer
# to a question an analyst is actually asking:
#
#   green  the answer is grounded, the evidence checks out
#   amber  it worked, but not the way it should have — a degraded answer, a
#          time filter that could not read every timestamp, a partial result
#   red    something is wrong and will produce bad answers — a model mismatch,
#          evidence that is not in its source document
#
# The mark's own accent is crimson. It is deliberately not used as the brand
# colour: an interface whose decoration is red cannot also say "red means
# trouble" and be believed.
STATUS_COLORS: Dict[str, str] = {
    'good': '#10b981',
    'partial': '#f59e0b',
    'problem': '#ef4444'
}

# From the mark: a plum-black ground with a violet accent, which is the same
# family as its crimson without taking the colour that has to mean something.
# Deeper than the previous pair so the gradients and glows in the stylesheet
# have room to sit on top of it.
THEME = {
    'base': 'dark',
    'primaryColor': '#a855f7',
    'backgroundColor': '#0c0a12',
    'secondaryBackgroundColor': '#191421',
    'textColor': '#f5f2fa',
    'font': 'sans serif'
}

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
   they belong to, against the main page, so they wear the quietest surface. */
.followup-row .stButton > button {
    background: rgba(255, 255, 255, 0.025);
    color: var(--text-secondary);
    border: 1px solid var(--border-faint);
    border-radius: 999px;
    font-weight: 400;
    font-size: 0.85rem;
    text-align: left;
    padding: 0.4rem 1.05rem;
    box-shadow: none;
}

.followup-row .stButton > button:hover {
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
   recessed, and tuned down so the prose answer stays the loudest thing. */
[data-testid="stText"] pre,
.stText pre {
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border-faint);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    font-family: ui-monospace, 'Cascadia Code', 'Consolas', monospace;
    font-size: 0.8rem;
    line-height: 1.55;
    white-space: pre-wrap;
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


# inject the stylesheet
def load_css(st) -> None:
    '''
    Args:
        st: The Streamlit module.
    '''
    st.markdown(STYLESHEET, unsafe_allow_html=True)


# text that has to survive a markdown renderer intact
def escape(value) -> str:
    '''
    Make arbitrary text safe to place in a string Streamlit will render as
    markdown, including inside a span written with `unsafe_allow_html`.

    Markdown is still applied to text between raw HTML tags, so a Windows
    path loses the backslash in `\\.osintgpt` and a filename with underscores
    turns into italics. Every value here is data — a path, a document
    reference, something a model wrote — and none of it is markup.

    Args:
        value: Text to escape.

    Returns:
        str: The same text, rendered literally.
    '''
    text = html.escape(str(value), quote=False)
    # Backslash first: it is the escape character, so escaping it after the
    # others would double the backslashes they just added.
    for character in MARKDOWN_SPECIAL:
        text = text.replace(character, f'\\{character}')

    return text


# a coloured badge
def badge(label: str, status: str) -> str:
    '''
    Args:
        label (str): Text to show.
        status (str): One of `good`, `partial`, `problem`.

    Returns:
        str: The badge as HTML, or plain text when the status is not one of \
            the three — an unknown status must not silently borrow a colour \
            that means something.
    '''
    if status not in STATUS_COLORS:
        return label

    return f'<span class="status-badge status-{status}">{label}</span>'


# the theme as Streamlit command-line flags
def theme_flags() -> list:
    '''
    The theme, passed at launch rather than written to a config file.

    A packaged app has no say over the working directory Streamlit reads its
    config from, so the theme travels with the launch command and works
    identically from a wheel.

    Returns:
        list: Flags for `streamlit run`.
    '''
    return [f'--theme.{key}={value}' for key, value in THEME.items()]