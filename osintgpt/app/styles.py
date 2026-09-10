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
THEME = {
    'base': 'dark',
    'primaryColor': '#a855f7',
    'backgroundColor': '#14101a',
    'secondaryBackgroundColor': '#1e1826',
    'textColor': '#f4f1f7',
    'font': 'sans serif'
}

STYLESHEET = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --accent-primary: #a855f7;
    --accent-secondary: #e879f9;
    --background-dark: #14101a;
    --background-light: #1e1826;
    --surface: rgba(255, 255, 255, 0.04);
    --border-subtle: rgba(255, 255, 255, 0.1);
    --text-primary: #f4f1f7;
    --text-secondary: #a89bb4;

    /* Reserved — meaning, not decoration. */
    --status-good: #10b981;
    --status-partial: #f59e0b;
    --status-problem: #ef4444;
}

.main .block-container {
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #14101a 0%, #1e1826 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

.osintgpt-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.osintgpt-subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

.stButton > button {
    background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
    color: #14101a;
    border: none;
    border-radius: 22px;
    padding: 0.45rem 1.6rem;
    font-weight: 600;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(168, 85, 247, 0.45);
}

/* A follow-up is a question, not a call to action. Making them look like
   primary buttons would compete with the answer they sit under. */
.followup-row .stButton > button {
    background: var(--surface);
    color: var(--text-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    font-weight: 400;
    text-align: left;
    box-shadow: none;
}

.followup-row .stButton > button:hover {
    border-color: var(--accent-primary);
    transform: none;
    box-shadow: none;
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    border-radius: 10px;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 10px;
}

[data-testid="stMetricValue"] {
    color: var(--accent-secondary);
}

.stExpander {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    background: var(--background-dark);
    border-right: 1px solid var(--border-subtle);
}

.status-badge {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 18px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-good {
    background: rgba(16, 185, 129, 0.15);
    color: var(--status-good);
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.status-partial {
    background: rgba(245, 158, 11, 0.15);
    color: var(--status-partial);
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.status-problem {
    background: rgba(239, 68, 68, 0.15);
    color: var(--status-problem);
    border: 1px solid rgba(239, 68, 68, 0.35);
}

/* A citation is something to click, so it should look like it. */
.citation-chip {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    margin: 0.1rem 0.2rem 0.1rem 0;
    border-radius: 14px;
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-family: 'Inter', monospace;
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
