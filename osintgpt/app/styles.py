# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: styles.py
# Description: The palette the app is built from and the helpers that apply
#   it. Plum and violet, taken from the mark — and three colours held back,
#   because they carry meaning here.
# =================================================================================

# import modules
import html

# type hints
from typing import Dict

from .stylesheet import STYLESHEET

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
    text = escape_html(value)
    # Backslash first: it is the escape character, so escaping it after the
    # others would double the backslashes they just added.
    for character in MARKDOWN_SPECIAL:
        text = text.replace(character, f'\\{character}')

    return text


# text that goes inside a block of raw HTML
def escape_html(value) -> str:
    '''
    Make arbitrary text safe inside an HTML block written with
    `unsafe_allow_html`.

    Distinct from `escape`, and the distinction is the renderer's: a string
    that opens with a block-level tag is raw HTML, and markdown is not parsed
    inside it. The backslashes `escape` adds are never consumed there, so they
    reach the screen — `exact\\_search`, `terms=\\[...\\]`. Inline HTML inside
    a paragraph is the other case, and still needs `escape`.

    Args:
        value: Text to escape.

    Returns:
        str: The same text, safe as HTML content.
    '''
    return html.escape(str(value), quote=False)


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