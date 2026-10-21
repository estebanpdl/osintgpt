'''Saved text identity is visible in the existing source preview.'''

import pytest

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest


def test_source_preview_shows_saved_version_pages_and_sections():
    app = AppTest.from_string('''
import streamlit as st
from types import SimpleNamespace
from osintgpt.app.views.chat import _sources
provenance = {'source_version': 'abc123def456' * 5 + 'abcd',
              'matches': [{'spans': [{'page': 47, 'section': 'Funding'}]}]}
entry = SimpleNamespace(payload={'passages': [{
    'citation': 'paper.pdf', 'ref': 'paper.pdf', 'text': 'Alpha funds Beta.',
    'provenance': provenance}]})
answer = SimpleNamespace(trace=SimpleNamespace(entries=[entry]), sources=['paper.pdf'])
_sources(st, answer)
''').run()
    assert not app.exception
    caption = ' '.join(item.value for item in app.caption)
    assert 'Saved text version abc123def456' in caption
    assert 'pages 47' in caption and 'sections Funding' in caption
