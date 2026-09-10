# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: registry.py
# Description: The tools as a model sees them. One table, so every provider is
#   offered exactly the same set and a trace from one reads against another.
# =================================================================================

# type hints
from typing import Any, Dict, List

# import osintgpt llm
from osintgpt.llm.calling import ToolSpec, tool_spec

from .tools import (
    REFS,
    SNIPPETS,
    ToolContext,
    ToolResult,
    exact_search,
    fetch_source,
    graph_query,
    list_documents,
    semantic_search,
    snowball_search
)

_STRING = {'type': 'string'}
_INTEGER = {'type': 'integer'}

# Descriptions are written for the model, and they are where the survey-first
# habit is taught: the tools do not enforce an order, so the wording is what
# makes a cheap survey the obvious first move.
TOOL_SPECS: List[ToolSpec] = [
    tool_spec(
        'semantic_search',
        'Find passages that mean something close to a query, even when they '
        'use different words. Use it for concepts and claims. It will not '
        'reliably find an exact identifier — use exact_search for those.',
        properties={
            'query': dict(_STRING, description='What to look for, in your own words.'),
            'limit': dict(_INTEGER, description='Passages to return, up to 30.'),
            'days': dict(_INTEGER, description='Only documents this recent.')
        },
        required=['query']
    ),
    tool_spec(
        'exact_search',
        'Find the exact characters given, anywhere in the corpus. Use it for '
        'handles, usernames, hashes, URLs, case numbers, error codes and '
        'names — the things an embedding blurs. Pass mode="refs" first to see '
        'WHICH documents match and HOW MANY times, with no content: that '
        'costs almost nothing, so survey widely before reading anything.',
        properties={
            'terms': {
                'type': 'array', 'items': _STRING,
                'description': 'Literal strings, in the script they are written in.'
            },
            'mode': dict(
                _STRING, enum=[SNIPPETS, REFS],
                description='"refs" for locations and counts, "snippets" for content.'
            ),
            'limit': _INTEGER,
            'days': _INTEGER
        },
        required=['terms']
    ),
    tool_spec(
        'list_documents',
        'List what the project holds before reading any of it. Cheap. Use it '
        'when you do not yet know what exists.',
        properties={
            'pattern': dict(_STRING, description='Only refs containing this.'),
            'limit': _INTEGER
        }
    ),
    tool_spec(
        'snowball',
        'Follow a thread outward: retrieve, then search for what the best '
        'passage says, and repeat. Answers "what else is adjacent to this" '
        'rather than "what matches this", so it reaches material a direct '
        'search misses. Each hop reports how far it has drifted from the '
        'original question — read that before trusting a late hop.',
        properties={
            'query': dict(_STRING, description='Where to start.'),
            'depth': dict(_INTEGER, description='Hops to take, up to 10.'),
            'threshold': {
                'type': 'number',
                'description': 'Stop when similarity falls below this.'
            }
        },
        required=['query']
    ),
    tool_spec(
        'graph_query',
        'Ask what the documents assert about an entity, or how two entities '
        'are connected. Every claim carries the document and sentence that '
        'assert it. Not every project has a graph; the tool says so if not.',
        properties={
            'entity': dict(_STRING, description='The name to ask about.'),
            'target': dict(
                _STRING,
                description='Given one, returns the chain connecting them.'
            ),
            'limit': _INTEGER
        },
        required=['entity']
    ),
    tool_spec(
        'fetch_source',
        'Read a document directly, by line. Use it after a search has shown '
        'a document matters and you need more than a passage. It returns a '
        'window and tells you next_offset when there is more.',
        properties={
            'ref': dict(_STRING, description='The document, as searches name it.'),
            'offset': dict(_INTEGER, description='First line, zero-based.'),
            'limit': dict(_INTEGER, description='Lines to return.')
        },
        required=['ref']
    )
]

_HANDLERS = {
    'semantic_search': semantic_search,
    'exact_search': exact_search,
    'list_documents': list_documents,
    'snowball': snowball_search,
    'graph_query': graph_query,
    'fetch_source': fetch_source
}

TOOL_NAMES = [spec.name for spec in TOOL_SPECS]


# run one tool the model asked for
def run_tool(
    context: ToolContext, name: str, arguments: Dict[str, Any]
) -> ToolResult:
    '''
    Dispatch a call by name.

    An unknown tool, or an argument the tool does not take, comes back as a
    result carrying the problem rather than an exception: the model can read
    that and correct itself, where a raised error would end the round.

    Args:
        context (ToolContext): Project and providers.
        name (str): Tool the model named.
        arguments (Dict[str, Any]): Arguments it supplied.

    Returns:
        ToolResult: What the tool returned, or why it could not run.
    '''
    handler = _HANDLERS.get(name)
    if handler is None:
        return ToolResult(
            tool=name,
            error=f'no tool named {name!r}; available: {", ".join(TOOL_NAMES)}'
        )

    try:
        return handler(context, **(arguments or {}))
    except TypeError as error:
        return ToolResult(
            tool=name, error=f'{name} could not be called that way: {error}'
        )
