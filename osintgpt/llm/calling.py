# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: calling.py
# Description: Tool calling in provider-neutral terms. Every backend speaks
#   this shape, so answer quality does not depend on which vendor was picked.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Any, Dict, List, Optional


# ToolSpec class
@dataclass(frozen=True)
class ToolSpec:
    '''
    A tool as the model is told about it.
    '''
    name: str
    description: str
    # JSON Schema for the arguments. Every provider accepts this shape, which
    # is why it is the one carried here rather than any vendor's wrapper.
    parameters: Dict[str, Any] = field(default_factory=dict)

    def schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': self.parameters.get('properties', {}),
            'required': self.parameters.get('required', [])
        }


# ToolCall class
@dataclass(frozen=True)
class ToolCall:
    '''
    One tool the model asked to run.
    '''
    # The provider's own identifier for this call. It travels back with the
    # result, and providers disagree about where it lives, so it is carried
    # rather than reconstructed.
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


# ModelTurn class
@dataclass(frozen=True)
class ModelTurn:
    '''
    What the model produced in one round: something said, something to run,
    or both.
    '''
    text: str = ''
    calls: List[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.calls)


# Exchange class
@dataclass(frozen=True)
class Exchange:
    '''
    A completed round: what the model asked for, and what it got back.

    Kept in neutral terms because the two vendors disagree about how a tool
    result is carried — a `tool` role against a `tool_result` block — so each
    provider rebuilds its own messages from this rather than sharing a format
    neither actually uses.
    '''
    turn: ModelTurn
    # Call id to the result, already serialized. What the model reads.
    results: Dict[str, str] = field(default_factory=dict)


class ToolCallingUnsupported(NotImplementedError):
    '''
    Raised when a backend cannot call tools.

    A distinct type rather than a bare NotImplementedError, because the loop
    catches exactly this to fall back to the static pipeline, and swallowing
    every NotImplementedError would hide real bugs.
    '''


# describe a tool for a model
def tool_spec(
    name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None
) -> ToolSpec:
    '''
    Args:
        name (str): Tool name, as the model will call it.
        description (str): What it does and when to reach for it.
        properties (dict, optional): JSON Schema properties.
        required (List[str], optional): Which of them are required.

    Returns:
        ToolSpec: The specification.
    '''
    return ToolSpec(
        name=name,
        description=description,
        parameters={
            'properties': properties or {},
            'required': required or []
        }
    )
