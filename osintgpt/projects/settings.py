# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: settings.py
# Description: The choices an operator makes per case — models, which retrieval
#   legs run, where vectors are stored, what it is allowed to cost.
# =================================================================================

# import submodules
from dataclasses import asdict, dataclass, fields

# type hints
from typing import Optional

# ProjectSettings class
@dataclass(frozen=True)
class ProjectSettings:
    '''
    Per-project configuration. An empty string means "not chosen here" and
    defers to whatever the caller supplies, so a project only records the
    decisions actually made for it.
    '''
    embedding_provider: str = 'openai'
    embedding_model: str = ''
    generation_provider: str = 'openai'
    generation_model: str = ''
    # Vision model for document ingestion; empty inherits the generation model.
    ingestion_model: str = ''
    semantic_enabled: bool = True
    lexical_enabled: bool = True
    # One generation call per document to build, so opting in is a decision an
    # operator makes for a case rather than a default they inherit.
    graph_enabled: bool = False
    storage_backend: str = 'sqlite'
    cost_ceiling_usd: Optional[float] = None

    # build from a parsed mapping
    @classmethod
    def from_dict(cls, data: dict):
        '''
        Build settings from a mapping, ignoring keys this version does not know.

        Tolerating unknown keys means a project written by a newer osintgpt
        still opens in an older one, minus the settings it cannot honour.

        Args:
            data (dict): Raw values, typically parsed from project.toml.

        Returns:
            ProjectSettings: A new instance.
        '''
        known = {f.name for f in fields(cls)}

        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    # values for the project file
    def to_dict(self) -> dict:
        '''
        A mapping suitable for TOML, with unset optionals dropped.

        Returns:
            dict: Field names to values.
        '''
        return {k: v for k, v in asdict(self).items() if v is not None}
