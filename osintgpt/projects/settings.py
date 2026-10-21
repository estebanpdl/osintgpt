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
from osintgpt.rate_settings import EmbeddingRateSettings

GRAPH_MODEL_FIELDS = ('graph_model_mode', 'graph_generation_provider', 'graph_generation_model')


def graph_model_settings(settings, defaults=None):
    '''Resolve graph choices without combining models from different provider defaults.'''
    def selection(value):
        mode, provider, model = (getattr(value, name, '') for name in GRAPH_MODEL_FIELDS)
        if mode not in ('', 'inherit', 'independent'):
            raise ValueError('graph_model_mode must be inherit, independent, or empty for user defaults.')
        return mode or ('independent' if provider or model else ''), provider, model

    mode, provider, model = selection(settings)
    default_mode, default_provider, default_model = selection(defaults)
    if default_mode != 'independent':
        default_provider, default_model = '', ''
    mode = mode or default_mode or 'inherit'
    if mode == 'inherit':
        provider, model = '', ''
    else:
        model = model or (default_model if not provider or provider == default_provider else '')
        provider = provider or default_provider
    return dict(zip(GRAPH_MODEL_FIELDS, (mode, provider, model)))

# ProjectSettings class
@dataclass(frozen=True)
class ProjectSettings(EmbeddingRateSettings):
    '''
    Per-project configuration. An empty string means "not chosen here" and
    defers to whatever the caller supplies, so a project only records the
    decisions actually made for it.
    '''
    embedding_provider: str = 'openai'
    embedding_model: str = ''
    generation_provider: str = 'openai'
    generation_model: str = ''
    graph_model_mode: str = ''
    graph_generation_provider: str = ''
    graph_generation_model: str = ''
    # Vision model for document ingestion; empty inherits the generation model.
    ingestion_model: str = ''
    semantic_enabled: bool = True
    lexical_enabled: bool = True
    # Extraction uses calls per text window and requires an explicit build.
    graph_enabled: bool = False
    # One extra generation call after each answer. On by default: it is small
    # against the answer it follows, and it surfaces connections an analyst
    # did not know to ask about, which is where findings usually are.
    suggest_followups: bool = True
    # Question-and-answer pairs carried into a follow-up, most recent first.
    # Zero is off; there is no separate switch.
    conversation_window: int = 3
    storage_backend: str = 'sqlite'
    cost_ceiling_usd: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        for name in GRAPH_MODEL_FIELDS:
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'{name} must be text.')
        graph_model_settings(self)

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
