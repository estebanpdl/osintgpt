'''Graph model selection respects scope boundaries and explicit chat inheritance.'''

import pytest

from osintgpt import Project
from osintgpt.projects import ProjectSettings, load_user_defaults, save_user_defaults


def effective(tmp_path, defaults=None, **choices):
    return Project.create('Models', home=tmp_path, settings=ProjectSettings(**choices)).effective_settings(defaults)


def test_existing_projects_inherit_without_rewriting_their_file(tmp_path):
    project = Project.create('Existing', home=tmp_path)
    text = project.paths.config.read_text(encoding='utf-8')
    project.paths.config.write_text('\n'.join(line for line in text.splitlines()
        if not line.startswith(('graph_model_mode', 'graph_generation_'))), encoding='utf-8')
    before = project.paths.config.read_bytes()
    assert Project.load(project.paths.root).effective_settings().graph_model_mode == 'inherit'
    assert project.paths.config.read_bytes() == before


def test_pair_round_trips_and_uses_user_defaults(tmp_path):
    defaults = ProjectSettings(graph_model_mode='independent', graph_generation_provider='anthropic',
                               graph_generation_model='extraction-model')
    save_user_defaults(tmp_path, defaults)
    assert load_user_defaults(tmp_path) == defaults
    selected = effective(tmp_path, defaults)
    assert selected.graph_model_mode == 'independent'
    assert selected.graph_generation_provider == 'anthropic'
    assert selected.graph_generation_model == 'extraction-model'
    project = Project.create('Pinned', home=tmp_path, settings=selected)
    assert Project.load(project.paths.root).settings == selected


def test_explicit_inherit_ignores_independent_defaults_and_dormant_pair(tmp_path):
    selected = effective(tmp_path, ProjectSettings(graph_generation_provider='anthropic', graph_generation_model='old'),
                         graph_model_mode='inherit', graph_generation_provider='ollama', graph_generation_model='dormant')
    assert selected.graph_model_mode == 'inherit'
    assert selected.graph_generation_provider == selected.graph_generation_model == ''


def test_project_provider_change_never_borrows_other_provider_model(tmp_path):
    selected = effective(tmp_path, ProjectSettings(graph_generation_provider='anthropic', graph_generation_model='old'),
                         graph_generation_provider='ollama')
    assert selected.graph_model_mode == 'independent'
    assert selected.graph_generation_provider == 'ollama' and selected.graph_generation_model == ''


@pytest.mark.parametrize('choices,expected', [
    ({'graph_generation_provider': 'anthropic'}, 'default-model'),
    ({'graph_generation_model': 'project-model'}, 'project-model'),
    ({'graph_model_mode': 'independent'}, 'default-model'),
])
def test_compatible_defaults_fill_unset_pair_fields(tmp_path, choices, expected):
    selected = effective(tmp_path, ProjectSettings(graph_generation_provider='anthropic',
                                                   graph_generation_model='default-model'), **choices)
    assert selected.graph_generation_provider == 'anthropic'
    assert selected.graph_generation_model == expected


@pytest.mark.parametrize('field,value', [('graph_model_mode', 'automatic'), ('graph_model_mode', None),
                                        ('graph_generation_provider', 5), ('graph_generation_model', False)])
def test_malformed_graph_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ProjectSettings(**{field: value})


def test_graph_validation_keeps_inherited_rate_validation():
    with pytest.raises(ValueError, match='openai_embedding_rpm'):
        ProjectSettings(openai_embedding_rpm=-1)
