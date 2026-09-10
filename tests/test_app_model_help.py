'''What the model fields tell an operator to type.'''

import pytest

from osintgpt.app.views.settings import MODEL_SHAPES, model_help
from osintgpt.llm import EMBEDDING_BACKENDS, GENERATION_BACKENDS

BACKENDS = {'embedding': EMBEDDING_BACKENDS, 'generation': GENERATION_BACKENDS}
ROLES = [
    (role, provider)
    for role, registry in BACKENDS.items()
    for provider in registry
]


@pytest.mark.parametrize('role,provider', ROLES)
def test_every_provider_is_told_whether_it_has_a_default(role, provider):
    '''
    "Leave empty for the provider default" is false for every backend that
    publishes none, and an operator who believes it gets an error instead of
    a default.
    '''
    text = model_help(role, provider, BACKENDS[role])
    spec = BACKENDS[role][provider]

    if spec.default_model:
        assert f'Leave empty to use {spec.default_model}' in text
    else:
        assert 'must be filled in' in text


@pytest.mark.parametrize('role,provider', ROLES)
def test_the_example_belongs_to_that_role(role, provider):
    '''
    An embedding field showing a chat model, or the reverse, is worse than
    showing nothing: it is a wrong answer to the question being asked.
    '''
    shape = MODEL_SHAPES.get((role, provider))
    if shape is None:
        return

    other = MODEL_SHAPES.get(
        ('generation' if role == 'embedding' else 'embedding', provider)
    )
    if other is not None:
        assert shape != other


@pytest.mark.parametrize('role,provider', ROLES)
def test_discovery_is_offered_only_where_it_works(role, provider):
    '''
    `doctor --check-providers` reports "provider does not support model
    discovery" for a backend that cannot be asked. Suggesting it there sends
    an operator to a dead end.
    '''
    text = model_help(role, provider, BACKENDS[role])
    offered = 'check-providers' in text

    assert offered is BACKENDS[role][provider].discovers_models


@pytest.mark.parametrize('role,provider', ROLES)
def test_it_says_an_id_is_wanted_rather_than_a_name(role, provider):
    assert 'not a display name' in model_help(role, provider, BACKENDS[role])


def test_every_shape_names_a_provider_that_exists():
    '''
    A shape for a provider nobody can select is dead text, and the pairing is
    the thing most likely to rot when a backend is added or renamed.
    '''
    for (role, provider) in MODEL_SHAPES:
        assert provider in BACKENDS[role], f'{role}/{provider}'


def test_an_unknown_provider_does_not_raise():
    # The selectbox cannot produce one, but a project file can carry a
    # provider this version does not know.
    assert 'must be filled in' in model_help(
        'embedding', 'not-a-provider', EMBEDDING_BACKENDS
    )
