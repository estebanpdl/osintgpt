'''No view is held behind a provider it does not use.'''

import pytest

from osintgpt.app.session import runtime_for
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.projects import Project


class Embedder:
    model = 'test-model'


class Generator:
    model = 'test-chat'


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def project(home):
    return Project.create('Case Lazy', home=home)


def refusing(*_args, **_kwargs):
    raise MissingEnvironmentVariableError(
        'OPENAI_API_KEY', hint='the openai provider needs it'
    )


class TestNothingIsBuiltUpFront:
    def test_a_runtime_is_built_for_a_project_with_no_credential(
        self, project, home, monkeypatch
    ):
        '''
        The whole point. Settings reads configuration and reports credential
        presence; holding it behind the credential it reports on left an
        operator unable to reach the one screen that would fix the problem.
        '''
        monkeypatch.setattr(
            'osintgpt.app.session.build_embedding_provider', refusing
        )
        monkeypatch.setattr(
            'osintgpt.app.session.build_generation_provider', refusing
        )

        runtime = runtime_for(project, home)

        assert runtime.project.slug == project.slug

    def test_the_failure_arrives_when_the_provider_is_used(
        self, project, home, monkeypatch
    ):
        monkeypatch.setattr(
            'osintgpt.app.session.build_embedding_provider', refusing
        )
        runtime = runtime_for(project, home)

        with pytest.raises(MissingEnvironmentVariableError):
            runtime.embedder

    def test_a_usable_generator_is_reachable_when_only_embedding_is_broken(
        self, project, home, monkeypatch
    ):
        '''
        The legs fail independently. A project that cannot embed can still
        answer from what is already indexed, and one broken provider must not
        take the other down with it.
        '''
        monkeypatch.setattr(
            'osintgpt.app.session.build_embedding_provider', refusing
        )
        monkeypatch.setattr(
            'osintgpt.app.session.build_generation_provider',
            lambda *a, **k: Generator()
        )

        assert runtime_for(project, home).generator.model == 'test-chat'


class TestProvidersAreBuiltOnce:
    def test_a_provider_is_reused_across_uses(
        self, project, home, monkeypatch
    ):
        calls = []

        def build(*_args, **_kwargs):
            calls.append(1)

            return Embedder()

        monkeypatch.setattr(
            'osintgpt.app.session.build_embedding_provider', build
        )
        runtime = runtime_for(project, home)
        first, second = runtime.embedder, runtime.embedder

        assert first is second
        assert len(calls) == 1

    def test_an_injected_builder_is_called_once_for_both(
        self, project, home
    ):
        '''
        The test seam hands back a pair. Calling it twice would build two
        unrelated halves and quietly discard one of each.
        '''
        calls = []

        def builder(effective, config):
            calls.append(1)

            return Embedder(), Generator()

        runtime = runtime_for(project, home, builder=builder)

        assert runtime.embedder.model == 'test-model'
        assert runtime.generator.model == 'test-chat'
        assert len(calls) == 1

    def test_an_injected_builder_is_not_called_until_used(
        self, project, home
    ):
        calls = []

        def builder(effective, config):
            calls.append(1)

            return Embedder(), Generator()

        runtime_for(project, home, builder=builder)

        assert calls == []


class TestTheSettingsViewNeedsNoProvider:
    def test_it_renders_for_a_project_whose_provider_cannot_be_built(
        self, project, home, monkeypatch
    ):
        from osintgpt.app.views import settings

        monkeypatch.setattr(
            'osintgpt.app.session.build_embedding_provider', refusing
        )
        monkeypatch.setattr(
            'osintgpt.app.session.build_generation_provider', refusing
        )
        runtime = runtime_for(project, home)

        rendered = _render(settings.render, runtime, home)

        assert any('Credentials' in line for line in rendered)


def _render(render, runtime, home):
    '''
    A Streamlit stand-in that records what a view wrote. Enough surface for
    the settings view, and no more — a fuller fake would be testing itself.
    '''
    written = []

    class Column:
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class St:
        def __getattr__(self, _name):
            def anything(*args, **_kwargs):
                if args and isinstance(args[0], str):
                    written.append(args[0])

                return ''
            return anything

        def columns(self, spec):
            count = spec if isinstance(spec, int) else len(spec)

            return [Column() for _ in range(count)]

        def selectbox(self, _label, options, index=0, **_kwargs):
            return options[index] if options else ''

        def text_input(self, _label, value='', **_kwargs):
            return value

        def checkbox(self, _label, value=False, **_kwargs):
            return value

        def button(self, *_args, **_kwargs):
            return False

    render(St(), runtime, home, {})

    return written
