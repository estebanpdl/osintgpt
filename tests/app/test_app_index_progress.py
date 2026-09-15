'''Real Streamlit indexing/recovery flows, driven by offline provider responses.'''

from types import SimpleNamespace

import pytest

from osintgpt import Project
from osintgpt.app.session import MATERIAL, SETTINGS, SELECTED, VIEW, runtime_revision
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.index_status import saved_index_status
from osintgpt.llm.embedding_pacing import EmbeddingPacer
from osintgpt.llm.embedding_retry import RetryPolicy
from osintgpt.llm.openai_compat import OpenAICompatEmbedding
from osintgpt.rate_settings import RateLimits
from tests.llm.test_embedding_errors import api_error
from tests.llm.test_rate_ledger import Clock, ledger

AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
SCRIPT = str(__import__('osintgpt.app.launch', fromlist=['script_path']).script_path())


class OfflineEndpoint:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.transient = False

    def create(self, *, input, model):
        self.calls.append(list(input))
        if self.fail and len(self.calls) > 1 or self.transient and len(self.calls) == 1:
            raise api_error()
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0]) for i in range(len(input))], usage=SimpleNamespace(prompt_tokens=len(input)))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    home = tmp_path / 'home'
    monkeypatch.setattr('osintgpt.projects.default_home', lambda: home)
    monkeypatch.setattr('osintgpt.projects.paths.default_home', lambda: home)
    project = Project.create('Progress case', home=home)
    path = project.paths.root / 'records.csv'
    path.write_text('id,message\n' + ''.join(f'{i},Record {i} contains evidence.\n' for i in range(5)) + 'empty,\nshort,tiny\n', encoding='utf-8')
    Corpus.load(project.paths.sources).register(path, FieldMapping(content=('message',), min_chars=10))
    endpoint = OfflineEndpoint()
    provider = OpenAICompatEmbedding.__new__(OpenAICompatEmbedding)
    provider.model, provider.provider = 'text-embedding-3-small', 'openai'
    provider.batch_size, provider.billable, provider.recorder = 2, True, None
    provider.client = SimpleNamespace(embeddings=endpoint)
    provider.pacer = EmbeddingPacer('openai', provider.model, 'https://offline.invalid', 'test', RateLimits(rpm=60), ledger=ledger(Clock()), retry_policy=RetryPolicy(max_attempts=2))
    provider.pacer.retry_jitter = lambda low, high: 0
    builds = []
    def build(*args, **kwargs):
        builds.append(kwargs)
        return provider
    monkeypatch.setattr('osintgpt.app.session.build_embedding_provider', build)
    monkeypatch.setattr('osintgpt.app.session.build_generation_provider', lambda *args, **kwargs: pytest.fail('No generation client should be needed'))
    yield project, endpoint, builds, provider, home
    st.cache_resource.clear()


def app_for(project, view=MATERIAL):
    app = AppTest.from_file(SCRIPT, default_timeout=60)
    app.session_state[SELECTED] = project.slug
    app.session_state[VIEW] = view
    return app.run()


def index(app):
    return next(b for b in app.button if b.label == 'Index now').click().run()


def messages(elements):
    return '\n'.join(element.value for element in elements)


def test_page_load_requires_no_provider_or_journal_creation(setup):
    project, endpoint, builds, _, _ = setup
    app = app_for(project)
    assert not app.exception and not builds and not endpoint.calls
    assert not project.paths.index_journal.exists()
    assert 'text-embedding-3-small' in messages(app.caption)


def test_failure_shows_saved_and_remaining_and_reopening_can_resume(setup):
    project, endpoint, _, _, _ = setup
    endpoint.fail = True
    app = index(app_for(project))
    assert not app.exception and not app.success
    assert 'failed' in messages(app.warning)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics['Saved checkpoint chunks'] == '2'
    assert metrics['Pending embedding chunks'] == '3'
    assert 'Retry 2/2' in messages(app.text)
    pending = saved_index_status(project)['pending'][0]
    assert pending['details']['statistics']['without_content'] == 1
    assert pending['details']['statistics']['too_short'] == 1
    reopened = app_for(project)
    assert {m.label: m.value for m in reopened.metric}['Saved checkpoint chunks'] == '2'
    before = len(endpoint.calls)
    endpoint.fail = False
    app = index(reopened)
    assert not app.exception and app.success
    assert endpoint.calls[before][0] == 'Record 2 contains evidence.'
    assert sum(len(batch) for batch in endpoint.calls[before:]) == 3
    published = saved_index_status(project)['published'][0]
    assert published['statistics']['records'] == 7 and published['statistics']['kept'] == 5
    before = len(endpoint.calls)
    app.run()  # Result survives widget reruns; no new requests.
    assert 'Last indexing run' in messages(app.caption)
    index(app)
    assert len(endpoint.calls) == before


def test_success_after_retry_shows_usage_once_and_exclusions(setup):
    project, endpoint, _, _, _ = setup
    endpoint.transient = True
    app = index(app_for(project))
    assert not app.exception
    assert '3 successful provider response(s)' in messages(app.caption)
    assert '4 embedding request attempt(s)' in messages(app.caption)
    assert 'excluded 1 with no content and 1 below the minimum' in messages(app.caption)


def test_cost_stop_is_presented_with_durable_checkpoint_and_resume_guidance(setup):
    project, endpoint, _, _, _ = setup
    app = app_for(project)
    next(w for w in app.number_input if w.label == 'Stop at estimated run cost (USD)').set_value(0.0)
    app = index(app)
    assert not app.exception and not app.success
    assert 'ceiling' in messages(app.warning)
    assert len(endpoint.calls) == 1
    assert saved_index_status(project)['pending'][0]['completed'] == 2
    assert 'turn off Rebuild all' in messages(app.info)


def test_provider_switch_is_flagged_without_building_a_client(setup):
    project, endpoint, builds, _, _ = setup
    index(app_for(project))
    before = len(builds)
    project.with_settings(embedding_provider='gemini', embedding_model='gemini-embedding-001').save()
    app = app_for(project)
    assert not app.exception and len(builds) == before
    assert 'gemini / gemini-embedding-001' in messages(app.caption)
    assert 'openai / text-embedding-3-small' in messages(app.warning)
    assert 'incompatible checkpoints cannot be reused' in messages(app.warning)


def test_settings_warns_before_provider_change_is_saved(setup):
    project, _, builds, _, _ = setup
    index(app_for(project))
    app = app_for(project, SETTINGS)
    before = len(builds)
    next(w for w in app.selectbox if w.label == 'Embedding provider').select('gemini').run()
    assert not app.exception and len(builds) == before
    assert 'rebuilds affected files' in messages(app.warning)


def test_rebuild_requires_explicit_selection_and_default_update_does_no_work(setup):
    project, endpoint, _, _, _ = setup
    app = index(app_for(project))
    before = len(endpoint.calls)
    index(app)
    assert len(endpoint.calls) == before
    next(w for w in app.checkbox if w.label.startswith('Rebuild all')).check().run()
    assert 'recompute all registered files' in messages(app.warning)
    index(app)
    assert len(endpoint.calls) == before + 3


def test_runtime_cache_revision_changes_for_external_settings_edits(setup):
    project, _, _, _, home = setup
    old = runtime_revision(project, home)
    assert old == runtime_revision(project, home)
    changed = project.with_settings(openai_embedding_tpm=2_000_000)
    assert old != runtime_revision(changed, home)


def test_zero_kept_records_shows_exclusions_and_makes_no_embedding_calls(setup):
    project, endpoint, _, _, _ = setup
    (project.paths.root / 'records.csv').write_text('id,message\n1,\n2,tiny\n', encoding='utf-8')
    app = index(app_for(project))
    assert not app.exception and not endpoint.calls
    assert 'Kept 0/2 records; excluded 1 with no content and 1 below the minimum' in messages(app.caption)


def test_preview_names_exclusions_and_does_not_claim_remaining_cost(setup):
    project, endpoint, builds, _, _ = setup
    app = app_for(project)
    next(w for w in app.checkbox if w.label == 'Preview what would be indexed').check().run()
    assert not app.exception and not endpoint.calls and not builds
    assert 'not an estimate of the remaining run cost' in messages(app.caption)
    assert app.dataframe[0].value['without_content'].iloc[0] == 1
