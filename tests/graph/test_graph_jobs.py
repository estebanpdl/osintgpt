'''Build ownership, safe cancellation, and durable recovery outside page execution.'''

from dataclasses import replace
from threading import Event
from time import monotonic

import pytest

from osintgpt import Project
from osintgpt.graph import graph_for, saved_graph_status
from osintgpt.graph.jobs import GraphJobs
from osintgpt.graph.run_settings import GraphRunSettings
from osintgpt.ingestion import Corpus

WAIT_SECONDS = 120


def project_with_source(tmp_path):
    project = Project.create('Job', home=tmp_path).with_settings(graph_enabled=True)
    (project.paths.root / 'a.md').write_text('Alpha funds Beta.')
    Corpus.load(project.paths.sources).register('a.md')
    return project


class Generator:
    def __init__(self, entered=None, release=None):
        self.entered, self.release, self.calls = entered, release, 0

    def generate(self, *args):
        self.calls += 1
        if self.entered:
            self.entered.set()
            assert self.release.wait(WAIT_SECONDS), 'Generator was never released'
        return '{"entities":[],"edges":[]}'


def join(jobs, project):
    deadline = monotonic() + WAIT_SECONDS
    thread = jobs.jobs[jobs.key(project)].thread
    # Completion is a state transition, not a ten-second performance budget.
    # Short joins yield to the worker and return immediately when it exits.
    while jobs.get(project)['active'] and monotonic() < deadline:
        thread.join(min(0.05, max(0, deadline - monotonic())))
    assert not jobs.get(project)['active'], jobs.get(project)
    thread.join(max(0, deadline - monotonic()))
    assert not thread.is_alive(), 'Completed job left its worker running'


def test_reruns_and_sessions_attach_to_one_build_and_cancel_saved_work(tmp_path):
    project = project_with_source(tmp_path)
    entered, release = Event(), Event()
    generator = Generator(entered, release)
    jobs, settings = GraphJobs(), GraphRunSettings()
    first, started = jobs.start(project, lambda: generator, settings, action='initial')
    try:
        assert started and entered.wait(WAIT_SECONDS)
        second, started = jobs.start(project, lambda: pytest.fail('Duplicate build'), settings, action='initial')
        assert not started and second['id'] == first['id']
        assert jobs.get(Project.load(project.paths.root))['id'] == first['id']
        assert jobs.cancel(project)
    finally:
        release.set()
        join(jobs, project)
    assert jobs.get(project)['status'] == 'interrupted'
    assert saved_graph_status(project)['status'] == 'interrupted'
    assert saved_graph_status(project)['checkpoints']['windows_saved'] == 1
    resumed = Generator()
    fresh_process = GraphJobs()
    fresh_process.start(project, lambda: resumed, settings, action='resume')
    join(fresh_process, project)
    assert saved_graph_status(project)['status'] == 'complete'
    assert resumed.calls == 0


def test_two_managers_cannot_duplicate_provider_requests(tmp_path):
    project = project_with_source(tmp_path)
    entered, release = Event(), Event()
    first, second = GraphJobs(), GraphJobs()
    generator = Generator(entered, release)
    other = Generator()
    first.start(project, lambda: generator, GraphRunSettings(), action='initial')
    try:
        assert entered.wait(WAIT_SECONDS)
        second.start(project, lambda: other, GraphRunSettings(), action='initial')
        join(second, project)
        assert 'already running' in second.get(project)['problem']
        assert other.calls == 0 and saved_graph_status(project)['status'] == 'building'
    finally:
        release.set()
        join(first, project)


def test_disabled_build_is_rejected_without_provider(tmp_path):
    project = project_with_source(tmp_path).with_settings(graph_enabled=False)
    with pytest.raises(ValueError, match='Enable'):
        GraphJobs().start(project, lambda: pytest.fail('Provider must not be touched'), GraphRunSettings())
    assert not (project.paths.root / 'graph.sqlite').exists()


@pytest.mark.parametrize('change', [{'rpm': -1}, {'context_budget': 20}, {'retry_attempts': 0},
                                   {'retry_seconds': float('nan')}, {'cost_ceiling_usd': -2},
                                   {'repair_quotes': 'yes'}])
def test_invalid_saved_build_options_are_rejected(change):
    with pytest.raises(ValueError):
        replace(GraphRunSettings(), **change)


def test_update_rebuild_and_project_isolation(tmp_path):
    project = project_with_source(tmp_path)
    other = Project.create('Other', home=tmp_path).with_settings(graph_enabled=True)
    jobs, generator, settings = GraphJobs(), Generator(), GraphRunSettings()
    for action, calls in [('initial', 1), ('update', 1), ('rebuild', 2)]:
        jobs.start(project, lambda: generator, settings, action=action)
        join(jobs, project)
        assert jobs.get(project)['status'] == 'complete'
        assert generator.calls == calls
    assert jobs.get(other) is None
    (project.paths.root / 'a.md').write_text('Alpha now funds Gamma.')
    jobs.start(project, lambda: generator, settings, action='update')
    join(jobs, project)
    assert generator.calls == 3


def test_initialization_is_inside_the_shared_build_lock(tmp_path, monkeypatch):
    from osintgpt.graph import build as module
    project = project_with_source(tmp_path)
    entered, release = Event(), Event()
    original = module.graph_for
    def initialize(project):
        entered.set()
        assert release.wait(WAIT_SECONDS)
        return original(project)
    monkeypatch.setattr(module, 'graph_for', initialize)
    first, second = GraphJobs(), GraphJobs()
    first.start(project, Generator, GraphRunSettings(), action='initial')
    try:
        assert entered.wait(WAIT_SECONDS)
        assert not (project.paths.root / 'graph.sqlite').exists()
        second.start(project, Generator, GraphRunSettings(), action='initial')
        join(second, project)
        assert 'already running' in second.get(project)['problem']
    finally:
        release.set()
        join(first, project)
