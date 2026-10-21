'''Process-wide build ownership survives browser sessions; recovery remains in the graph journal.'''

from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from uuid import uuid4

from .build import build_graph
from .events import GraphCancelled


@dataclass
class GraphJob:
    id: str = field(default_factory=lambda: uuid4().hex)
    cancel: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    data: dict = field(default_factory=lambda: {'active': True, 'status': 'starting', 'events': []})
    thread: Thread | None = None

    def snapshot(self):
        with self.lock:
            return deepcopy({**self.data, 'id': self.id, 'cancel_requested': self.cancel.is_set()})

    def observe(self, event):
        with self.lock:
            if event.kind == 'started':
                self.data['status'] = 'building'
                self.data['graph_model'] = event.data.get('graph_model', {})
            if event.kind != 'progress':
                self.data['last_event'] = {'kind': event.kind, 'data': event.data}
            if event.kind == 'progress':
                self.data.update(event.data)
            if event.kind not in ('progress', 'waiting'):
                self.data['events'] = (self.data['events'] + [self.data['last_event']])[-25:]


class GraphJobs:
    def __init__(self):
        self.lock, self.jobs = Lock(), {}

    @staticmethod
    def key(project):
        return str(project.paths.root.resolve()).casefold()

    def get(self, project):
        with self.lock:
            job = self.jobs.get(self.key(project))
            return job.snapshot() if job else None

    def start(self, project, generator_factory, settings, *, action='update'):
        if action not in ('initial', 'update', 'resume', 'rebuild'):
            raise ValueError('Unknown graph build action.')
        if not project.settings.graph_enabled:
            raise ValueError('Enable relationship graph use before building.')
        with self.lock:
            previous = self.jobs.get(self.key(project))
            if previous and previous.snapshot()['active']:
                return previous.snapshot(), False
            job = GraphJob()
            job.data['action'] = action
            self.jobs[self.key(project)] = job
            job.thread = Thread(target=self._run, args=(job, project, generator_factory, settings, action),
                                daemon=True, name=f'graph-{project.id}')
            job.thread.start()
            return job.snapshot(), True

    def cancel(self, project):
        with self.lock:
            job = self.jobs.get(self.key(project))
            if job and job.snapshot()['active']:
                job.cancel.set()
                return True
            return False

    @staticmethod
    def _run(job, project, generator_factory, settings, action):
        try:
            report = build_graph(project, generator_factory(), options=settings.extraction(),
                                 request_limits=settings.limits(), incremental=action == 'update',
                                 rebuild=action == 'rebuild', on_event=job.observe,
                                 should_cancel=job.cancel.is_set)
            result = {'status': report.status, 'summary': report.summary, 'problem': report.refused}
        except GraphCancelled as error:
            result = {'status': 'interrupted', 'problem': str(error)}
        except BaseException as error:
            result = {'status': 'failed', 'problem': str(error) or type(error).__name__}
        finally:
            with job.lock:
                job.data.update(result, active=False)


JOBS = GraphJobs()
