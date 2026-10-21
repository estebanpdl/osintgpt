'''Stage measurements shared by the offline exploration benchmark (no time thresholds).'''

import gc
import json
import platform
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from graph_exploration import expected, fixture, memory, timing
from osintgpt import Project
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.graph.export import serialize_package
from osintgpt.graph.exploration import ExplorationCursor
from osintgpt.graph.inspection import OperatorQuery
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.graph.store import GraphStore
from osintgpt.ingestion import Corpus
import osintgpt.graph.operator_exploration as exploration
import osintgpt.graph.operator_exports as exports


def star_fixture(home, count=5005):
    project = Project.create('Synthetic graph', home=home).with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'notes.md').write_text('Example evidence.', encoding='utf8')
    Corpus.load(project.paths.sources).register('notes.md')
    edges = [Edge('Alpha', f'Identity {i}', 'links', 'notes.md', 'Example evidence.',
                  source_id='alpha', target_id=f'n{i}') for i in range(count)]
    with graph_for(project) as graph:
        graph.add([Entity('alpha', 'Alpha'), *[Entity(f'n{i}', f'Identity {i}') for i in range(count)]], edges)
    return project, edges


class Stages:
    '''Inclusive wall timers and call counts; nested measurements must not be added.'''

    def __init__(self, stack):
        self.values = defaultdict(lambda: {'calls': 0, 'seconds': 0.0})
        hooks = [(exploration, 'verify_claims', 'evidence_verification'),
                 (ExplorationCursor, 'advance', 'adjacency_traversal'),
                 (Corpus, 'mappings', 'corpus_mappings'), (Corpus, 'mapping_for', 'mapping_for'),
                 (OperatorQuery, 'details', 'assertion_details'),
                 (GraphStore, 'entity_candidates', 'identity_candidates'),
                 (exports, 'query_export_context', 'export_identity_context'),
                 (exports, 'selection_package', 'export_selection_copy')]
        if hasattr(GraphStore, 'entities_by_ids'):
            hooks.append((GraphStore, 'entities_by_ids', 'identity_batch'))
        for owner, name, label in hooks:
            stack.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), label)))

    def wrap(self, original, label):
        @wraps(original)
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.values[label]['calls'] += 1
                self.values[label]['seconds'] += time.perf_counter() - start
        return call

    def measure(self, operation):
        self.values.clear()
        start = time.perf_counter()
        value = operation()
        return value, {'seconds': time.perf_counter() - start, 'inclusive_stages': dict(self.values)}


def measure_stages(project, edges, start, depth, stages):
    gc.collect()
    before = memory()
    query, initialization = stages.measure(lambda: OperatorExploration(project, start, depth=depth))
    try:
        def discover():
            batches = []
            while query.state == 'running':
                begun = time.perf_counter()
                query.advance(batch_size=30 if not batches else 100)
                batches.append(time.perf_counter() - begun)
            assert query.state == 'complete', query.result
            return timing(batches)
        batches, discovery = stages.measure(discover)
        ids = [row['assertion_id'] for row in query.rows]
        assert len(ids) == len(set(ids)) and set(ids) == expected(edges, start, depth)
        page, paging = stages.measure(lambda: query.page(page_size=30))
        assert [row['assertion_id'] for row in page['rows']] == ids[:30]
        selections = {}
        for scope, selected in [('table_page', ids[:30]), ('diagram', ids[30:105]), ('all_matching', ids)]:
            package, preparation = stages.measure(lambda: query.prepare_export(scope=scope, assertion_ids=selected))
            assert [row['assertion_id'] for row in package['edges']] == selected
            assert package['metadata']['selection']['matching_total'] == len(ids)
            formats = {}
            for suffix in ('.json', '.cypherl'):
                encoded, serialized = stages.measure(lambda: serialize_package(package, suffix))
                formats[suffix] = {**serialized, 'bytes': len(encoded.encode('utf8'))}
            selections[scope] = {'preparation': preparation, 'formats': formats, 'assertions': len(selected)}
        return {'initialization': initialization, 'discovery': discovery, 'batches': batches,
                'page_30': paging, 'exports': selections, 'assertions': len(ids),
                'memory': {'before_query': before, 'after_exports': memory()}}
    finally:
        query.close()


def run_stages(output, repeats=3):
    if repeats < 1:
        raise ValueError('Use at least one repeat.')
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline benchmark forbids network/provider calls.')
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'provider_calls': 0, 'operator_projects_accessed': False, 'repeats': repeats, 'fixtures': {},
              'limitations': ['Sequential runs; fresh query caches, shared OS/process caches across repeats.',
                  'Lightweight inclusive stage timers; no cProfile or tracemalloc. Nested times must not be summed.',
                  'Backend times exclude browser/Streamlit scheduling. Browser harness measures those separately.',
                  'Process peak memory includes fixture setup and earlier repeats; not per-operation allocation.',
                  'Synthetic machine-specific observations, not capacity or model-quality guarantees.']}
    with tempfile.TemporaryDirectory(prefix='osintgpt-phase16-benchmark-') as directory, ExitStack() as stack:
        for target in ('socket.socket.connect', 'osintgpt.llm.build_generation_provider',
                       'osintgpt.llm.build_embedding_provider'):
            stack.enter_context(patch(target, forbidden))
        for name, factory, start, depth in [('versioned_3159', fixture, 'root', 12),
                                            ('star_5005', star_fixture, 'alpha', 1)]:
            project, edges = factory(Path(directory) / name)
            entry = {'assertions': len(edges), 'depth': depth, 'database_bytes':
                     (project.paths.root / 'graph.sqlite').stat().st_size, 'runs': []}
            report['fixtures'][name] = entry
            with ExitStack() as timers:
                stages = Stages(timers)
                for number in range(repeats):
                    result = measure_stages(project, edges, start, depth, stages)
                    entry['runs'].append(result)
                    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
                    print(json.dumps({'fixture': name, 'run': number + 1,
                          'discovery_seconds': result['discovery']['seconds'], 'export_seconds':
                          {key: value['preparation']['seconds'] for key, value in result['exports'].items()}}), flush=True)
    return report
