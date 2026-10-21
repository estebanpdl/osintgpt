'''Offline operator-exploration measurements over disposable, sourced synthetic graphs.'''

import argparse
import gc
import json
import platform
import tempfile
import time
import tracemalloc
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import patch

from osintgpt import Project
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.graph.inspection import query_messages
from osintgpt.graph.operator_exploration import OperatorExploration
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot


def fixture(home, identities=1000, parallel=3):
    """Use real source snapshots and quotations, without constructing any model."""
    project = Project.create('Offline exploration validation', home=home).with_settings(graph_enabled=True)
    names = {'root': 'Starting identity', 'isolated': 'Isolated identity',
             **{f'n{i}': f'Identity {i}' for i in range(identities)},
             **{f'p{i}': f'Path identity {i}' for i in range(1, 13)}}
    specs = []
    for i in range(identities):
        for j in range(parallel):
            source, target = ('root', f'n{i}') if i % 5 else (f'n{i}', 'root')
            specs.append((source, target, 'links' if j % 2 == 0 else 'reports', i % 10))
        if i % 7 == 0:
            specs.append((f'n{i}', f'n{(i + 1) % identities}', 'connects', i % 10))
    specs += [('root' if i == 1 else f'p{i-1}', f'p{i}', 'path step', i % 10) for i in range(1, 13)]
    specs += [('root', 'root', 'self description', 0), ('n1', 'n2', 'cycle', 1),
              ('n2', 'n3', 'cycle', 2), ('n3', 'n1', 'cycle', 3)]
    grouped = defaultdict(list)
    for number, (source, target, relation, document) in enumerate(specs):
        quote = f'Synthetic record {number}: {names[source]} → {names[target]}. Texto de prueba 日本語 مثال.'
        if number == 1:
            quote += ' Synthetic supporting passage.' * 3000 + ' FINAL-MARKER-日本語'
        grouped[f'source-{document}.md'].append((number, source, target, relation, quote))
    corpus, edges = Corpus.load(project.paths.sources), []
    for ref, rows in grouped.items():
        path = project.paths.root / ref
        path.write_text('\n\n'.join(row[4] for row in rows), encoding='utf-8')
        corpus.register(ref)
        snapshot = load_snapshot(path, ref=ref, cache_dir=project.paths.extracts)
        document = snapshot.document(ref)
        for number, source, target, relation, quote in rows:
            provenance = snapshot.provenance(document, quote)
            provenance['supporting'] = [{'quote': quote, 'matches': provenance['matches']}]
            edges.append(Edge(names[source], names[target], relation, ref, quote,
                source_id=source, target_id=target, evidence_quotes=(quote,), provenance=provenance,
                relation_polarity=('positive', 'negative', 'unknown')[number % 3],
                assertion=('stated', 'alleged', 'uncertain', 'hypothetical')[number % 4]))
    with graph_for(project) as graph:
        graph.add([Entity(key, name, type='synthetic') for key, name in names.items()], edges)
    return project, edges


def expected(edges, start='root', depth=12, direction='connectivity', refs=None):
    """Independent small adjacency oracle; do not call the production traversal."""
    adjacency = defaultdict(list)
    for edge in edges:
        if refs is not None and edge.ref not in refs:
            continue
        if direction != 'incoming':
            adjacency[edge.source_id].append((edge.target_id, edge.assertion_id))
        if direction != 'outgoing':
            adjacency[edge.target_id].append((edge.source_id, edge.assertion_id))
    seen, assertions, queue = {start}, set(), deque([(start, 0)])
    while queue:
        node, level = queue.popleft()
        if level >= depth:
            continue
        for other, assertion in adjacency[node]:
            assertions.add(assertion)
            if other not in seen:
                seen.add(other)
                queue.append((other, level + 1))
    return assertions


def memory():
    """Windows process counters include native allocations and earlier fixture setup."""
    if platform.system() != 'Windows':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set', 'peak_paged_pool',
                'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private_bytes')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not read(current(), ctypes.byref(counters), counters.cb):
        return None
    return {key: getattr(counters, key) for key in ('working_set', 'peak_working_set', 'private_bytes')}


def timing(samples):
    ordered = sorted(samples)
    return {'samples': len(ordered), 'total_seconds': sum(ordered),
            'p50_ms': ordered[len(ordered) // 2] * 1000,
            'p95_ms': ordered[min(len(ordered) - 1, int(len(ordered) * .95))] * 1000,
            'maximum_ms': max(ordered) * 1000}


def finish(query, batch_size):
    samples = []
    while query.state == 'running':
        started = time.perf_counter()
        query.advance(batch_size=batch_size)
        samples.append(time.perf_counter() - started)
    assert query.state == 'complete', query.result
    return samples


def measure(project, edges, batch_size=100, trace=False):
    gc.collect()
    before = memory()
    if trace:
        tracemalloc.start()
    started = time.perf_counter()
    query = OperatorExploration(project, 'root', depth=12)
    initialization = time.perf_counter() - started
    try:
        first = time.perf_counter()
        query.advance(batch_size=min(batch_size, 30))
        first = time.perf_counter() - first
        query.pause()
        paused_ids = [row['assertion_id'] for row in query.rows]
        query.advance(batch_size=batch_size)
        assert [row['assertion_id'] for row in query.rows] == paused_ids
        assert query.result['truncated'] and not query.result['inspection']['discovery_complete']
        query.resume()
        batches = [first, *finish(query, batch_size)]
        ids = [row['assertion_id'] for row in query.rows]
        assert len(ids) == len(set(ids)) and set(ids) == expected(edges)
        assert query_messages(query.result)[1] == []
        page_times, paged = [], []
        for offset in range(0, len(ids), 30):
            started = time.perf_counter()
            page = query.page(offset=offset, page_size=30)
            page_times.append(time.perf_counter() - started)
            paged.extend(row['assertion_id'] for row in page['rows'])
        assert paged == ids
        started = time.perf_counter()
        matches = query.matching(text='FINAL-MARKER-日本語', verification='found')
        search_seconds = time.perf_counter() - started
        assert len(matches) == 1 and len(matches[0]['quotation_preview']) < 65536
        started = time.perf_counter()
        detail = query.details(matches[0]['assertion_id'])
        detail_seconds = time.perf_counter() - started
        assert len(detail['evidence'].encode('utf-8')) > 65536
        assert all(row['verification']['current'] == 'found' for row in query.rows)
        exports = {}
        for scope, selected in [('table_page', ids[:30]), ('diagram', ids[30:105]), ('all_matching', ids)]:
            started = time.perf_counter()
            package = query.prepare_export(scope=scope, assertion_ids=selected)
            encoded = json.dumps(package, ensure_ascii=False).encode('utf-8')
            exports[scope] = {'seconds': time.perf_counter() - started, 'bytes': len(encoded),
                              'assertions': len(package['edges'])}
            assert [row['assertion_id'] for row in package['edges']] == selected
            assert package['metadata']['selection']['matching_total'] == len(ids)
            assert all('relation_polarity' in row and 'polarity' not in row for row in package['edges'])
            del package, encoded
        traced = dict(zip(('current_bytes', 'peak_bytes'), tracemalloc.get_traced_memory())) if trace else None
        during = memory()
        result = {'initialization_seconds': initialization, 'batch_size': batch_size,
                  'first_batch_size': min(batch_size, 30), 'batches': timing(batches),
                  'pages_30_rows': timing(page_times), 'full_quote_search_seconds': search_seconds,
                  'complete_evidence_seconds': detail_seconds, 'exports': exports, 'discovered_assertions': len(ids),
                  'identities_visited': query.result['nodes_visited'],
                  'memory': {'before_query': before, 'after_exports': during, 'python_allocations': traced}}
    finally:
        query.close()
        if trace:
            tracemalloc.stop()
    return result


def run(identities=1000, parallel=3, batch_size=100, trace=False):
    if identities < 30 or parallel < 2 or batch_size < 1:
        raise ValueError('Use at least 30 identities, two parallel assertions, and a positive batch size.')
    def forbidden(*args, **kwargs):
        raise AssertionError('This benchmark forbids network/provider calls.')
    with tempfile.TemporaryDirectory(prefix='osintgpt-exploration-benchmark-') as directory, \
            patch('socket.socket.connect', forbidden), \
            patch('osintgpt.llm.build_generation_provider', forbidden), \
            patch('osintgpt.llm.build_embedding_provider', forbidden):
        started = time.perf_counter()
        project, edges = fixture(Path(directory), identities, parallel)
        setup_seconds = time.perf_counter() - started
        database_bytes = (project.paths.root / 'graph.sqlite').stat().st_size
        metrics = measure(project, edges, batch_size, trace)
        return {'kind': 'synthetic_operator_exploration', 'provider_calls': 0, 'operator_projects_modified': False,
                'environment': {'python': platform.python_version(), 'platform': platform.platform()},
                'configuration': {'identities': identities, 'parallel_assertions': parallel, 'depth': 12,
                                  'tracemalloc': trace},
                'fixture': {'assertions': len(edges), 'database_bytes': database_bytes, 'setup_seconds': setup_seconds,
                            'longest_quote_bytes': max(len(edge.evidence.encode('utf-8')) for edge in edges)},
                'measurements': metrics,
                'limitations': ['Synthetic topology and local text sources; not a real-corpus or model-quality benchmark.',
                    'Batch timings exclude Streamlit scheduling and browser rendering.',
                    'Tracemalloc timings include instrumentation overhead when enabled.',
                    'Python allocation tracking excludes SQLite/native/browser allocations.',
                    'Windows peak working set covers the entire process, including imports and fixture creation.',
                    'Export timings include complete evidence preparation and JSON encoding.',
                    'Results are observations on this machine, not supported-capacity thresholds.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identities', type=int, default=1000)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--trace-memory', action='store_true')
    parser.add_argument('--stages', action='store_true', help='Measure both Phase 16 fixtures with inclusive stage timers.')
    parser.add_argument('--repeats', type=int, default=3, help='Sequential fresh-query repetitions for --stages.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stages:
        from graph_performance import run_stages
        run_stages(args.output, args.repeats)
        raise SystemExit(0)
    report = run(args.identities, args.parallel, args.batch_size, args.trace_memory)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['measurements'], ensure_ascii=False))
