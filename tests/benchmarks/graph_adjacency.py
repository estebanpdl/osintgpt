'''Compare whole-graph materialization with bounded indexed adjacency reads.'''

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from osintgpt.graph import Edge, GraphStore


def measure(operation):
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - start)
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'median_ms': round(statistics.median(durations) * 1000, 3), 'peak_python_bytes': peak}


def run(sizes=(1000, 10000, 50000)):
    results = []
    for count in sizes:
        with GraphStore(':memory:') as store:
            store.add([], (Edge(f'node-{i}', f'node-{i+1}', 'relates', 'benchmark.md', f'Claim {i}')
                           for i in range(count)))
            for shape, entity in (('sparse', 'node-0'), ('hub', 'hub')):
                if shape == 'hub':
                    store.add([], (Edge('hub', f'node-{i}', 'relates', 'benchmark.md', f'Hub {i}')
                                   for i in range(count)))
                def whole():
                    return [edge for edge in store.edges()
                            if (edge.source_id or edge.source) == entity or (edge.target_id or edge.target) == entity][:30]
                old, baseline = measure(whole)
                new, indexed = measure(lambda: store.adjacent(entity, limit=30))
                assert [e.assertion_id for e in old] == [e.assertion_id for e in new]
                plans = [row[3] for row in store.connection.execute(
                    'EXPLAIN QUERY PLAN SELECT id FROM edges WHERE source_key = ? ORDER BY id LIMIT 31', (entity,))]
                results.append({'edges': store.edge_count, 'shape': shape, 'results': len(new),
                                'whole_graph': baseline, 'indexed': indexed, 'query_plan': plans})
    return {'scope': 'Offline synthetic adjacency only; excludes identity discovery and evidence verification.',
            'repetitions': 3, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
