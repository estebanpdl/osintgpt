'''Bounded, directed traversal of selected identities through indexed adjacency reads.'''

from collections import deque
from dataclasses import dataclass, field

from .records import Edge, merge_key

MAX_DEPTH = 4
MAX_RESULTS = 60
MAX_NODES = 256
MAX_EDGES = 2000
DIRECTIONS = ('incoming', 'outgoing', 'connectivity')


def bounded_integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}.')
    return value


@dataclass(frozen=True)
class GraphHit:
    edge: Edge
    depth: int = 1
    from_id: str = ''
    to_id: str = ''

    @property
    def ref(self):
        return self.edge.ref

    @property
    def evidence(self):
        return self.edge.evidence

    @property
    def against_direction(self):
        return bool(self.from_id and self.from_id != endpoint(self.edge, 'source'))


@dataclass(frozen=True)
class GraphPath:
    edges: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    @property
    def length(self):
        return len(self.edges)

    @property
    def refs(self):
        return list(dict.fromkeys(edge.ref for edge in self.edges))

    def __len__(self):
        return len(self.edges)


@dataclass
class Traversal:
    hits: list = field(default_factory=list)
    path: GraphPath | None = None
    reasons: list = field(default_factory=list)
    nodes_visited: int = 0
    edges_examined: int = 0

    @property
    def truncated(self):
        return bool(self.reasons)

    def stop(self, reason):
        if reason not in self.reasons:
            self.reasons.append(reason)


class AmbiguousEntity(ValueError):
    def __init__(self, query, candidates, truncated=False):
        self.query, self.candidates, self.truncated = query, candidates, truncated
        super().__init__('Ambiguous entity name; select an ID: ' + ', '.join(
            f'{entity.key} ({entity.name})' for entity in candidates))


def matching_entities(store, value):
    return store.entity_candidates(value)[0]


def resolve_entity(store, value):
    candidates = matching_entities(store, value)
    if len(candidates) > 1:
        raise AmbiguousEntity(value, candidates[:20], len(candidates) > 20)
    return candidates[0].key if candidates else None


def endpoint(edge, side):
    return getattr(edge, side + '_id') or merge_key(getattr(edge, side))


def walk(store, start, *, goal=None, direction='connectivity', depth=2, limit=20,
         refs=None, max_nodes=MAX_NODES, max_edges=MAX_EDGES):
    if direction not in DIRECTIONS:
        raise ValueError('Direction must be incoming, outgoing, or connectivity.')
    bounded_integer(depth, 'depth', 1, MAX_DEPTH)
    bounded_integer(limit, 'limit', 1, MAX_RESULTS)
    bounded_integer(max_nodes, 'max_nodes', 1, MAX_NODES)
    bounded_integer(max_edges, 'max_edges', 1, MAX_EDGES)
    result = Traversal(nodes_visited=1)
    if start == goal:
        result.path = GraphPath()
        return result
    queue, visited, seen_edges = deque([(start, 0, [])]), {start}, set()
    while queue:
        node, level, chain = queue.popleft()
        if level >= depth:
            result.stop('depth_boundary')
            continue
        remaining = max_edges - result.edges_examined
        if remaining <= 0:
            result.stop('edge_limit')
            break
        edges = store.adjacent(node, direction, refs, limit=remaining + 1)
        for edge in edges[:remaining]:
            result.edges_examined += 1
            other = endpoint(edge, 'target') if endpoint(edge, 'source') == node else endpoint(edge, 'source')
            step = GraphHit(edge, level + 1, node, other)
            if other not in visited and len(visited) == max_nodes:
                result.stop('node_limit')
                return result
            if goal is not None:
                if other == goal:
                    steps = chain + [step]
                    if len(steps) > limit:
                        result.stop('result_limit')
                        return result
                    result.path = GraphPath([hit.edge for hit in steps], steps)
                    result.nodes_visited = len(visited | {other})
                    return result
            elif edge.assertion_id not in seen_edges:
                if len(result.hits) == limit:
                    result.stop('result_limit')
                    return result
                seen_edges.add(edge.assertion_id)
                result.hits.append(step)
            if other not in visited:
                visited.add(other)
                result.nodes_visited = len(visited)
                queue.append((other, level + 1, chain + [step] if goal is not None else []))
        if len(edges) > remaining:
            result.stop('edge_limit')
            break
    return result


def neighbors(store, entity, limit=30, refs=None, *, direction='connectivity'):
    return neighborhood(store, entity, depth=1, limit=limit, refs=refs, direction=direction)


def neighborhood(store, entity, depth=2, limit=60, refs=None, *, direction='connectivity'):
    start = resolve_entity(store, entity)
    return walk(store, start, direction=direction, depth=depth, limit=limit, refs=refs).hits if start else []


def path_between(store, source, target, max_depth=MAX_DEPTH, refs=None, *, direction='connectivity'):
    start, goal = resolve_entity(store, source), resolve_entity(store, target)
    if not start or not goal:
        return None
    return walk(store, start, goal=goal, direction=direction, depth=max_depth, refs=refs).path
