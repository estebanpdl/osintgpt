'''Resumable operator traversal. Model/CLI traversal keeps its existing bounded contract.'''

from collections import deque
from dataclasses import dataclass, field

from .traversal import DIRECTIONS, GraphHit, GraphPath, endpoint


def positive_depth(depth):
    if type(depth) is not int or depth < 1:
        raise ValueError('depth must be a positive integer.')
    return depth


@dataclass
class ExplorationCursor:
    start: str
    goal: str | None = None
    direction: str = 'connectivity'
    depth: int = 1
    refs: list | None = None
    queue: deque = field(init=False)
    visited: set = field(init=False)
    seen: set = field(default_factory=set)
    parents: dict = field(default_factory=dict)
    active: tuple | None = None
    path: GraphPath | None = None
    edges_examined: int = 0
    complete: bool = False
    exhausted: bool = False

    def __post_init__(self):
        positive_depth(self.depth)
        if self.direction not in DIRECTIONS:
            raise ValueError('Direction must be incoming, outgoing, or connectivity.')
        self.queue = deque([(self.start, 0)])
        self.visited = {self.start}
        if self.start == self.goal:
            self.path, self.complete = GraphPath(), True

    def expand(self, depth):
        positive_depth(depth)
        if depth <= self.depth:
            raise ValueError('Expansion depth must be greater than the previous depth.')
        self.depth = depth
        self.complete = self.path is not None or self.exhausted

    def advance(self, store, *, batch_size=100, cancelled=lambda: False):
        '''Bound each work batch, not total discoveries; retain the exact next edge cursor.'''
        positive_depth(batch_size)
        hits, work = [], 0
        while not self.complete and work < batch_size and not cancelled():
            if self.active is None:
                if not self.queue:
                    self.complete = self.exhausted = True
                    break
                node, level = self.queue[0]
                if level >= self.depth:
                    self.complete = True
                    break  # Keep the frontier for a later depth expansion.
                self.queue.popleft()
                self.active = (node, level, 0)
                work += 1  # Isolated nodes also consume batch work.
            node, level, after = self.active
            allowance = max(1, batch_size - work)
            rows = store.adjacent_page(node, self.direction, self.refs, after=after, limit=allowance + 1)
            for key, edge in rows[:allowance]:
                if cancelled():
                    return hits
                self.active = (node, level, key)
                self.edges_examined += 1
                work += 1
                other = endpoint(edge, 'target') if endpoint(edge, 'source') == node else endpoint(edge, 'source')
                step = GraphHit(edge, level + 1, node, other)
                if self.goal is None and edge.assertion_id not in self.seen:
                    self.seen.add(edge.assertion_id)
                    hits.append(step)
                if other not in self.visited:
                    self.visited.add(other)
                    self.queue.append((other, level + 1))
                    if self.goal is not None:
                        self.parents[other] = step
                if other == self.goal:
                    steps, current = [], other
                    while current != self.start:
                        previous = self.parents[current]
                        steps.append(previous)
                        current = previous.from_id
                    steps.reverse()
                    self.path = GraphPath([item.edge for item in steps], steps)
                    self.complete = True
                    return steps
            if len(rows) <= allowance:
                self.active = None
        if self.active is None and not self.queue:
            self.complete = self.exhausted = True
        elif self.active is None and self.queue[0][1] >= self.depth:
            self.complete = True
        return hits
