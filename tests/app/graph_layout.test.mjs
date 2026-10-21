import test from 'node:test';
import assert from 'node:assert/strict';
import {circularPositions, edgeGeometry, edgeSlots, forceStep, layoutState, snapshot} from '../../osintgpt/app/assets/graph_layout.mjs';

const edges = [
    {assertion_id: 'forward', source_id: 'a', target_id: 'b'},
    {assertion_id: 'reverse', source_id: 'b', target_id: 'a'},
    {assertion_id: 'parallel', source_id: 'a', target_id: 'b'},
    {assertion_id: 'loop', source_id: 'a', target_id: 'a'},
];

test('parallel, reversed, and self assertions keep distinct selectable geometry after moving nodes', () => {
    const positions = circularPositions(['a', 'b']), slots = edgeSlots(edges);
    const geometry = () => edges.map(edge => edgeGeometry(edge, positions, slots.get(edge.assertion_id)));
    const before = geometry();
    assert.equal(new Set(before.map(p => `${p.x},${p.y}`)).size, edges.length);
    positions.set('a', {x: 300, y: 120});
    const after = geometry();
    assert.equal(new Set(after.map(p => p.path)).size, edges.length);
    after.forEach((p, i) => { assert.notEqual(p.path, before[i].path); assert.ok(Number.isFinite(p.x + p.y)); });
    assert.deepEqual(edgeSlots([...edges].reverse()), slots);
});

test('force layout changes visible geometry only, handles coincident nodes, and preserves assertion data', () => {
    const positions = new Map([['a', {x: 400, y: 200}], ['b', {x: 400, y: 200}], ['hidden', {x: 1, y: 2}]]);
    const original = structuredClone(edges);
    for (let i = 0; i < 160; i++) forceStep([{id: 'a'}, {id: 'b'}], edges, positions, 1 - i / 200);
    assert.ok(Math.hypot(positions.get('a').x - positions.get('b').x, positions.get('a').y - positions.get('b').y) > 60);
    assert.deepEqual(positions.get('hidden'), {x: 1, y: 2});
    assert.deepEqual(edges, original);
});

test('filter rerenders retain hidden nodes and viewport while a new scope resets them', () => {
    const data = {scope: 'same', all_ids: ['a', 'b'], nodes: [{id: 'a'}]};
    const state = layoutState(data);
    state.positions.set('b', {x: -10, y: 800}); state.view = {x: 15, y: 5, k: 2};
    const saved = snapshot('same', state);
    assert.deepEqual(layoutState({...data, layout: saved}), state);
    assert.deepEqual(layoutState({...data, scope: 'new', layout: saved}).positions, circularPositions(['a']));
});

test('a selected diagram spreads its visible nodes independently of a much larger query', () => {
    const all_ids = Array.from({length: 800}, (_, i) => String(i));
    const ids = all_ids.slice(0, 10), nodes = ids.map(id => ({id}));
    assert.deepEqual(layoutState({scope: 'large', all_ids, nodes}).positions, circularPositions(ids));
});

test('a bounded 60-assertion diagram stays finite across cycles and disconnected components', () => {
    const ids = Array.from({length: 120}, (_, i) => String(i));
    const rows = Array.from({length: 60}, (_, i) => ({assertion_id: String(i), source_id: ids[2 * i], target_id: ids[2 * i + 1]}));
    const positions = circularPositions(ids);
    for (let i = 0; i < 160; i++) forceStep(ids.map(id => ({id})), rows, positions, 1 - i / 200);
    for (const p of positions.values()) assert.ok(Number.isFinite(p.x + p.y));
    assert.equal(positions.size, 120);
});
