// Pure geometry shared by the browser component and offline tests. No graph writes.
export const WIDTH = 900, HEIGHT = 560;

export function circularPositions(ids) {
    return new Map(ids.map((id, i) => {
        const angle = 2 * Math.PI * i / Math.max(ids.length, 1);
        return [id, {x: WIDTH / 2 + (ids.length > 1 ? 285 * Math.cos(angle) : 0),
                     y: HEIGHT / 2 + (ids.length > 1 ? (HEIGHT / 2 - 65) * Math.sin(angle) : 0)}];
    }));
}

export function layoutState(data) {
    // Arrange the selected diagram, not thousands of undisplayed identities.
    const positions = circularPositions(data.nodes.map(node => node.id));
    const allowed = new Set(data.all_ids);
    const saved = data.layout?.scope === data.scope ? data.layout : null;
    for (const p of saved?.positions || []) {
        if (allowed.has(p.id) && Number.isFinite(p.x) && Number.isFinite(p.y)) {
            positions.set(p.id, {x: p.x, y: p.y});
        }
    }
    return {positions, view: {...(saved?.view || {x: 0, y: 0, k: 1})}};
}

export function snapshot(scope, state) {
    return {scope, positions: Array.from(state.positions, ([id, p]) => ({id, ...p})), view: {...state.view}};
}

export function edgeSlots(edges) {
    const groups = new Map();
    for (const edge of edges) {
        const pair = JSON.stringify([edge.source_id, edge.target_id].sort());
        if (!groups.has(pair)) groups.set(pair, []);
        groups.get(pair).push(edge);
    }
    const slots = new Map();
    for (const group of groups.values()) {
        group.sort((a, b) => a.assertion_id.localeCompare(b.assertion_id));
        group.forEach((edge, i) => slots.set(edge.assertion_id, i - (group.length - 1) / 2));
    }
    return slots;
}

export function edgeGeometry(edge, positions, slot) {
    const a = positions.get(edge.source_id), b = positions.get(edge.target_id);
    if (edge.source_id === edge.target_id) {
        const r = 35 + Math.abs(slot) * 22, side = slot < 0 ? -1 : 1;
        return {path: `M ${a.x - 8} ${a.y - 7} C ${a.x - r * 2} ${a.y - r * side * 3} ${a.x + r * 2} ${a.y - r * side * 3} ${a.x + 8} ${a.y - 7}`,
                x: a.x, y: a.y - (7 + r * side * 9) / 4};
    }
    // Use a canonical pair orientation so opposing arrows do not overlap.
    const [left, right] = edge.source_id < edge.target_id ? [a, b] : [b, a];
    const dx = right.x - left.x, dy = right.y - left.y, distance = Math.hypot(dx, dy) || 1;
    const bend = slot * 44;
    const control = {x: (a.x + b.x) / 2 - dy / distance * bend,
                     y: (a.y + b.y) / 2 + dx / distance * bend};
    const offset = (p, q, radius) => {
        const length = Math.hypot(q.x - p.x, q.y - p.y) || 1;
        return {x: p.x + (q.x - p.x) / length * radius, y: p.y + (q.y - p.y) / length * radius};
    };
    const from = offset(a, control, 12), to = offset(b, control, 17);
    return {path: `M ${from.x} ${from.y} Q ${control.x} ${control.y} ${to.x} ${to.y}`,
            x: (from.x + 2 * control.x + to.x) / 4, y: (from.y + 2 * control.y + to.y) / 4};
}

export function forceStep(nodes, edges, positions, cooling) {
    const delta = new Map(nodes.map(({id}) => [id, {x: 0, y: 0}]));
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const a = positions.get(nodes[i].id), b = positions.get(nodes[j].id);
            let dx = a.x - b.x, dy = a.y - b.y;
            if (Math.hypot(dx, dy) < .01) { dx = .2 * (i + 1); dy = -.3 * (j + 1); }
            const distance = Math.max(Math.hypot(dx, dy), 1), force = 7500 / (distance * distance);
            const da = delta.get(nodes[i].id), db = delta.get(nodes[j].id);
            da.x += dx / distance * force; da.y += dy / distance * force;
            db.x -= dx / distance * force; db.y -= dy / distance * force;
        }
    }
    const pairs = new Set();
    for (const edge of edges) {
        const pair = JSON.stringify([edge.source_id, edge.target_id].sort());
        if (pairs.has(pair) || edge.source_id === edge.target_id) continue;
        pairs.add(pair); // Repeated assertions do not imply a stronger spring.
        const a = positions.get(edge.source_id), b = positions.get(edge.target_id);
        const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
        const pull = (distance - 140) * .025;
        const da = delta.get(edge.source_id), db = delta.get(edge.target_id);
        da.x += dx / distance * pull; da.y += dy / distance * pull;
        db.x -= dx / distance * pull; db.y -= dy / distance * pull;
    }
    for (const {id} of nodes) {
        const p = positions.get(id), d = delta.get(id);
        d.x += (WIDTH / 2 - p.x) * .008; d.y += (HEIGHT / 2 - p.y) * .008;
        p.x = Math.max(45, Math.min(WIDTH - 45, p.x + Math.max(-12, Math.min(12, d.x)) * cooling));
        p.y = Math.max(45, Math.min(HEIGHT - 45, p.y + Math.max(-12, Math.min(12, d.y)) * cooling));
    }
}
