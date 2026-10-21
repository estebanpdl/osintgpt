// Bundled layout/download helpers are prepended by graph_chart.py. Source text uses textContent.
export default function diagram(component) {
    const {data, parentElement, setStateValue, setTriggerValue} = component;
    // V2 can call the renderer again for updated data without unmounting the host.
    const previous = parentElement.querySelector('.diagram');
    const tableMode = Boolean(previous?.dataset.scope === data.scope && previous.classList.contains('table-view'));
    previous?.dispose?.(true);
    const state = layoutState(data), NS = 'http://www.w3.org/2000/svg';
    let animation = null, zoomTimer = null, gesture = null, selected = data.selected_id;
    // Preserve the fullscreen element across Streamlit reruns (drag, selection, layout).
    const root = previous || document.createElement('div');
    root.replaceChildren();
    root.className = 'diagram';
    root.dataset.scope = data.scope;
    if (!previous) parentElement.appendChild(root);
    const element = (tag, attrs = {}, parent = root, text = null) => {
        const svgTag = ['svg', 'g', 'defs', 'marker', 'path', 'circle', 'text', 'title'].includes(tag);
        const el = svgTag ? document.createElementNS(NS, tag) : document.createElement(tag);
        for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
        if (text !== null) el.textContent = text;
        parent.appendChild(el);
        return el;
    };
    const viewport = element('div', {class: 'graph-viewport'});
    const toolbar = element('div', {class: 'toolbar', role: 'toolbar', 'aria-label': 'Graph controls'}, viewport);
    const icons = {
        rearrange: 'M3 6h3c6 0 6 12 12 12h3m-4-4 4 4-4 4M3 18h3c6 0 6-12 12-12h3m-4-4 4 4-4 4',
        reset: 'M3 10a9 9 0 1 1 2 8M3 3v7h7',
        full: 'M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5',
        exit: 'M3 8h5V3m8 0v5h5M3 16h5v5m8 0v-5h5',
        table: 'M3 3h18v18H3zM3 9h18M3 15h18M9 3v18',
        graph: 'M6 6l12 12M18 6 6 18M3 3h6v6H3zM15 3h6v6h-6zM3 15h6v6H3zM15 15h6v6h-6z',
        png: 'M3 6h4l2-3h6l2 3h4v15H3zM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
        svg: 'M12 3v12m-4-4 4 4 4-4M4 17v4h16v-4',
    };
    const iconLabel = (button, label, icon) => {
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);
        button.setAttribute('data-tooltip', label);
        button.replaceChildren();
        const glyph = element('svg', {viewBox: '0 0 24 24', 'aria-hidden': 'true', focusable: 'false'}, button);
        element('path', {d: icons[icon]}, glyph);
    };
    const action = (label, icon) => {
        const button = element('button', {type: 'button'}, toolbar);
        iconLabel(button, label, icon);
        return button;
    };
    const rearrange = action('Rearrange layout', 'rearrange');
    const reset = action('Reset layout', 'reset');
    const fullscreen = action('Full screen', 'full');
    const toggle = action('View data', 'table');
    const png = action('Download PNG', 'png');
    const vector = action('Download SVG', 'svg');
    const status = element('span', {class: 'layout-status', role: 'status'}, viewport, 'Layout ready.');
    const legend = element('div', {class: 'legend', 'aria-label': 'Graph legend'}, viewport);
    const item = (label, cls) => {
        const span = element('span', {}, legend);
        element('i', {class: 'swatch ' + cls, 'aria-hidden': 'true'}, span);
        element('span', {}, span, label);
    };
    item(data.nodes.some(n => n.id === data.starting_id) ? 'Starting identity' : 'Starting identity (outside displayed filters)', 'starting');
    item('Other identity', 'other');
    element('strong', {}, legend, 'Relation polarity:');
    for (const polarity of ['positive', 'negative', 'unknown']) item(polarity, 'polarity ' + polarity);
    const svg = element('svg', {class: 'graph', viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
        'aria-label': 'Relationship graph. Drag nodes, pan empty space, or select relationship dots.'}, viewport);
    const defs = element('defs', {}, svg);
    // Component-provided keys keep marker IDs unique when several diagrams are mounted.
    const markerPrefix = 'arrow-' + String(component.key).replace(/[^a-zA-Z0-9_-]/g, '_');
    for (const polarity of ['positive', 'negative', 'unknown']) {
        const marker = element('marker', {id: markerPrefix + polarity, viewBox: '0 0 10 10', refX: 9,
            refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto', markerUnits: 'strokeWidth'}, defs);
        element('path', {d: 'M 0 0 L 10 5 L 0 10 z', class: 'arrow ' + polarity}, marker);
    }
    const scene = element('g', {}, svg), slots = edgeSlots(data.edges), edges = [], nodes = [];
    const save = () => setStateValue('layout', snapshot(data.scope, state));
    const stop = () => { if (animation !== null) cancelAnimationFrame(animation); animation = null; };
    const pick = id => {
        selected = id;
        draw();
        setTriggerValue('selected', {scope: data.scope, assertion_id: id});
    };
    const tablePanel = element('div', {class: 'diagram-data', tabindex: 0, 'aria-label': 'Displayed assertion data'}, viewport);
    element('p', {}, tablePanel, `${data.edges.length} displayed assertions. Select Inspect to open the evidence panel below the diagram; leave full screen to read it.`);
    const table = element('table', {}, tablePanel);
    element('caption', {}, table, 'Data for the displayed diagram');
    const header = element('tr', {}, element('thead', {}, table));
    const columns = [['source', 'Source'], ['relation', 'Relationship'], ['target', 'Target'],
        ['relation_polarity', 'Relation polarity'], ['assertion', 'Qualification'], ['ref', 'Document'],
        ['assertion_id', 'Assertion ID'], ['source_id', 'Source ID'], ['target_id', 'Target ID']];
    element('th', {scope: 'col'}, header, 'Evidence');
    for (const [, label] of columns) element('th', {scope: 'col'}, header, label);
    const body = element('tbody', {}, table), tableRows = [];
    for (const edge of data.edges) {
        const row = element('tr', {'data-assertion-id': edge.assertion_id}, body);
        const button = element('button', {type: 'button', 'aria-label': 'Inspect ' + edge.assertion_id},
            element('td', {}, row), 'Inspect');
        button.onclick = () => pick(edge.assertion_id);
        for (const [key] of columns) element('td', {}, row, edge[key]);
        tableRows.push({row, button, id: edge.assertion_id});
    }
    const showTable = value => {
        root.classList.toggle('table-view', value);
        iconLabel(toggle, value ? 'View graph' : 'View data', value ? 'graph' : 'table');
        toggle.setAttribute('aria-pressed', String(value));
        tablePanel.hidden = !value;
        rearrange.disabled = reset.disabled = value;
    };
    toggle.onclick = () => showTable(!root.classList.contains('table-view'));
    showTable(tableMode);
    const fullscreenChanged = () => {
        iconLabel(fullscreen, root.matches(':fullscreen') ? 'Exit full screen' : 'Full screen',
                  root.matches(':fullscreen') ? 'exit' : 'full');
        fullscreen.setAttribute('aria-pressed', String(root.matches(':fullscreen')));
    };
    fullscreen.onclick = async () => {
        try {
            if (root.matches(':fullscreen')) await document.exitFullscreen();
            else await root.requestFullscreen();
        } catch { status.textContent = 'Full screen is unavailable in this browser. You can still pan and zoom.'; }
    };
    document.addEventListener('fullscreenchange', fullscreenChanged);
    fullscreenChanged();
    for (const [button, format] of [[png, 'png'], [vector, 'svg']]) {
        button.onclick = async () => {
            button.disabled = true;
            try {
                await downloadDiagram(svg, legend, data.starting_id, format);
                status.textContent = format.toUpperCase() + ' download prepared.';
            } catch { status.textContent = 'Could not prepare the image. Try Download SVG or retry.'; }
            finally { button.disabled = false; }
        };
    }
    for (const edge of data.edges) {
        const polarity = ['positive', 'negative', 'unknown'].includes(edge.relation_polarity) ? edge.relation_polarity : 'unknown';
        const path = element('path', {class: 'edge ' + polarity, 'marker-end': `url(#${markerPrefix + polarity})`,
            'data-source-id': edge.source_id, 'data-target-id': edge.target_id,
            'data-assertion-id': edge.assertion_id}, scene);
        const dot = element('circle', {class: 'claim ' + polarity, r: 7, tabindex: 0, role: 'button',
            'data-assertion-id': edge.assertion_id,
            'aria-label': `${edge.source} — ${edge.relation} → ${edge.target}. ${edge.assertion}. Inspect evidence.`}, scene);
        element('title', {}, dot, `${edge.source} — ${edge.relation} → ${edge.target}\nRelation polarity: ${edge.relation_polarity}\nAssertion: ${edge.assertion}\n${edge.ref}\n${edge.assertion_id}`);
        dot.onclick = () => pick(edge.assertion_id);
        dot.onkeydown = event => {
            if (['Enter', ' '].includes(event.key)) { event.preventDefault(); pick(edge.assertion_id); }
        };
        edges.push({edge, path, dot});
    }
    for (const node of data.nodes) {
        const starting = node.id === data.starting_id;
        const group = element('g', {class: 'node' + (starting ? ' starting' : ''), tabindex: 0,
            'data-entity-id': node.id, role: 'button',
            'aria-label': `${node.name}${starting ? ', starting identity' : ''}. Drag or use arrow keys to move.`}, scene);
        element('circle', {r: starting ? 13 : 11}, group);
        element('text', {y: -21}, group, Array.from(node.name).slice(0, 40).join(''));
        element('title', {}, group, `${node.name}\n${node.id}${starting ? '\nStarting identity' : ''}`);
        group.onkeydown = event => {
            const moves = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
            if (!moves[event.key]) return;
            event.preventDefault(); stop();
            const p = state.positions.get(node.id), step = event.shiftKey ? 20 : 5;
            p.x += moves[event.key][0] * step; p.y += moves[event.key][1] * step;
            draw(); save();
        };
        nodes.push({node, group});
    }
    function draw() {
        const {x, y, k} = state.view;
        scene.setAttribute('transform', `translate(${x} ${y}) scale(${k})`);
        for (const {row, button, id} of tableRows) {
            row.classList.toggle('selected', selected === id);
            button.setAttribute('aria-pressed', String(selected === id));
        }
        for (const {node, group} of nodes) {
            const p = state.positions.get(node.id);
            group.setAttribute('transform', `translate(${p.x} ${p.y})`);
        }
        for (const {edge, path, dot} of edges) {
            const geometry = edgeGeometry(edge, state.positions, slots.get(edge.assertion_id));
            path.setAttribute('d', geometry.path);
            dot.setAttribute('cx', geometry.x); dot.setAttribute('cy', geometry.y);
            dot.classList.toggle('selected', selected === edge.assertion_id);
            dot.setAttribute('aria-pressed', String(selected === edge.assertion_id));
        }
    }
    const point = event => {
        const p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse());
    };
    svg.onpointerdown = event => {
        if (event.button !== 0 || gesture || event.target.closest('.claim')) return;
        stop(); clearTimeout(zoomTimer);
        const node = event.target.closest('.node'), at = point(event);
        gesture = {pointer: event.pointerId, id: node?.getAttribute('data-entity-id'), at,
                   view: {...state.view}, moved: false};
        if (gesture.id) gesture.position = {...state.positions.get(gesture.id)};
        svg.setPointerCapture(event.pointerId);
        svg.classList.add('moving'); event.preventDefault();
    };
    svg.onpointermove = event => {
        if (!gesture || event.pointerId !== gesture.pointer) return;
        const at = point(event), dx = at.x - gesture.at.x, dy = at.y - gesture.at.y;
        gesture.moved ||= Math.hypot(dx, dy) > 1;
        if (gesture.id) state.positions.set(gesture.id,
            {x: gesture.position.x + dx / state.view.k, y: gesture.position.y + dy / state.view.k});
        else state.view = {...gesture.view, x: gesture.view.x + dx, y: gesture.view.y + dy};
        draw();
    };
    const end = event => {
        if (!gesture || event.pointerId !== gesture.pointer) return;
        const moved = gesture.moved;
        gesture = null; svg.classList.remove('moving');
        if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
        if (moved) { status.textContent = 'Layout adjusted.'; save(); }
    };
    svg.onpointerup = end; svg.onpointercancel = end; svg.onlostpointercapture = end;
    svg.addEventListener('wheel', event => {
        event.preventDefault();
        if (gesture) return;
        stop();
        const at = point(event), old = state.view, k = Math.max(.15, Math.min(6, old.k * Math.exp(-event.deltaY * .001)));
        state.view = {k, x: at.x - (at.x - old.x) * k / old.k, y: at.y - (at.y - old.y) * k / old.k};
        draw(); clearTimeout(zoomTimer); zoomTimer = setTimeout(save, 180);
    }, {passive: false});
    rearrange.onclick = () => {
        stop(); clearTimeout(zoomTimer); state.view = {x: 0, y: 0, k: 1};
        status.textContent = 'Rearranging displayed nodes…';
        let step = 0;
        const tick = () => {
            for (let i = 0; i < 8; i++, step++) forceStep(data.nodes, data.edges, state.positions, 1 - step / 200);
            draw();
            if (step < 160) animation = requestAnimationFrame(tick);
            else { animation = null; status.textContent = 'Layout rearranged.'; save(); }
        };
        animation = requestAnimationFrame(tick);
    };
    reset.onclick = () => {
        stop(); clearTimeout(zoomTimer);
        const initial = circularPositions(data.nodes.map(node => node.id));
        for (const node of data.nodes) state.positions.set(node.id, initial.get(node.id));
        state.view = {x: 0, y: 0, k: 1}; status.textContent = 'Circular layout restored.';
        draw(); save();
    };
    draw();
    const dispose = (keepRoot = false) => {
        stop(); clearTimeout(zoomTimer);
        document.removeEventListener('fullscreenchange', fullscreenChanged);
        if (!keepRoot) root.remove();
    };
    root.dispose = dispose;
    // A stale cleanup from an earlier render must not remove the new controller's root.
    return () => { if (root.dispose === dispose) dispose(); };
}
