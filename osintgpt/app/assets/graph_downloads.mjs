// Export the displayed diagram locally. Source text is serialized as text, never markup.
export function diagramImage(svg, legend, startingId) {
    const NS = 'http://www.w3.org/2000/svg';
    const width = Math.max(900, Math.round(svg.clientWidth)), height = Math.round(svg.clientHeight) || HEIGHT;
    const output = document.createElementNS(NS, 'svg');
    output.setAttribute('xmlns', NS);
    output.setAttribute('width', width); output.setAttribute('height', height + 106);
    output.setAttribute('viewBox', `0 0 ${width} ${height + 106}`);
    const background = getComputedStyle(svg).backgroundColor, color = getComputedStyle(legend).color;
    const add = (tag, attrs, text) => {
        const el = document.createElementNS(NS, tag);
        for (const [name, value] of Object.entries(attrs)) el.setAttribute(name, value);
        if (text !== undefined) el.textContent = text;
        output.appendChild(el); return el;
    };
    add('rect', {width: '100%', height: '100%', fill: background});
    const label = {fill: color, 'font-family': 'sans-serif', 'font-size': 13};
    add('text', {...label, x: 16, y: 24, 'font-size': 16, 'font-weight': 'bold'}, 'Relationship graph — displayed assertions');
    add('text', {...label, x: 16, y: 46}, 'Starting identity: ' + startingId);
    let x = 16;
    for (const entry of legend.children) {
        const swatch = entry.querySelector('.swatch');
        if (swatch) {
            const style = getComputedStyle(swatch);
            add('circle', {cx: x + 5, cy: 65, r: 5, fill: style.backgroundColor,
                stroke: swatch.classList.contains('starting') ? color : style.backgroundColor});
            x += 17;
        }
        const text = add('text', {...label, x, y: 70}, entry.textContent);
        // The six fixed legend labels fit within the minimum export width.
        x += text.textContent.length * 7 + 18;
    }
    const clone = svg.cloneNode(true);
    const originals = [svg, ...svg.querySelectorAll('*')], copies = [clone, ...clone.querySelectorAll('*')];
    const properties = ['fill', 'stroke', 'stroke-width', 'stroke-linejoin', 'stroke-linecap',
        'font-family', 'font-size', 'font-weight', 'text-anchor', 'paint-order', 'opacity'];
    originals.forEach((element, i) => {
        const style = getComputedStyle(element);
        for (const name of properties) copies[i].style.setProperty(name, style.getPropertyValue(name));
    });
    clone.setAttribute('x', 0); clone.setAttribute('y', 82);
    clone.setAttribute('width', width); clone.setAttribute('height', height);
    clone.removeAttribute('class');
    output.appendChild(clone);
    add('text', {...label, x: 16, y: height + 101, 'font-size': 11},
        'Arrows retain assertion direction. Layout does not indicate importance or evidential strength.');
    return {text: new XMLSerializer().serializeToString(output), width, height: height + 106};
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = filename;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function downloadDiagram(svg, legend, startingId, format) {
    const image = diagramImage(svg, legend, startingId);
    const blob = new Blob([image.text], {type: 'image/svg+xml;charset=utf-8'});
    if (format === 'svg') { downloadBlob(blob, 'relationship-graph.svg'); return; }
    const url = URL.createObjectURL(blob);
    try {
        const bitmap = new Image();
        bitmap.src = url; await bitmap.decode();
        const canvas = document.createElement('canvas');
        canvas.width = image.width * 2; canvas.height = image.height * 2;
        canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        const png = await new Promise(done => canvas.toBlob(done, 'image/png'));
        if (!png) throw new Error('PNG conversion failed');
        downloadBlob(png, 'relationship-graph.png');
    } finally { URL.revokeObjectURL(url); }
}
