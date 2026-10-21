// Run with an already installed Playwright/Chromium; all HTTP requests stay on loopback.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {once} from 'node:events';
import {tmpdir} from 'node:os';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';

const {chromium} = await import(process.env.OSINTGPT_PLAYWRIGHT_MODULE
    ? pathToFileURL(process.env.OSINTGPT_PLAYWRIGHT_MODULE).href : 'playwright');
const reserve = createServer();
await new Promise(done => reserve.listen(0, '127.0.0.1', done));
const port = reserve.address().port;
await new Promise(done => reserve.close(done));
const url = `http://127.0.0.1:${port}`;
const exploration = process.env.OSINTGPT_TEST_SCENARIO === 'exploration';
const measurement = {scenario: exploration ? 'exploration' : 'layout',
    theme: process.env.OSINTGPT_TEST_THEME || 'light', external_http_requests: 0};
const server = spawn(process.env.OSINTGPT_TEST_PYTHON || 'python', ['-m', 'streamlit', 'run',
    exploration ? 'tests/app/fixtures/graph_exploration_app.py' : 'tests/app/fixtures/graph_layout_app.py', '--server.headless=true', '--server.address=127.0.0.1',
    `--server.port=${port}`, '--server.fileWatcherType=none', '--browser.gatherUsageStats=false'],
    {cwd: resolve('.'), windowsHide: true, stdio: 'pipe'});
let logs = '';
server.stdout.on('data', data => { logs += data; });
server.stderr.on('data', data => { logs += data; });
let browser, page;
const screenshotDir = await mkdtemp(join(tmpdir(), 'osintgpt-layout-browser-'));
try {
    const deadline = Date.now() + 45000;
    while (true) {
        try { if ((await fetch(url + '/_stcore/health')).ok) break; } catch {}
        if (Date.now() > deadline || server.exitCode !== null) throw new Error('Streamlit did not start: ' + logs);
        await new Promise(done => setTimeout(done, 200));
    }
    browser = await chromium.launch({headless: true, ...(process.env.OSINTGPT_TEST_CHROMIUM
        ? {executablePath: process.env.OSINTGPT_TEST_CHROMIUM} : {})});
    page = await browser.newPage({viewport: {width: 1280, height: 1050},
        colorScheme: process.env.OSINTGPT_TEST_THEME || 'light'});
    const protocol = await page.context().newCDPSession(page);
    await protocol.send('Performance.enable');
    const memory = async () => {
        const {metrics} = await protocol.send('Performance.getMetrics');
        return Object.fromEntries(metrics.filter(row => ['JSHeapUsedSize', 'JSHeapTotalSize', 'Nodes'].includes(row.name))
            .map(row => [row.name, row.value]));
    };
    measurement.chromium = browser.version();
    measurement.before_navigation = await memory();
    const errors = [], external = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('dialog', dialog => { errors.push('Unexpected dialog: ' + dialog.message()); dialog.dismiss(); });
    await page.route('**/*', route => {
        const host = new URL(route.request().url()).hostname;
        if (host === '127.0.0.1' || host === 'localhost') return route.continue();
        external.push(route.request().url()); return route.abort();
    });
    const started = performance.now();
    await page.goto(url);
    if (exploration) {
        const {checkExploration} = await import('./graph_exploration.browser.mjs');
        measurement.interactions = await checkExploration(page, screenshotDir, Number(process.env.OSINTGPT_TEST_ASSERTIONS || 805));
        assert.deepEqual(errors, []);
        assert.deepEqual(external, []);
        console.log('Screenshot: ' + join(screenshotDir, 'progressive-pages.png'));
    } else {
    const node = id => page.locator(`.node[data-entity-id="${id}"]`);
    const position = id => node(id).getAttribute('transform');
    const allPositions = () => page.locator('.node').evaluateAll(nodes => nodes.map(n => [n.dataset.entityId, n.getAttribute('transform')]));
    const wait = async predicate => {
        const until = Date.now() + 15000;
        while (!await predicate()) {
            if (Date.now() > until) throw new Error('Timed out waiting for interaction');
            await new Promise(done => setTimeout(done, 100));
        }
    };
    const saved = async () => JSON.parse(await page.locator('[data-testid="stCode"]').innerText());
    const choose = async (label, value) => {
        const input = page.getByRole('combobox', {name: label, exact: true});
        await input.click();
        await input.fill(value);
        await page.getByRole('option', {name: value, exact: true}).click();
    };
    await node('a').waitFor();
    assert.equal((await page.locator('svg.graph').boundingBox()).height, 560);
    const controls = await page.locator('.toolbar button').evaluateAll(buttons => buttons.map(button => ({
        label: button.getAttribute('aria-label'), title: button.title,
        text: button.textContent, icon: Boolean(button.querySelector('svg[aria-hidden="true"]')),
    })));
    assert.equal(controls.length, 6);
    assert.ok(controls.every(button => button.label === button.title && button.text === '' && button.icon));
    const graphBox = await page.locator('svg.graph').boundingBox();
    for (const selector of ['.toolbar', '.legend']) {
        const box = await page.locator(selector).boundingBox();
        assert.ok(box.x >= graphBox.x && box.y >= graphBox.y &&
                  box.x + box.width <= graphBox.x + graphBox.width && box.y + box.height <= graphBox.y + graphBox.height,
                  selector + ' should sit inside the graph.');
    }
    await page.getByRole('button', {name: 'Rearrange layout', exact: true}).focus();
    assert.equal(await page.locator('.toolbar button').first().evaluate(el => getComputedStyle(el, '::after').opacity), '1');
    assert.equal(await page.locator('.node.starting').count(), 1);
    assert.equal(await page.locator('.node.starting').getAttribute('data-entity-id'), 'a');
    assert.notEqual(await node('a').locator('circle').evaluate(el => getComputedStyle(el).fill),
                    await node('b').locator('circle').evaluate(el => getComputedStyle(el).fill));
    assert.equal(await page.locator('.claim').count(), 6);
    assert.equal(await page.locator('svg.graph img, svg.graph script').count(), 0);
    assert.equal(await node('c').locator('text').textContent(), '<img src=x onerror=alert(1)>');
    const initial = await allPositions();
    const edgeBefore = await page.locator('.edge').evaluateAll(nodes => nodes.map(n => n.getAttribute('d')));
    assert.equal(new Set(edgeBefore).size, 6);
    const box = await node('a').locator('circle').boundingBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down(); await page.mouse.move(box.x + 90, box.y + 65, {steps: 10}); await page.mouse.up();
    await wait(async () => (await saved())?.positions?.some(p => p.id === 'a' && p.x !== 735));
    const moved = await position('a');
    assert.notEqual(moved, initial[0][1]);
    assert.equal(await position('b'), initial[1][1]);
    await page.getByRole('button', {name: 'Unrelated rerun', exact: true}).click();
    await wait(async () => await position('a') === moved);
    await choose('Display filter', 'Indirect only');
    await wait(async () => await node('a').count() === 0);
    assert.equal(await page.locator('.node.starting').count(), 0);
    await page.getByText('Starting identity (outside displayed filters)', {exact: true}).waitFor();
    await choose('Display filter', 'None');
    await wait(async () => await page.locator('.diagram').count() === 0);
    await choose('Display filter', 'All');
    await node('a').waitFor(); assert.equal(await position('a'), moved);
    await page.locator('.claim[data-assertion-id="assert-1"]').click();
    await page.getByText('Evidence assertion: assert-1', {exact: true}).waitFor();
    assert.equal(await page.locator('.claim.selected').getAttribute('data-assertion-id'), 'assert-1');
    const beforeRearrange = await allPositions();
    await page.getByRole('button', {name: 'Rearrange layout', exact: true}).click();
    await wait(async () => JSON.stringify(await allPositions()) !== JSON.stringify(beforeRearrange));
    await wait(async () => (await saved()).positions.find(p => p.id === 'a').x !== Number(moved.match(/translate\(([^ ]+)/)[1]));
    await page.getByText('Evidence assertion: assert-1', {exact: true}).waitFor();
    assert.equal(await page.locator('.edge').count(), 6);
    assert.equal(await page.locator('.edge[data-assertion-id="assert-1"]').getAttribute('data-source-id'), 'b');
    assert.equal(await page.locator('.edge[data-assertion-id="assert-1"]').getAttribute('data-target-id'), 'a');
    await page.screenshot({path: join(screenshotDir, 'rearranged.png'), fullPage: true});
    await page.getByRole('button', {name: 'Reset layout', exact: true}).click();
    await wait(async () => JSON.stringify(await allPositions()) === JSON.stringify(initial));
    assert.equal(await page.locator('.claim.selected').getAttribute('data-assertion-id'), 'assert-1');
    const canvas = await page.locator('svg.graph').boundingBox();
    await page.mouse.move(canvas.x + 20, canvas.y + 20); await page.mouse.down();
    await page.mouse.move(canvas.x + 55, canvas.y + 45, {steps: 5}); await page.mouse.up();
    await wait(async () => (await saved()).view.x !== 0);
    assert.deepEqual(await allPositions(), initial);
    await page.mouse.move(canvas.x + 200, canvas.y + 100); await page.mouse.wheel(0, -180);
    await wait(async () => (await saved()).view.k > 1);
    await choose('Inspect assertion', 'assert-2');
    await wait(async () => await page.locator('.claim.selected').getAttribute('data-assertion-id') === 'assert-2');
    await choose('Result', 'Second');
    await wait(async () => await page.locator('.node.starting').getAttribute('data-entity-id') === 'b');
    assert.deepEqual(await allPositions(), initial);
    await node('b').focus(); await page.keyboard.press('ArrowRight');
    await wait(async () => await position('b') !== initial[1][1]);
    // Fullscreen must survive the component rerun after dragging or inspecting a row.
    const fullscreen = () => page.locator('.diagram').evaluate(el => el.matches(':fullscreen'));
    await page.getByRole('button', {name: 'Full screen', exact: true}).click();
    await wait(fullscreen);
    const fullscreenBefore = await position('b'), fullBox = await node('b').locator('circle').boundingBox();
    await page.mouse.move(fullBox.x + fullBox.width / 2, fullBox.y + fullBox.height / 2);
    await page.mouse.down(); await page.mouse.move(fullBox.x + 70, fullBox.y - 30, {steps: 10}); await page.mouse.up();
    await wait(async () => await position('b') !== fullscreenBefore);
    await wait(async () => (await saved()).positions.find(p => p.id === 'b').x !== 455);
    assert.equal(await fullscreen(), true);
    const positionsAfterDrag = await allPositions();
    await page.getByRole('button', {name: 'View data', exact: true}).click();
    assert.equal(await page.locator('.diagram-data tbody tr').count(), 6);
    assert.equal(await page.locator('.diagram-data img, .diagram-data script').count(), 0);
    await page.getByRole('button', {name: 'Inspect assert-4', exact: true}).click();
    await wait(async () => await page.getByText('Evidence assertion: assert-4', {exact: true}).count() === 1);
    await wait(async () => await page.locator('.diagram-data tr.selected').getAttribute('data-assertion-id') === 'assert-4');
    assert.equal(await fullscreen(), true);
    assert.equal(await page.getByRole('button', {name: 'View graph', exact: true}).count(), 1);
    await page.getByRole('button', {name: 'View graph', exact: true}).click();
    assert.deepEqual(await allPositions(), positionsAfterDrag);
    for (const format of ['SVG', 'PNG']) {
        const pending = page.waitForEvent('download');
        await page.getByRole('button', {name: 'Download ' + format, exact: true}).click();
        const download = await pending, filename = join(screenshotDir, download.suggestedFilename());
        await download.saveAs(filename);
        const content = await readFile(filename);
        if (format === 'SVG') {
            const details = await page.evaluate(text => {
                const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
                return {errors: doc.querySelectorAll('parsererror').length,
                    nodes: doc.querySelectorAll('.node').length, claims: doc.querySelectorAll('.claim').length,
                    starting: doc.querySelector('.node.starting')?.getAttribute('data-entity-id'),
                    fill: doc.querySelector('.node.starting circle')?.style.fill,
                    unsafe: doc.querySelectorAll('img, image, script').length,
                    text: doc.documentElement.textContent};
            }, content.toString('utf8'));
            assert.equal(details.errors, 0); assert.equal(details.nodes, 4); assert.equal(details.claims, 6);
            assert.equal(details.starting, 'b'); assert.equal(details.fill, 'rgb(168, 85, 247)');
            assert.equal(details.unsafe, 0);
            assert.ok(details.text.includes('Starting identity: b') && details.text.includes('Relation polarity:'));
        } else {
            assert.equal(content.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
            assert.ok(content.readUInt32BE(16) >= 1800 && content.readUInt32BE(20) > 1120);
        }
    }
    await page.screenshot({path: join(screenshotDir, 'fullscreen.png'), fullPage: true});
    await page.getByRole('button', {name: 'Exit full screen', exact: true}).click();
    await wait(async () => !await fullscreen());
    assert.equal((await page.locator('svg.graph').boundingBox()).height, 560);
    assert.deepEqual(await allPositions(), positionsAfterDrag);
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    console.log('Browser checks passed: force/reset, drag/pan/zoom, scope/filter persistence, same-name highlight, Unicode/literal HTML, selection, arrows, parallel/self assertions, keyboard movement, 560px canvas, fullscreen across reruns, data view, SVG/PNG downloads.');
    console.log('Screenshot: ' + join(screenshotDir, 'rearranged.png'));
    }
    measurement.scenario_ms = performance.now() - started;
    measurement.after_interactions = await memory();
    measurement.screenshots = screenshotDir;
    measurement.limitations = ['Synthetic graph; measurements are observations, not performance thresholds.',
        'Interaction timings include Playwright, Streamlit reruns, and browser rendering.',
        'Heap measurements are Chromium JavaScript heap snapshots, not peak memory or total browser/server memory.',
        'Continue-to-complete excludes earlier batches before pause.'];
    const report = process.env.OSINTGPT_TEST_REPORT || join(screenshotDir, 'measurements.json');
    await writeFile(report, JSON.stringify(measurement, null, 2) + '\n');
    console.log('Measurements: ' + report);
} catch (error) {
    if (page) {
        await page.screenshot({path: join(screenshotDir, 'failure.png'), fullPage: true});
        console.error('Failure screenshot: ' + join(screenshotDir, 'failure.png'));
        console.error(await page.locator('[role="option"]').allTextContents());
    }
    console.error(logs.slice(-6000));
    throw error;
} finally {
    if (browser) await browser.close();
    if (server.exitCode === null) {
        const closed = once(server, 'exit');
        server.kill();
        await closed;
    }
}
