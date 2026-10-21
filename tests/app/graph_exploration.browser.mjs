import assert from 'node:assert/strict';
import {join} from 'node:path';
import {readFile} from 'node:fs/promises';

export async function checkExploration(page, screenshots, count = 805) {
    assert.ok(count >= 805);
    const metrics = {assertions: count, stop_to_continue_ms: [], stop_click_to_ack_ms: []}, pages = Math.ceil(count / 30);
    let started;
    const wait = async predicate => {
        const until = Date.now() + 45000;
        while (!await predicate()) {
            if (Date.now() > until) throw new Error('Timed out waiting for progressive exploration');
            await new Promise(done => setTimeout(done, 100));
        }
    };
    const choose = async (label, text) => {
        const input = page.getByRole('combobox', {name: label, exact: true});
        await input.click(); await input.fill(text);
        await page.getByRole('option', {name: text, exact: true}).click();
    };
    const number = async (label, value) => {
        const input = page.getByRole('spinbutton', {name: label, exact: true});
        await input.fill(String(value)); await input.press('Enter');
    };
    const ids = () => page.locator('.claim').evaluateAll(nodes => nodes.map(node => node.dataset.assertionId));
    const pause = async () => {
        const stop = page.getByRole('button', {name: 'Stop exploration', exact: true});
        await stop.waitFor();
        await page.evaluate(() => {
            window.__osintgptStopClickAt = null;
            const clicked = event => {
                if (event.target.closest('button')?.textContent.trim() !== 'Stop exploration') return;
                window.__osintgptStopClickAt = performance.now();
                document.removeEventListener('click', clicked, true);
            };
            document.addEventListener('click', clicked, true);
        });
        // Exclude waiting for the initial query/button to appear from Stop latency.
        const started = performance.now();
        await stop.click();
        await page.getByRole('button', {name: 'Continue exploration', exact: true}).waitFor();
        metrics.stop_to_continue_ms.push(performance.now() - started);
        // Same browser clock; exclude Playwright's pre-click scrolling/stability waits.
        const acknowledged = await page.evaluate(() => window.__osintgptStopClickAt === null ? null :
            performance.now() - window.__osintgptStopClickAt);
        assert.ok(acknowledged !== null && acknowledged >= 0);
        metrics.stop_click_to_ack_ms.push(acknowledged);
    };
    await page.getByRole('textbox', {name: 'Starting entity', exact: true}).fill('Alpha');
    await page.getByRole('textbox', {name: 'Starting entity', exact: true}).press('Enter');
    await page.getByRole('button', {name: 'Find identities', exact: true}).click();
    await choose('Select identity', 'Alpha (unspecified type) · alpha');
    assert.equal(await page.getByRole('spinbutton', {name: 'Traversal depth', exact: true}).inputValue(), '1');
    await page.getByRole('button', {name: 'Explore neighborhood', exact: true}).click();
    await pause();
    const paused = await page.getByText(/^Neighborhood: .*total unknown/).innerText();
    await new Promise(done => setTimeout(done, 1200));
    assert.equal(await page.getByText(/^Neighborhood: .*total unknown/).innerText(), paused);
    started = performance.now();
    await page.getByRole('button', {name: 'Continue exploration', exact: true}).click();
    await wait(async () => {
        const label = page.getByText(/^Neighborhood: /);
        return await label.count() > 0 && await label.innerText() !== paused;
    });
    metrics.continue_to_progress_ms = performance.now() - started;
    await page.getByText(`Neighborhood: ${count} assertions found; ${count} returned.`, {exact: true}).waitFor({timeout: 120000});
    await wait(async () => (await ids()).length === 30);
    metrics.continue_to_complete_ms = performance.now() - started;
    const first = await ids();
    assert.equal(await page.getByRole('button', {name: 'Expand one more step', exact: true}).count(), 0);
    const pager = await Promise.all([
        page.getByRole('button', {name: 'Previous page', exact: true}).boundingBox(),
        page.getByRole('spinbutton', {name: 'Table page', exact: true}).boundingBox(),
        page.getByRole('button', {name: 'Next page', exact: true}).boundingBox(),
    ]);
    assert.ok(Math.max(...pager.map(box => box.y + box.height)) - Math.min(...pager.map(box => box.y + box.height)) < 3,
              'Page buttons and page input should share a bottom alignment.');
    await page.getByRole('spinbutton', {name: 'Table page', exact: true}).scrollIntoViewIfNeeded();
    await page.screenshot({path: join(screenshots, 'pagination.png'), fullPage: true});
    started = performance.now();
    await page.getByRole('button', {name: 'Next page', exact: true}).click();
    await page.getByText(new RegExp(`^Table page 2 of ${pages}:`)).waitFor();
    await wait(async () => (await ids())[0] !== first[0]);
    const second = await ids();
    assert.equal(second.length, 30);
    assert.ok(second.every(id => !first.includes(id)));
    metrics.next_page_with_diagram_ms = performance.now() - started;
    const exports = page.locator('[data-testid="stExpander"]').filter({hasText: 'Export query results'});
    await exports.locator('summary').click();
    const download = async (scope, count) => {
        await exports.getByText(scope, {exact: true}).click();
        assert.equal(await exports.getByRole('radio', {name: scope, exact: true}).isChecked(), true);
        const started = performance.now();
        await exports.getByRole('button', {name: 'Prepare graph downloads', exact: true}).click();
        await exports.getByText(new RegExp(`^Prepared ${count} assertions;`)).waitFor();
        const event = page.waitForEvent('download');
        await exports.getByRole('button', {name: 'Download graph JSON', exact: true}).click();
        const file = await event;
        const packageData = JSON.parse(await readFile(await file.path(), 'utf8'));
        assert.equal(packageData.edges.length, count);
        metrics[scope === 'Current table page' ? 'page_export_download_ms' : 'all_export_download_ms'] =
            performance.now() - started;
        return packageData;
    };
    const pageExport = await download('Current table page', 30);
    assert.deepEqual(pageExport.edges.map(row => row.assertion_id), second);
    assert.equal(pageExport.metadata.selection.complete_matching_scope, false);
    const allExport = await download('All matching results', count);
    assert.equal(allExport.metadata.selection.complete_matching_scope, true);
    assert.equal(allExport.metadata.query.verification_state.checked_assertions, count);
    await exports.screenshot({path: join(screenshots, 'query-exports.png')});
    await exports.locator('summary').click();
    await choose('Diagram selection', 'First matching assertions');
    await wait(async () => (await ids())[0] === first[0]);
    assert.deepEqual(await ids(), first);
    await choose('Diagram selection', 'Current table page');
    await wait(async () => (await ids())[0] === second[0]);
    // Dense subsets can overlap before manual rearrangement; keyboard selection
    // exercises the same evidence callback without depending on one geometry.
    await page.locator('.claim').nth(1).focus();
    await page.keyboard.press('Enter');
    await wait(async () => await page.locator('.claim.selected').getAttribute('data-assertion-id') === second[1]);
    await page.getByText(second[1] + ' · notes.md', {exact: true}).waitFor();
    await choose('Diagram selection', 'Hidden');
    await wait(async () => await page.locator('.diagram').count() === 0);
    await choose('Diagram selection', 'First matching assertions');
    await number('Assertions in diagram', 75);
    await wait(async () => (await ids()).length === 75);
    started = performance.now();
    await number('Table page', pages);
    await page.getByText(new RegExp(`^Table page ${pages} of ${pages}: assertions ${(pages - 1) * 30 + 1}–${count}`)).waitFor();
    assert.equal((await ids()).length, 75);
    metrics.last_page_independent_diagram_ms = performance.now() - started;
    started = performance.now();
    await page.getByRole('button', {name: 'Rearrange layout', exact: true}).click();
    await page.locator('.layout-status').filter({hasText: 'Layout rearranged.'}).waitFor();
    metrics.rearrange_75_assertions_ms = performance.now() - started;
    await page.screenshot({path: join(screenshots, 'progressive-pages.png'), fullPage: true});
    await number('Traversal depth', 2);
    await page.getByRole('button', {name: 'Explore neighborhood', exact: true}).click();
    await pause();
    assert.equal(await page.getByText(new RegExp(`^Neighborhood: ${count} assertions found so far; total unknown`)).count(), 1);
    console.log(`Progressive browser checks passed: automatic batches, stop/continue, depth-1 default, complete ${count}-assertion discovery, aligned pagination, page/all-matching JSON downloads, diagram follows page by default, independent selections, 75 selected assertions, hidden diagram, evidence selection, and depth-input expansion.`);
    return metrics;
}
