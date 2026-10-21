# Offline graph layout checks

The diagram uses bundled JavaScript and CSS with Streamlit's native component API.
These checks use synthetic identities and assertions; no research project or model
provider is opened.

Run the Python checks from the repository root in the OsintGPT environment:

```powershell
python -m pytest -q tests/app/test_graph_layout.py tests/app/test_graph_view.py tests/graph/test_graph_inspection.py
python -m pytest -q tests/app/test_graph_progressive.py tests/graph/test_progressive_exploration.py
```

Run the pure geometry checks with Node.js:

```powershell
node --test tests/app/graph_layout.test.mjs
```

For actual browser interactions, use an existing Playwright and Chromium installation:

```powershell
node tests/app/graph_layout.browser.mjs
```

If they are not on the default paths, set `OSINTGPT_PLAYWRIGHT_MODULE` to the absolute
path of Playwright's `index.mjs`, `OSINTGPT_TEST_CHROMIUM` to the Chromium executable,
and `OSINTGPT_TEST_PYTHON` to the Python executable in the OsintGPT environment.
Set `OSINTGPT_TEST_THEME=dark` to repeat with the browser's dark color preference.
Set `OSINTGPT_TEST_SCENARIO=exploration` to exercise the real exploration controls
against a temporary synthetic 805-assertion graph. This scenario checks automatic
batch continuation, stop/resume, the depth-1 default, aligned page controls, a
diagram that follows table pages by default, independent selections (including
more than 60 assertions), evidence selection, actual JSON downloads for the
current page and all 805 matching assertions, and expansion through the depth
input and query button. It uses the same loopback browser harness; no operator
project is read or modified. Unset the variable to run the layout-only scenario.
Set `OSINTGPT_TEST_ASSERTIONS` to a larger count (at least 805) to repeat the
exploration scenario at another scale. `OSINTGPT_TEST_REPORT` optionally sets
the JSON measurement output path; otherwise it is saved beside the screenshots.
Reports include discovery, Stop-to-Continue-button and Continue-to-progress timings,
page/diagram and export interaction timings, Chromium
JavaScript heap snapshots, browser version, and theme. The exploration scenario
also times completion of rearranging a 75-assertion diagram. These are observations,
not performance thresholds. Heap snapshots are neither peak memory nor total
browser/server memory, and resume timing excludes work completed before pause.
Stop timing begins after the Stop button appears and includes Playwright's click
and the UI acknowledgement; it does not measure interruption inside a work batch.
The separate click-to-acknowledgement measurement starts at the browser click event,
excluding pre-click scrolling/stability waits, and ends after Continue appears.
The script starts its own temporary loopback Streamlit server, blocks and reports
external HTTP requests, disables source-file hot reload for a stable fixture,
and closes that server and browser when done. Screenshots
are written to an `osintgpt-layout-browser-*` folder in the system temporary directory.

Browser checks cover rearrange/reset, node drag versus canvas pan, zoom, persistence
through selection/filter reruns, new-result reset, same-name identities, Unicode
labels, literal HTML handling, opposite/parallel/self assertions, and keyboard movement.
They also check the 560-pixel canvas, fullscreen retention across drag/selection
reruns, the diagram's data-table toggle, and actual PNG/SVG downloads with retained
IDs, colors, counts, and legend. Downloaded images are saved beside the screenshots.
The icon toolbar is checked for accessible names and focus tooltips; both toolbar
and legend must lie inside the graph canvas.
The geometry tests also exercise a 60-assertion, 120-node diagram. These are interaction
and data-preservation checks, not extraction-quality benchmarks. Migration validation
uses the offline Python fixtures separately. The
[Phase 15 closeout report](../../plan/phase15-validation.md) maps the combined checks
to the phase requirements and records observed performance and memory.

`test_graph_filter_exports.py` checks all-matching/page/diagram scopes, full-text
filters, empty selections, and hiding prepared downloads after a source edit.
`tests/graph/test_operator_filter_snapshot.py` checks cache reuse, evidence-status
filters, full quotations and provenance, incomplete/path exports, source and
snapshot invalidation, and rejection of changes during verification/export.
