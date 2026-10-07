# The sourced graph

Graph construction makes one extraction call per text window and at most one
additional quotation-correction call for that window. Each structured
record is processed separately; short prose files fit in one window and long
documents require several calls. It never runs as an
indexing or search side effect. Enable it, then build explicitly:

```bash
osintgpt config set graph_enabled true
osintgpt graph build
```

The build command warns before provider calls, prints `position/total ref`
progress, and reports documents, records loaded, windows, generation calls,
failed requests, invalid responses, and recorded token/cost usage separately.
Call counts cover the generation-provider interface; internal SDK retries are
not measured. Missing usage or prices are reported as incomplete, never as zero.
Rerun `graph build` to resume compatible saved windows and refresh new or changed
sources. `--incremental` performs the same refresh but requires an initial build;
`--rebuild` discards saved work and extracts registered sources again.
The two flags cannot be combined.

Requests have bounded input, entity context, and output. Defaults reserve 4,096
output tokens and 4,096 UTF-8 bytes for up to 32 relevant entity candidates within
a 32,768-unit context budget. Input uses a conservative UTF-8 byte estimate, not
an English characters-per-token ratio or an exact model tokenizer. Set
`--context-budget` below your model's context capacity and `--output-tokens`
within its output limit. `--max-record-calls` (default 500) bounds all extraction
and correction calls per record across resumptions, including failed calls.
Raise this limit to continue after exhausting it. `--no-repair-quotes` disables corrections.
Native generation providers receive the output cap; custom providers lacking
that parameter are reported as not enforcing it. A 1,000-page document is still
processed in overlapping paragraph/sentence windows, never sent whole.

```bash
osintgpt graph build --incremental --json
osintgpt graph build --rebuild --project CASE_SLUG
osintgpt graph status --project CASE_SLUG --json
```

`graph status` reads saved state without opening a provider. Settings also shows
the state and last-run counters. States distinguish never built, building,
complete, complete with gaps, interrupted, failed, and stale. An extraction
that finds no relationships is still a completed build, and its documents are
reused by a subsequent build with unchanged extraction settings. Source changes
are flagged as stale. Compatible saved windows survive interruption; changed
source text, mappings, models, prompts, or extraction settings require fresh work.
Status separates reused windows from new calls and retains recent run accounting.
Unregistering a source removes its contribution on the next build. Missing files
that are still registered retain their claims and produce a warning. Failed updates
preserve the old document's claims until a replacement finishes processing.
Rejected candidates leave the graph complete with gaps. An incremental skip
preserves that status; use `--rebuild` to try extracting those documents again.

Turning off `graph_enabled` stops graph extraction and graph retrieval by the
answering model while preserving stored data. Explicit CLI inspection and export
remain available. New graphs use schema version 4; incompatible databases
are refused without modifying them. There is no migration path. If an earlier
test graph exists, move its `graph.sqlite` aside before building a fresh one.

Indexing, graph construction, verification, and source reads share versioned
extracted text in `extracts/text/`. A graph build reuses existing PDF page
transcriptions without making vision calls. If pages have no cached text, the
build reports the missing page numbers and completes with gaps. Index the PDF
to transcribe those pages, then rerun `graph build`. Fully unreadable pages are
not sent to the graph model.

The indexing recipe changed to attach this provenance to chunks. The next
indexing run re-embeds previously indexed files under the new recipe; graph
construction remains a separate explicit operation.

Each claim records its source text version, file and record reference, and
matching quote locations: PDF pages, available headings, paragraphs, and
character offsets in the extracted record text (zero-based, end exclusive).
Repeated quotations retain all matching locations within supplied source ranges.
Each relationship needs one to eight separately located quotations. Matching
normalizes Unicode NFC and whitespace only: no stop-word removal, case folding,
punctuation stripping, translation, or fuzzy acceptance. Meaningful words remain
intact in every language. Only the current record window and explicitly supplied
earlier excerpts from the same record can support a claim.

An unlocated quote may receive one correction attempt, for up to 12 candidates
within the request budget. That request can change only the quotations. Invalid
entities and unresolved relationships are excluded from the accepted graph;
`graph rejected` retains the original proposal, reason, source version, window,
timestamp, and correction outcome. Audit records remain after rebuilds, including
successful corrections, so they are history rather than the current gap count.

```bash
osintgpt graph rejected --project CASE_SLUG --json
osintgpt graph rejected --project CASE_SLUG --ref records.csv --json
```

A match proves textual presence. It does not prove the relationship true or that
the model interpreted the quotation correctly. The model recognizes negation,
allegations, uncertainty, and hypothetical statements in the source language;
code validates the resulting fields and quoted support without language-specific
keyword lists. Missing attribution, dates, or qualifications remain empty/unknown
and do not discard an otherwise supported relationship. Event time and statement
time are separate, quoted values; ambiguous dates are not parsed or guessed.

CSV, Excel, JSON, and JSONL records keep separate references such as
`records.csv#case-4821`. Use a unique identity field for references that survive
row reordering. Without one, the loader uses a row position; duplicate
identities are rejected. Entity candidates can cross record boundaries, but
earlier source excerpts are carried only within the same record and text version.

Entities have persistent app-assigned IDs, supported aliases, and distinguishing
context. A later window receives a bounded set of relevant candidates plus recent
same-record entities, with source references. Names alone never merge nodes.
Cross-document automatic reuse requires a matching, source-supported identifier
and issuer with the same nonempty entity type. Uncertain matches remain separate;
this can leave duplicate real-world entities for later review.

Traversal is keyless once a graph exists. `neighbors` shows sourced claims
touching an entity; a depth above one walks farther out. `path` finds the
shortest chain within the requested depth. Every row includes its source
document, qualifications, and exact evidence text. Names and aliases can return
multiple distinct entities. Use `graph entities QUERY --json` to see candidate IDs,
aliases, distinguishing context, and source references. Both neighborhood and path
queries require an unambiguous identity and return candidates when several match.

```bash
osintgpt graph neighbors Neral-7 --depth 2 --limit 20
osintgpt graph path Neral-7 "Station Kestrel" --max-depth 4 --direction outgoing
```

Use JSON output to inspect the evidence provenance along with each claim:

```bash
osintgpt graph neighbors Neral-7 --project walkthrough --json
osintgpt graph path Neral-7 "Station Kestrel" \
  --project walkthrough --json
```

Verification compares evidence with both its saved text snapshot and the
current source, without new transcription calls. `--json` exposes each claim's
`status` against current text, `snapshot_status`, `source_state`, and current
quote locations. A deleted or changed source can fail the current check while
the saved snapshot still verifies. `--strict` exits non-zero for a missing or
unreadable quotation in the current source. Export preserves provenance as a
JSON object in `.json` and a JSON-encoded string property in `.cypherl`:

```bash
osintgpt graph verify --strict
osintgpt graph verify --json
osintgpt graph export graph.cypherl
osintgpt graph export graph.json
```

CYPHERL is ready for line-oriented import into Memgraph or Neo4j. An exported
relationship is a sourced assertion from the corpus, not proof that the
assertion is true.

The answering model receives assertion IDs, polarity, assertion status, attribution,
event/statement dates, every supporting excerpt, provenance, and saved/current
verification results. `incoming`, `outgoing`, and default `connectivity` select
how traversal follows stored arrows. A reverse traversal keeps the original
assertion's meaning. A path may include a negative or alleged claim and must not
be presented as proof of an affirmative real-world connection.

Use `graph evidence ASSERTION_ID --json` with an ID from a query to read saved
source text around its supporting quotation. `--quote-index` selects a quotation,
`--occurrence` selects its location, and `--offset` follows the returned
`next_offset` when the passage needs several pages. All are zero-based. Model
tools use the same service through `graph_evidence`. `quote_complete` distinguishes
a complete quotation from one split across response pages.

Queries disclose stale/incomplete build state and depth, result, work, or response
limits. Depth cannot exceed 4; candidates/assertions cannot exceed 60. The response
evidence budget defaults to 24,000 bytes (`--max-bytes`, maximum 65,536). When a
complete path cannot fit, its assertion IDs remain available for individual
evidence reads. No path found is not evidence of no relationship. Semantic and
lexical retrieval remain available, and graph proximity does not change passage
ranking. Existing app passage previews display saved text
versions and available page/section labels. The dedicated Graph screen and
interactive evidence inspector remain planned work.

The extraction instructions live in
`osintgpt/prompts/templates/graph_extraction.md.j2`; the bounded correction prompt
is `osintgpt/prompts/templates/graph_quote_repair.md.j2`. Source text and candidate
context are JSON in the user message, separate from system instructions. Offline
tests cover multilingual quotation gates, IDs, budgets, and field preservation;
live model quality still needs evaluation on operator-reviewed multilingual cases.
