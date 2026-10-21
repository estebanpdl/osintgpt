# The sourced graph

Graph construction starts with an extraction request per text window. Interactive
identity search continues while useful work remains; quotation and structured-response
corrections can add requests. Local searches are counted
separately from paid model requests. Each structured
record is processed separately; short prose files fit in one window and long
documents require several calls. It never runs as an
indexing or search side effect. Enable it, then build explicitly:

```bash
osintgpt config set graph_enabled true
osintgpt graph build
```

By default, extraction and quotation correction inherit the Chat/RAG model.
To save an independent graph choice for both CLI and app builds:

```bash
osintgpt config set graph_generation_provider openai
osintgpt config set graph_generation_model EXTRACTION_MODEL_ID
osintgpt config set graph_model_mode independent
osintgpt graph build --dry-run
```

Replace `EXTRACTION_MODEL_ID` with a model supported by the chosen provider.
`config set graph_model_mode inherit` explicitly returns to Chat/RAG inheritance,
including when user defaults specify a separate graph model. Saving makes no
requests. With an independent graph choice, changing chat models preserves graph
work; changing the actual graph model requires fresh extraction on the next build.

For one invocation, use `graph build --provider PROVIDER --model MODEL`, optionally
with `--dry-run`. These flags never change saved settings. `--model` alone keeps
the selected graph provider; a provider-only override needs a registry default
(built-in generation providers currently have none). Use the same overrides to
resume compatible work. Previews show the resolved selection, affected published
sources, and incompatible saved windows/responses before requests are made.

The build command warns before provider calls, prints `position/total ref`
progress, and reports documents, records loaded, windows, generation calls,
failed requests, invalid responses, and recorded token/cost usage separately.
Built-in providers report HTTP attempts, including controlled retries. Custom
providers without request hooks retain provider-interface call counts and report
unenforced controls. Missing usage or prices are reported as incomplete, never as zero.
Rerun `graph build` to resume compatible saved windows and refresh new or changed
sources. `--incremental` performs the same refresh but requires an initial build;
`--rebuild` discards saved work and extracts registered sources again.
The two flags cannot be combined.

Source windows default to 32,768 UTF-8 bytes of serialized source text, subject
to the existing 24,000-character cap and overlap. Change this with
`--source-window-bytes` (`--context-budget` remains an alias). It is a passage
size, not a limit on the whole request or conversation. Identity lookup,
evidence paging, and conversation restarts are automatic. Useful work has no
fixed turn/call ceiling; total requests and final cost are unknown. Cancellation
and an explicit spending ceiling still stop a build and preserve compatible work.

Output is provider-managed: OpenAI/Gemini requests omit an application output
cap; Anthropic uses its resolved model maximum and streaming. Provider limits
still apply. `--output-tokens` and `--max-record-calls` are retired with guidance;
saved app `output_tokens`, `context_units`, and `max_calls` are ignored and removed
on save. Library `context_units`/`max_calls` arguments are deprecated and ignored.
`--no-repair-quotes` disables quotation corrections. A 1,000-page document is
processed in overlapping paragraph/sentence windows, never sent whole.

```bash
osintgpt graph build --incremental --json
osintgpt graph build --rebuild --project CASE_SLUG
osintgpt graph status --project CASE_SLUG --json
```

`graph diagnostics --project CASE_SLUG --json` inspects retained generation
responses and parser/protocol errors without opening a provider. Add `--raw` to
include original text and provider continuation, which may contain source material;
use `--offset`/`--limit` for pagination. Invalid responses remain inspectable after
correction or publication. Repeated invalid completion does not repeat a paid
request on every resume. Older releases' deleted responses cannot be reconstructed.

`graph status` reads saved state without opening a provider. Settings shows a
brief status summary; Graph contains build counters, source refresh details,
and published extraction models. States distinguish never built, building,
complete, complete with gaps, interrupted, failed, and stale. An extraction
that finds no relationships is still a completed build, and its documents are
reused by a subsequent build with unchanged extraction settings. Source changes
are flagged as stale. Compatible saved windows survive interruption; changed
source text, mappings, models, prompts, or extraction settings require fresh work.
Status separates reused windows from new calls and retains recent run accounting.
`graph status --json` reports the latest run's selection separately from
`extraction_models`, which lists published models by source/version. Partial
refreshes can leave multiple models represented; legacy unknown identities remain
unknown. Graph export metadata carries these distinctions too.
Unregistering a source removes its contribution on the next build. Missing files
that are still registered retain their claims and produce a warning. Failed updates
preserve the old document's claims until a replacement finishes processing.
Rejected candidates leave the graph complete with gaps. An incremental skip
preserves that status; use `--rebuild` to try extracting those documents again.

Turning off `graph_enabled` stops graph extraction and graph retrieval by the
answering model while preserving stored data. Explicit CLI inspection and export
remain available. New graphs use schema version 7. Preview a supported upgrade
with `osintgpt graph migrate --dry-run`, then run `osintgpt graph migrate` to
back up and upgrade the active `graph.sqlite` without model calls. Add
`--project CASE_SLUG` to choose the project. The original version-0 and known
version-5 and version-6 layouts are supported; unknown layouts are refused unchanged. Empty
graphs become never built. Populated version-0 graphs preserve their records but
remain marked as legacy until their sources are re-extracted with `graph build`.
Migration does not reconstruct missing qualifications or historical source text.

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
first normalizes Unicode NFC and whitespace. Recognized Markdown presentation
can also be projected to visible words while retaining original source locations
and excerpts. Plain text and code stay literal. There is no stop-word removal,
case folding, general punctuation stripping, translation, or fuzzy acceptance. Meaningful words remain
intact in every language. Only the current record window and explicitly supplied
earlier excerpts from the same record can support a claim.

Unlocated entity and relationship quotations may each receive one correction
attempt for each eligible candidate. Requests are batched when actual provider
capacity requires it; there is no twelve-candidate cutoff. They can change only quotations. Unsupported essential evidence stays rejected; an invalid
optional entity field produces a warning without discarding the supported core.
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
Cross-document reuse can use a matching supported identifier and issuer, or an
explicit model decision citing compatible, distinguishing evidence from candidate
lookup and the current source. The model can search beyond the initial context
through read-only tools on supported adapters. Decisions retain mentions and
source locations; uncertain matches remain separate with supported claims.
Inspect with `graph reconcile --query NAME --json`. Combine reviewed IDs with
`--from-id`, `--to-id`, and `--reason`; preview first, then add `--apply` to back up
and save. `--undo DECISION_ID --apply` reverses a compatible operator merge;
`--split-source SOURCE_REF` separates a file's contribution from `--from-id`.
These actions make no model calls. `graph revalidate` previews local corrections
to saved rejected proposals; `--apply` publishes only supported corrections.
Old audits missing their original ranges or endpoint IDs remain rejected with
an explanation. Migration does not itself merge identities or re-extract sources.

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

CLI exports now include project identity, source versions, build timestamps,
stale/incomplete status, and saved/current quotation verification. JSON keeps
`entities` and `edges` and adds `metadata`; CYPHERL carries an `OsintGPTExport`
metadata node and JSON-string properties for structured evidence. Entity keys
are scoped by project ID when importing. Re-import updates verification without
duplicating an assertion, but does not delete assertions from earlier imports.

The answering model receives assertion IDs, `relation_polarity`, assertion status, attribution,
event/statement dates, every supporting excerpt, provenance, and saved/current
verification results. `incoming`, `outgoing`, and default `connectivity` select
how traversal follows stored arrows. A reverse traversal keeps the original
assertion's meaning. A path may include a negative or alleged claim and must not
be presented as proof of an affirmative real-world connection.

`relation_polarity` keeps the values `positive`, `negative`, and `unknown`; the
human-readable label is **Relation polarity**. Older stored `polarity` fields are
read compatibly, and historical raw diagnostics keep the original spelling.
This field rename does not require a graph rebuild or a new provider call.
Human-readable exploration separates direct assertions, distinct neighbors, and
returned neighborhood assertions. Reaching the selected depth is scope information;
actual work and response limits still warn. CLI JSON retains the bounded retrieval
metadata, including `depth_boundary`. Display caps have not been removed.

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
ranking. In the app, open **Graph** to build, resume, cancel, search identities,
filter neighborhoods, and inspect connections. Selecting a relationship opens
its quotations, qualifications, saved source version, and current/snapshot checks.
Read surrounding saved text or compare other assertions for the same entity pair.
Build settings are saved per project for app runs; CLI flags remain independent.
New answers also retain graph evidence previews when conversations are reopened.
See the [app graph workflow](../../README.md#work-with-the-graph-in-the-app) for
recovery behavior and query limits. **Export full graph** and **Export displayed
results** prepare JSON/CYPHERL downloads, including filters and query limits.
**Ask → Export answers and findings** prepares Markdown/CSV for selected saved
answers and authored canon pages. These exports preserve stored content without
model calls; query exports retain the evidence shown, while full-graph exports
check current sources during preparation.

The extraction instructions live in
`osintgpt/prompts/templates/graph_extraction.md.j2`; the bounded correction prompt
is `osintgpt/prompts/templates/graph_quote_repair.md.j2`. Source text and candidate
context are JSON in the user message, separate from system instructions. Offline
tests cover multilingual quotation gates, IDs, budgets, and field preservation;
live model quality still needs evaluation on operator-reviewed multilingual cases.
For local model-role checks and prepared controlled comparisons, see the
[benchmark guide](../../tests/benchmarks/README.md). Its paired runner defaults to
offline fixtures and can reuse saved graph evidence without new extraction or
embedding calls. These development scripts are separate from `osintgpt graph build`.
