<div align="center">

# OSINT GPT

<img src="https://raw.githubusercontent.com/estebanpdl/osintgpt/main/images/osintgpt.png" alt="osintgpt osint gpt" width="33%" height="33%" />

[![GitHub forks](https://img.shields.io/github/forks/estebanpdl/osintgpt.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/estebanpdl/osintgpt/network/)
[![GitHub stars](https://img.shields.io/github/stars/estebanpdl/osintgpt?style=social)](https://github.com/estebanpdl/osintgpt/stargazers)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/estebanpdl/osintgpt/blob/main/LICENSE)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://twitter.com/estebanpdl)
[![Made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Twitter estebanpdl](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/estebanpdl)

</div>

`osintgpt` indexes documents an analyst already holds and answers questions
from that material with citations. It is built for examining research
collections without mixing projects or losing the path back to a source. Use
it from the command line or as a Python library, with hosted or local models
and SQLite, Qdrant, or Postgres storage.
Embeddings support OpenAI, Gemini, Voyage, Ollama, and sentence-transformers;
generation supports OpenAI, Gemini, Anthropic, and Ollama.

## Install

The core install includes the document readers and command-line interface:

```bash
pip install osintgpt
```

Check the installed version with `osintgpt --version` (or `osintgpt -V`).

Local embeddings are the one opt-in extra because they bring
sentence-transformers and torch:

```bash
pip install osintgpt[local]
```

See [osintgpt on PyPI](https://pypi.org/project/osintgpt/) for package metadata
and release history.

## Quickstart

By default, embeddings and answers use OpenAI. Set `OPENAI_API_KEY` and
`OPENAI_GPT_MODEL` in your environment first, or configure the
[fully local path](#keep-data-local).

```bash
osintgpt project create "Research notes"
osintgpt project use research-notes
osintgpt add ./material
osintgpt index
osintgpt ask "Who did Alpha Corp fund?"
```

A real `ask` ends like this:

```text
Alpha Corp funded Beta Ltd in March. [1]

Sources
• material/alpha.md

Ask next
1. Who else did Alpha Corp fund?
2. What happened after March?
```

A project with neither indexed passages nor an enabled, built graph is also a
valid state: `ask` explains that nothing was retrieved and does not call the
generation model.

<a id="how-it-searches"></a>

## Tri-retrieval RAG: semantic, lexical, and relational

Retrieval-augmented generation (RAG) finds relevant material and gives it to the
answering model as evidence. `osintgpt` provides three complementary ways to find
that material inside a project:

| Retrieval method | What it finds | When to use it | Preparation and model calls |
|---|---|---|---|
| **Semantic** | Indexed passages with meaning close to the query, even with different wording | Topics, concepts, and paraphrased claims | Run `index` first. Each search embeds the query with the model used for the index. |
| **Lexical** | Indexed passages containing supplied literal terms | Names, handles, URLs, hashes, and case or account identifiers | Uses the indexed text. Explicit terms require no embedding or generation call. |
| **Relational / graph** | Extracted entities and sourced relationship claims, including chains between entities | Who is connected to whom, through which claims and documents | Enable and explicitly build the graph using generation calls. Inspecting the saved graph requires no model call. |

Semantic and lexical retrieval return stored text passages. Graph retrieval returns
structured claims with entity IDs, source references, supporting quotations, and
qualifications such as negation or allegation status. The answering model can
request more source text when those results need context. Retrieval scores indicate
relevance, not whether a claim is true.

### Search passages by meaning or literal terms

After the [quickstart](#quickstart) has registered and indexed your material:

```bash
osintgpt search "funding for field operations" --top-k 10
osintgpt search --exact "Neral-7" --exact "Station Kestrel" --full
osintgpt search "funding for field operations" --exact "Neral-7" --semantic --json
osintgpt search "Who financed Station Kestrel?" --derive-terms
```

- A query without `--exact` runs semantic search. `--top-k` limits the returned
  passages; `--full` prints their complete stored text rather than a preview.
- Repeating `--exact` searches for any of the supplied terms and ranks passages
  by how many distinct terms they contain. Matching uses substrings in indexed
  text or author metadata, with case handling provided by the storage backend.
  Terms shorter than two characters are ignored. With explicit terms, add
  `--semantic` and query text to run semantic search too.
- `--derive-terms` makes an additional generation call to choose literal terms
  from the question, then runs semantic and lexical retrieval. Term selection is
  model-driven, without an English stop-word list. If no usable terms are returned,
  the search proceeds with semantic results.

When both passage methods run, **reciprocal rank fusion** combines their ranked
lists, rewarding passages found by both. `--json` reports each result's contributing
`legs` and `ranks`. This search mode combines semantic and lexical results; graph
claims are retrieved through the graph commands or the answering model's graph tools.
Multilingual semantic matching and term selection depend on the selected models.

### How the answering model uses the three methods

```bash
osintgpt ask "What links Neral-7 to Station Kestrel?" --trace
osintgpt ask "What links Neral-7 to Station Kestrel?" --json
osintgpt ask "What do the documents say about field funding?" --static --passages 8
```

Default `ask` lets the model survey documents, choose semantic, lexical, or graph
tools, follow promising passages with `snowball`, and fetch source text before
answering with citations. It chooses tools as needed; a question does not
automatically run all three retrieval methods. `--trace` displays that activity,
and JSON answers include the trace. Graph retrieval is offered only when
`graph_enabled` is true and needs an explicitly built graph to return claims.

`--static` currently retrieves semantic passages once and answers from those
passages, without lexical or graph retrieval. `--passages` controls this static
path. The same semantic pipeline is the fallback when model tool calling is
unavailable, with the reason recorded in the trace. Tool-based answering also
works with an enabled, built graph when no passages have been indexed; the static
pipeline still needs indexed passages.

For relationship questions, the model can discover identities with `graph_entities`,
select an ID, use `graph_query`, and read a quotation's source context with
`graph_evidence`. Matching names or aliases can represent different identities:
the model receives candidates and should request clarification when their context
does not settle the choice. Every returned claim carries its assertion ID,
supporting quotations, source version and locations, qualifications, and separate
saved/current evidence checks. Answers are instructed to cite each supporting
link and preserve negation, allegations, uncertainty, and dates.

Graph results distinguish build activity from stale or incomplete coverage and
disclose query limits. A missing path
does not prove that no relationship exists. Semantic and lexical tools remain
available when the graph cannot help. Graph-discovered passages can supply extra
context; they do not automatically receive a higher passage-search rank.

### Build, inspect, and maintain the relationship graph

Graph construction is a separate operation from indexing and asking questions.
For the selected project, enable it and start an explicit build:

```bash
osintgpt config set graph_enabled true
osintgpt graph build
osintgpt graph status --json
```

Graph extraction inherits the Chat/RAG provider and model unless you select an
independent pair. To use a separate extraction model for the selected project,
replace `EXTRACTION_MODEL_ID` with a model supported by your chosen provider:

```bash
osintgpt config set graph_generation_provider openai
osintgpt config set graph_generation_model EXTRACTION_MODEL_ID
osintgpt config set graph_model_mode independent
osintgpt graph build --dry-run --json
```

These shared settings apply to CLI and app builds, including quotation correction.
They use the existing provider credentials; chat answering keeps its own model.
The preview's `provider` object reports the resolved graph provider and model
without making model calls; `graph_model` also identifies inheritance, independent
selection, or an invocation override. A missing or invalid independent selection
is reported instead of falling back to the chat model.

To override the graph model for one build or preview without saving any settings:

```bash
osintgpt graph build --provider openai --model EXTRACTION_MODEL_ID --dry-run
osintgpt graph build --provider openai --model EXTRACTION_MODEL_ID
```

`--model` alone keeps the selected graph provider. `--provider` alone requires a
registered provider default; built-in graph providers currently require `--model`.
Repeat the same overrides when resuming that build. Previews identify affected
published sources, changes to source text or extraction settings, and saved windows
or responses that cannot be reused. Unknown pricing is shown as unknown.

With an independent graph model, changing the Chat/RAG model preserves compatible
graph work. Changing the actual graph provider/model schedules fresh extraction
on the next explicit build, including `--incremental`; saving settings never starts
extraction. Pinning the same pair previously inherited from Chat/RAG also preserves
compatible work. To return to inheritance, run
`osintgpt config set graph_model_mode inherit`. This explicit choice overrides
independent user defaults and ignores any saved graph pair. An unset mode uses
project graph choices, then user defaults, otherwise Chat/RAG inheritance. In
inheritance mode, changing the chat model also changes the next graph build's model.

The build reads versioned extracted text from registered sources. Long documents
use overlapping text windows that prefer paragraph and sentence boundaries,
including complete sentences in the overlap when they fit. Very long sentences
or text without recognized punctuation use a bounded character fallback; source
text is preserved. Structured records are processed separately. Each
window starts with an extraction request. The graph model can request local
entity searches and evidence reads, continuing while useful work remains.
Quotation corrections and structured-response corrections can add requests;
their totals depend on the content. All model turns contribute to recorded usage;
local searches are reported separately. Graph builds reuse cached
PDF transcriptions. Missing page text is reported;
index the PDF with transcription enabled, then rerun `graph build`.

| Command | Operator use | Useful options |
|---|---|---|
| `graph build` | Build or refresh sourced relationships, resuming compatible saved windows | Unchanged completed sources are reused. `--incremental` performs the same refresh but requires an initial build; `--rebuild` discards saved work and extracts registered sources again. These flags cannot be combined. |
| `graph status` | Inspect saved build state, document/window counts, generation calls, and recorded usage | `--json` includes detailed counters and incomplete-usage indicators. |
| `graph diagnostics` | Inspect retained generation responses and failure reasons without model calls | `--offset` and `--limit` page through records; `--raw` explicitly includes original local response content. |
| `graph migrate` | Upgrade a supported older graph format without model calls | `--dry-run` previews the upgrade and counts without changing files. Applying it creates a SQLite backup and keeps `graph.sqlite` as the active file. |
| `graph entities QUERY` | Find identities by ID, name, or alias before selecting one | `--json` includes aliases, distinguishing context, and source references; `--limit` bounds candidates. |
| `graph neighbors ENTITY` | Inspect sourced claims around one unambiguous identity | `--depth` explores farther; `--limit` bounds assertions; `--direction` selects incoming, outgoing, or connectivity. |
| `graph path A B` | Find a shortest chain within the traversal limits | `--max-depth` bounds the path; `--direction` controls traversal. Ambiguous names return candidates. |
| `graph evidence ASSERTION_ID` | Read saved source text around a supporting quotation | `--quote-index` and `--occurrence` are zero-based; `--offset` follows the returned `next_offset`. |
| `graph rejected` | Review rejected entity/relationship proposals and successful quotation corrections | `--json` includes the candidate, reason, source version, window, timestamp, and correction outcome; `--ref` filters a file or record. |
| `graph reconcile` | Inspect mentions and identity decisions, combine reviewed identities, or separate a source contribution | Defaults to inspection/preview. `--query`, `--offset`, `--from-id`, `--to-id`, `--reason`, `--split-source`, `--undo`, and `--apply` support audited corrections with backups, without model calls. |
| `graph revalidate` | Recheck saved rejected proposals locally under current evidence rules | Defaults to preview. `--apply` backs up and publishes supported corrections; `--ref` restricts sources; `--offset` follows the returned `next_offset`. Missing original evidence ranges or endpoint IDs are explained and never guessed. |
| `graph verify` | Check every stored supporting quotation against saved text and current sources | `--json` distinguishes snapshot/current results; `--strict` exits non-zero for missing or unreadable evidence in current sources. |
| `graph export PATH` | Export entities, claims, qualifications, evidence, verification, and build context | The destination suffix selects `.json` or `.cypherl`; CYPHERL supports Memgraph/Neo4j import. |

All graph commands accept `--project CASE_SLUG` and `--json`. After building,
inspection, verification, and export do not call a model. For example:

```bash
osintgpt graph entities "Neral-7" --json
osintgpt graph neighbors "Neral-7" --depth 2 --limit 20 --json
osintgpt graph path "Neral-7" "Station Kestrel" --max-depth 4 --direction outgoing
osintgpt graph rejected --json
osintgpt graph rejected --ref "records.csv#case-4821" --json
osintgpt graph verify --strict
osintgpt graph export graph.json
osintgpt graph export graph.cypherl
```

Use a returned entity ID when a name is ambiguous. Copy an `assertion_id` from
`neighbors --json` or `path --json` into `osintgpt graph evidence ASSERTION_ID --json`.
Evidence reads retain the saved text version even if the current file changed.
`quote_complete` reports whether the selected quotation fits the returned page;
follow `next_offset` until the required text has been read.

`outgoing` follows stored source → target arrows, `incoming` follows them backwards,
and the default `connectivity` permits both. Traversal never reverses the assertion:
if A funds B, reading the incoming link from B still says A funds B. Paths can
include negative or alleged assertions and do not establish an affirmative fact.
CLI graph queries cap depth at 4, returned assertions/candidates at 60, and traversal at 256
identities and 2,000 examined edges. Inspection responses default to a 24,000-byte
evidence budget, configurable with `--max-bytes` up to 65,536. JSON reports
truncation and its reason. An oversized path is marked incomplete and retains its
assertion IDs for separate evidence reads; it is never presented as a complete
path with a missing link.

Source scope defaults to all published sources. Repeat `--ref SOURCE` on
`entities`, `neighbors`, or `path` to select sources, with no 20-source ceiling.
Identity candidates show direct assertion counts and distinct connected identities
across all sources and within the selection. Several sourced assertions can connect
the same two identities. These counts include both arrow directions and all
qualifications, before traversal direction and display filters. An isolated node, a node excluded by the source filter,
and a query limited by direction or budget have different explanations.

In the Graph tab, neighborhood counts also include assertions connecting other
identities reached during traversal. The result summary separates assertions found,
assertions returned, matches within that returned result, and table/diagram counts.
While exploration is running or stopped, the total is unknown. The Graph tab no longer
drops assertions or entire paths because their evidence exceeds 64 KiB. Its table
shows short quotation previews; select an assertion to inspect its full quotations
and provenance. Quotation searches and prepared exports use the full saved text,
including text beyond those previews. Inspection and exports make no model calls.
Run an existing query again to load the new result format; no rebuild or migration
is needed for this browsing change.

Use **Rearrange layout** to spread the displayed nodes with a local force layout.
Drag an individual node to position it; drag empty space to pan, and scroll to
zoom. The starting identity has a purple fill and contrasting outline; the legend
distinguishes it from relation-polarity colors. Positions remain during evidence
selection and display-filter changes for the same result. **Reset layout** restores
the displayed nodes to their circular positions and resets the view; a different query
or graph revision starts a fresh layout. These controls change only the session's
display, make no model calls, and do not change counts, relationships, or exports.
Distance and position do not indicate importance or evidential strength.
The canvas is 560 pixels high. **Full screen** expands it to the display; use
**Exit full screen** to return. **View data** switches to a table of the same
displayed assertions, and **View graph** restores the diagram. **Download PNG**
and **Download SVG** save its current layout, colors, arrows, and legend locally.
Image downloads capture the displayed graph; use the existing graph-data export
controls for evidence and provenance.
These controls appear as icons in the graph's upper-right corner; hover or focus
an icon to read its label. The identity and **Relation polarity** legend sits
inside the graph.

**Traversal depth** defaults to **1** for direct connections. Enter a larger
positive depth using the input or `+`, then click **Explore neighborhood** (or
**Find connection**). Increasing depth reuses existing progress when the identity,
sources, direction, and graph are unchanged; lowering depth starts a new query.
**Direct connections** starts again at depth 1. The app continues in work batches,
without the former depth-4, 60-assertion, or total node/edge ceilings. **Stop
exploration** pauses after the current batch; **Continue exploration** resumes
the saved frontier in this session. Reaching the chosen depth is normal scope
completion. A connection is shown only when its complete path is found.

**Rows per page**, **Previous page**, **Next page**, and **Table page** let you
inspect all discovered assertions. **Diagram selection** defaults to **Current
table page**, so page navigation changes both the table and graph. You can instead
show a chosen number of first matching assertions, assertions chosen across pages,
or no diagram; those selections are independent of table navigation.
**Show more assertions in diagram** adds
to that display without expanding the search. Counts distinguish discovered
assertions, filter matches, table rows, and diagram assertions. Filters and query
exports apply to discovered results; an unfinished query remains labelled incomplete.
Neighborhood rows follow traversal discovery: nearer steps first, then saved
database order within each visited identity. Positions such as 1–10 or 31–37 are
not importance, confidence, or date rankings. Connection rows follow path order.

Larger explorations use more local time and memory. Progress is retained in this
session; restarting the app requires a new query. If a build or identity review
changes the graph, or checked sources change, run a new exploration before continuing
or exporting. Evidence checks are reused within the query; source content,
availability, registrations, and cached text changes invalidate that check state.
Every discovered assertion is checked before entering the table. While exploration
is unfinished, the remaining assertion count and final verified total are unknown. These
controls make no model calls. Model-tool and CLI retrieval retain their existing
depth, result, work, and response-byte limits.

Rejected proposals do not enter the accepted graph. An invalid optional entity
alias, identifier, or extra quotation can instead produce a field warning while
the supported entity and dependent claims remain. Audit history survives
rebuilds and also contains repaired quotations and locally accepted corrections, so its length is not
the current number of unresolved claims. Use `graph status` for the current build
state and `graph rejected --json` to investigate individual outcomes. The Graph
tab displays saved work, source refresh details, published extraction models,
and rejected/corrected counts. Settings retains a brief graph-status summary.

Quotation checking first matches literal text, allowing Unicode NFC and whitespace
differences. For Markdown representations, it can also match recognized emphasis,
headings, blockquotes, and links while retaining the original excerpt, character
locations, source version, and matching method. Plain text stays literal; code,
URLs, mathematical expressions, case, diacritics, and meaningful words remain
significant. Unsupported formatting still requires a source-faithful quotation.
There is no stop-word removal, translation, or fuzzy acceptance. A located quote proves textual presence;
the model still interprets the relationship. Claims retain relation polarity, allegation or
uncertainty status, attribution, and separate event/statement dates when available.
Missing optional metadata stays empty or unknown. A graph path can contain negative
or alleged claims and should be read with its evidence and qualifications.
The field is `relation_polarity`, displayed as **Relation polarity**, with unchanged
values `positive`, `negative`, and `unknown`. The relationship wording and separate
assertion status retain their existing meanings; polarity is not sentiment or confidence.
Date fields retain the source's wording and language; they need not be ISO dates.

Graphs in older formats, including schema 7, require an explicit `graph migrate` upgrade
to schema 8 before inspection or building. Migration renames `polarity` to
`relation_polarity` in published assertions and saved accepted windows while
preserving values, IDs, evidence, and usage. Raw historical responses, rejected
proposals, identity decisions, and saved conversation evidence retain their
original field names. The rename alone requires no model calls or rebuild.
Compatible completed sources/windows remain reusable; compatible older saved
responses are normalized in memory while retaining their exact original request
hashes. See [migration and recovery](operator-documents/graph-build-and-recovery.md#upgrade-an-existing-graph-to-schema-8).

During extraction, the graph model can search published identities and accepted
mentions from the current build, then read their sourced evidence before proposing
reuse. Matching names alone do not combine nodes. Compatible identifiers or an
explicit, evidence-backed contextual decision can justify reuse; unresolved
mentions keep separate IDs and their supported relationships. Type compatibility
and evidence are recorded for review. These checks make decisions traceable;
they cannot prove that the model interpreted identity correctly.

Use `osintgpt graph reconcile --query "ENTITY NAME" --json` to inspect mentions,
candidate evidence, and decisions. To combine reviewed identities, preview with
`graph reconcile --from-id SOURCE_ID --to-id TARGET_ID --reason "SOURCE-BASED REASON" --json`,
then repeat with `--apply`. Applying backs up the database and retains original
assertions and old-ID mappings. `--undo DECISION_ID --apply` reverses a compatible
operator merge; undo dependent later merges first. To correct reuse across files,
use `--from-id ID --split-source SOURCE_REF --reason "REASON"`, preview, then apply.
Splitting separates that source's entire contribution, not an individual mention
inside it. These actions do not call a model or require re-extracting the corpus.

Graph builds no longer require an output-token setting. OpenAI and Gemini use
their provider defaults; Anthropic automatically resolves the selected model's
required output allowance and receives the response through streaming. Provider
limits still apply. Missing required Anthropic metadata is reported before
extraction. Complete responses are validated for structure and source evidence
without the former 200-object or 256-KiB rejection thresholds.

`--source-window-bytes` (also accepted as `--context-budget`) defaults to
32,768 UTF-8 bytes of serialized source text. The 24,000-character window cap
and overlap also apply. This sizes source passages; it does not cap the growing
identity conversation. Continuation and evidence delivery are automatic: useful
lookups can continue across new requests, with exact evidence saved and retrievable.
Known provider capacity can cause smaller passages or a new conversation. Unknown
model capacity is reported as unknown; serialized bytes are not billed tokens.

There is no fixed identity-turn or per-record call ceiling. Follow-up requests
continue while useful work remains, so final request totals and cost are unknown.
Use cancellation or an optional spending ceiling when needed. `--no-repair-quotes`
disables quotation correction. The retired `--max-record-calls` and
`--output-tokens` flags explain their replacements. Saved app `context_units`,
`max_calls`, and `output_tokens` values are ignored and removed on the next save.
Compatible earlier windows and received responses can be recovered without a
rebuild; preview first. Changing source-window sizing can require fresh extraction.

Preview the workload without calling a provider or changing the graph:

```bash
osintgpt graph build --dry-run --json
osintgpt graph build --rpm 20 --tpm 100000 --retry-attempts 4 --retry-seconds 120
```

The preview counts readable records, saved windows/responses, and pending
windows, and identifies unchanged, unavailable, and removed sources. Saved raw
responses still require validation and can reduce new requests. Follow-up counts
and future spend are unknown. Anthropic limits can be read from matching saved
metadata; otherwise they remain unresolved until the build starts. Reading
sources may populate the local text cache.

`--rpm` and `--tpm` apply to graph generation independently of embedding quotas.
Both default to zero, which adds no operator ceiling; provider retry delays still
apply. TPM pacing reserves input bytes plus the known model output allowance;
with provider-default output it estimates twice the input bytes. This estimate
does not constrain generation and is not an exact billed-token count.
These flags apply to this invocation; repeat them when resuming. Processes using
the same CLI home, provider endpoint, credential, and model share generation
pacing. Transient failures use at most six HTTP attempts per request over a
300-second retry budget by default. Active generation uses a separate 600-second
read timeout; streamed responses can continue while data arrives. The retry time
budget limits further attempts, not an already-running response.
Authentication, billing, permission, and invalid-request failures stop the build.
Malformed model output is handled separately through a targeted structured-response
correction, with both responses retained. Repeated invalid output stops that source
with diagnostics; it does not trigger an endless retry loop. Local storage failures
are not retried. Changing pacing keeps compatible saved extraction work.

Graph extraction and quotation correction request structured JSON through
[OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs),
[Anthropic](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
[Gemini](https://ai.google.dev/gemini-api/docs/openai#structured-output), and
[Ollama](https://docs.ollama.com/api/openai-compatibility) adapters. The selected
model and endpoint must support the requested schema; a rejection is reported
without automatically repeating the request in a weaker output mode. Custom
providers without schema support continue with local validation. Interactive
extraction uses native tool turns with a structured final response, or a structured
submission tool where the adapter cannot combine tools and JSON response formats.
The workload preview discloses adapter support; it does not contact the provider
to establish the chosen model's capabilities. Unsupported model responses fail
explicitly without an automatic paid fallback. Adapters without interactive support
retain the supplied-context and identifier-based extraction route and report that limitation.

Reported truncation, refusal, or incomplete generation cannot become an accepted
extraction, even if the returned JSON looks valid. A failed correction leaves its
candidate rejected and records the reason. All responses still undergo local
output and quotation checks; structured JSON does not establish that an
interpretation is correct. `graph status --json` reports calls without provider
schema enforcement (`structured_output_unenforced`) or completion metadata
(`completion_status_unavailable`). If the provider still reports truncation,
reduce the source input budget or choose a model with more output capacity;
changing extraction settings requires fresh extraction.

Inspect saved generation failures without calling a model:

```bash
osintgpt graph diagnostics --project CASE_SLUG --json
osintgpt graph diagnostics --project CASE_SLUG --raw --json
```

The default view shows response state, request identity, completion status, and
parser/protocol problems. `received` awaits validation; `invalid` retains a failed
response; `superseded` identifies work tied to a replaced request. Response
validation and acceptance of individual claims are separate: use `graph rejected`
for evidence and identity decisions. `--raw` explicitly includes saved response text and
provider continuation; it can contain source material. Use `--offset` and `--limit`
to page through entries. The app offers the same inspection under **Generation
diagnostics**. Invalid responses remain available for diagnosis and are not
silently replayed as successful extractions. Responses deleted by older releases
cannot be recovered. No additional schema migration is needed for these audits.

Usage messages distinguish an active request awaiting usage, completed requests
with missing usage, and recorded tokens whose model price is unknown. Unpriced
tokens do not by themselves mean extraction failed. Progress also separates local
lookups, reused responses, recovered legacy work, and conversation restarts.

After an interruption, rerun `osintgpt graph build` with the same extraction options.
Completed windows, received extraction/correction/tool-turn responses, and local
lookup results are saved in
`graph.sqlite` and reused when the source text,
mapping, model, prompts, windowing policy, output contract, and extraction settings
still match. Changed inputs require fresh extraction; source refresh details
explain incompatible checkpoints separately from each source's processing outcome.
An unusable checkpoint does not mean its source was deleted or skipped. Previous
published claims remain available until that source's replacement completes.
Saved identity proposals are rechecked against current published candidates before
publication; lost identity support leaves a separate unresolved mention with its
supported claims. Built-in providers save responses before validation,
quotation repair, or cost accounting can stop the run. Reused responses undergo
the same evidence checks without being counted as new calls or new usage. Invalid
or truncated responses are discarded after validation, so they do not block a
later retry. An interrupted HTTP request with no saved response may need repeating.

Usage and call counts describe the latest run; `graph status --json` reports HTTP
attempts, retries, reused windows/responses, latest activity, and up to ten previous
runs. Missing usage from failed attempts is disclosed. Saved pending-window counts
cover started records; `--dry-run` includes later records too. Settings in the app
shows this recovery state. Custom providers without request/response hooks retain
provider-call counts; `request_controls_unenforced` indicates that controlled
retries and saving before cost accounting cannot be guaranteed.

Graph status reports the latest run's model separately from `extraction_models`,
which describes each published source's saved model and source version. A partial
refresh can leave sources extracted by different models. Missing historical model
information remains unknown; changing settings does not relabel stored claims.

Builds refresh new or changed sources and remove contributions from sources you
unregister. A missing or unreadable file that is still registered is reported as
unavailable and retains its published claims. A failed update also keeps the prior
document's claims; its replacement becomes available only after processing finishes.
Status flags a graph that no longer matches the current sources. Use `--rebuild`
to repeat extraction, including documents with rejected candidates.

For an older graph format, preview and apply an explicit migration:

```bash
osintgpt graph migrate --project CASE_SLUG --dry-run
osintgpt graph migrate --project CASE_SLUG
```

Migration supports the original unversioned format (version 0) and the known
version-5, version-6, and version-7 layouts, upgrading them to version 8. It works with graph retrieval
disabled and makes no model calls. It creates a timestamped
`graph.backup-vVERSION-*.sqlite` beside `graph.sqlite`, validates the conversion,
and commits the upgrade together. Failed conversions roll back. Unknown layouts
are refused. Stop running graph builds before migrating; another build or database
writer prevents the upgrade. `--json` includes record and field-rename counts, schema versions, and the
backup path. A current graph is left unchanged; an absent graph is not created.
After migrating, refresh the app's **Graph** view to read the upgraded status.
Migration preserves existing claims and saved work; it does not merge namesakes
or run extraction. Identity review, source filtering, and current evidence checks
work on the upgraded graph immediately. The schema-8 field rename alone does not
invalidate completed sources or compatible saved work. Genuine source, model,
or extraction-rule changes can still require processing on the next explicit
build; use a workload preview first. Local
`graph revalidate` can recover saved proposals only when their original source
ranges and endpoint identities were recorded. Older incomplete audits still need
explicit re-extraction; the command explains these cases without spending tokens.

An empty original-format graph becomes **never built**. Populated original-format graphs
retain their entities, relationships, and quotations, with missing qualifications
left unknown. Graph status, the app, and retrieval responses identify legacy
records needing re-extraction. Migration cannot reconstruct historical source
snapshots, usage, or previously merged identities. Use an explicit `graph build`
to re-extract registered legacy sources with the new checks; this can make model
calls. Missing sources retain their claims, and entities without source associations
may remain marked as legacy. Version-5 upgrades preserve saved build/checkpoint
data and existing extraction metadata. `--rebuild` repeats extraction but does not
migrate an incompatible schema.

Renaming `graph.sqlite` to another filename is a manual alternative when you want
to start a fresh graph and retain the old file separately. OsintGPT only opens
the exact active filename; it does not discover or import renamed backups.
Turning `graph_enabled` off preserves stored claims while disabling
extraction and the answering model's graph tools. Saved graphs remain inspectable
in the app's **Graph** view.

Graph validation currently covers offline regression tests and synthetic document
recovery. Multilingual extraction accuracy, live answer improvement, and cost or
memory requirements for representative long documents remain unmeasured. A quote
matching its source establishes textual support; it does not establish that the
relationship was interpreted correctly or that the source's claim is true.
Inspect identity, qualifications, and surrounding evidence before relying on a claim.
The [benchmark guide](tests/benchmarks/README.md) explains offline validation and
prepared comparisons that hold either the answering model or the extracted graph
and supplied evidence fixed. Reports separate extraction, answering, and embedding
usage; the paired runner defaults to offline fixtures. Live evaluations remain pending.

### Work with the graph in the app

Select a project and open **Graph** in the sidebar. The view shows saved build
state, entity/assertion counts, current source coverage, extraction gaps, and
last-run details without contacting a provider. A completed source with no
relationships still counts toward coverage. Refresh the status after external
source changes.

Build activity and published coverage are shown separately. "The last graph build
finished with gaps", "failed", or "was interrupted" describe the previous run;
they do not mean a build is running now. A stale graph needs an explicit update.
Published claims remain inspectable when available, and missing run counters are
shown as unknown. An unmatched identity search or an identity with no accepted
relationships has its own query message; neither proves that the source material
contains no relationship.

The [graph operator guides](operator-documents/README.md) explain each status and
warning, build recovery, coverage counters, source refresh outcomes, evidence
checks, saved-proposal revalidation, exports, and manual identity review. Follow
the [combine/undo/source-separation workflow](operator-documents/graph-identity-mentions-and-resolution-decisions.md#combine-identities-in-the-app)
to correct identities after reviewing their sources, without another model call.

In **Settings**, choose **Chat/RAG provider/model** for answering and use
**Graph extraction** for graph builds. **Use Chat/RAG settings** explicitly selects
inheritance; turn it off to choose an independent provider/model. Changing graph
providers clears the model field rather than carrying an unrelated identifier.
The app shows the effective choice and whether its credential is available locally;
this does not test provider access. Saving applies the choices to both CLI and app.

Under **Build and recovery**, preview the workload, then explicitly start a
build, update changed sources, resume saved work, or rebuild all sources. Enable
**Relationship graph** in Settings before building. Builds run independently of
the page: navigation, browser refreshes, and another session do not start another
build. **Cancel build** requests a stop at the next safe boundary; an active model
request may finish first. If the app process stops, reopen it and explicitly
resume compatible saved work. A build started by the CLI must be cancelled there.

**Build settings** exposes source-window sizing in UTF-8 bytes, quotation
correction, RPM/TPM, retries, and an optional run cost ceiling. Identity context
and continuation are automatic.
The build caption shows the resolved graph model and whether it inherits Chat/RAG
settings. Save preferences to reuse them in later app sessions; these project-local
preferences do not replace CLI flags.
An active build keeps its starting settings. A cost ceiling is checked after a
response, so it can be exceeded by that response; unknown usage or pricing also
stops a run with an active ceiling.

The next-build model and a running build's model are shown separately. A changed
model or build option clears the cached workload preview; preview again before
starting. Published model details remain available independently of these choices.

To explore, search a name, alias, or entity ID and select the intended identity.
**Neighborhood** returns nearby assertions; **Connection** asks for a second
identity and displays evidence for every returned step. Source and direction
filters apply before traversal. **All sources** includes every published source;
**Selected sources** requires an explicit selection and supports more than 20.
Selecting none does not silently switch to all sources. Identity choices show
relationship counts for both scopes. Relationship, quotation-verification, and text
filters narrow discovered results before table pagination. **Traversal depth**
starts at 1; increase it and run the query to continue from the existing frontier
when the other query settings and graph are unchanged. **Stop exploration** pauses work and **Continue exploration**
resumes it. **Rows per page** and **Diagram selection** independently control
the table and canvas; the diagram follows the current table page by default.
Unfinished exploration shows results found so far; it does
not establish that no relationship exists.

Click a relationship dot, select a table row, or use **Inspect assertion** to see
every supporting quotation, saved source version and location, qualifications,
and saved/current verification. **Read source context** opens saved text around
the selected quotation. Other assertions for the same pair are available for
comparison; the app does not automatically decide whether they corroborate or
contradict one another. **Evidence checks and rejected proposals** offers a full
verification summary and a bounded audit preview. These inspection actions make
no model calls.

**Identity mentions and resolution decisions** supports inspection, preview,
application, and correction of identity decisions. **Recheck saved rejected
proposals** previews local evidence corrections before applying them with a backup.
Both workflows are available without starting a graph build.

New tool-based answers in **Ask** retain graph claims, paths, and source-context
previews in their saved conversation. Older answers without saved tool evidence
cannot gain previews retrospectively. Verification shown in a saved answer is
the result at answer time; use Graph to check current sources again.

### Export graphs, answers, and findings

In **Graph**, open **Export full graph** and choose **Prepare graph downloads**
for JSON or CYPHERL. This includes all published entities/assertions and checks
their evidence against saved and current source text. An incomplete or stale
graph can still be exported; its status travels with the file.

After a neighborhood or connection query, open **Export query results**.
Choose **All matching results**, **Current table page**, or **Displayed diagram**,
then prepare the downloads. All matching results applies the current filters across
the discovered query scope before pagination; it is independent of the diagram.
These exports retain the evidence and verification
shown by that query, its direction, source filters, query limits, and traversal
order. Metadata records exactly which assertions were exported and whether the
query and matching set are complete. A subset of a path is not labelled a complete
path. A stopped or unfinished search has an unknown final matching total.
Graph or checked-source changes hide selected-query downloads on the next query
check, such as a page/filter change or app rerun; run a new exploration and prepare
them again. Downloading an already prepared file serves that fixed snapshot
without rechecking sources.

JSON retains the `entities` and `edges` arrays and adds `metadata` with the
project identity, source versions, build timestamps, selection/filter context,
and stale/incomplete/limited status. Edges include the same citation,
qualification, location, and verification fields used by retrieval. CYPHERL
stores nested evidence and verification as JSON strings, adds an
`OsintGPTExport` metadata node, and scopes imported entities by project ID.
Importing updates verification properties without creating another copy of the
same assertion. It does not remove assertions from earlier imports.

Export metadata includes saved extraction models by source/version, identifying
mixed and unknown models separately from the latest run's selection. Exports of
displayed results retain the model metadata captured with that selection.

In **Ask**, open **Export answers and findings**. Select answers from the open
conversation, existing Markdown pages under `canon/`, or both, and prepare
Markdown or CSV downloads. Answer exports retain the original response, cited
sources, and recorded passage/graph evidence, including uncertainty and source
versions. Canon pages retain their authored text and links plus a content hash;
their linked sources are not independently verified. Older answers without
saved evidence are explicitly labelled. Nothing is summarized by a model.

CSV uses UTF-8 and stores structured citations/evidence as JSON inside cells.
Formula-like text receives a leading apostrophe for spreadsheet use; the
`spreadsheet_escaped_fields` column identifies affected cells so software can
recover their exact original text by removing that one added character.

Downloads are fixed snapshots of the last preparation. Prepare again after
changing sources, a query, or authored findings. Identical inputs produce
identical files, without an export-time timestamp or model calls.

See the [graph walkthrough](examples/cli/04-the-graph.md) for evidence provenance,
entity identity, and extraction limits. Use `osintgpt graph --help` for the command
list and `osintgpt graph COMMAND --help` for a command's options.

## Canon

Every project has a `canon/` directory for plain Markdown synthesis. Pages in
it are indexed automatically alongside primary material, and `[[wiki links]]`
work in Obsidian. `osintgpt` does not yet populate those pages; today it
provides the directory, indexing, and link structure.

## Keep data local

**Fully local means the `[local]` extra for embeddings,
[Ollama](https://ollama.com) running on your machine for generation, and the
default SQLite store; nothing leaves your machine at query time.** These are
separate pieces: the Python extra installs sentence-transformers and torch,
while Ollama is a separately installed local server and costs nothing from
pip.

```bash
osintgpt config set embedding_provider sentence-transformers
osintgpt config set generation_provider ollama
osintgpt config set generation_model MODEL_NAME
osintgpt doctor
```

Replace `MODEL_NAME` with a model already available to Ollama. A model fetched
on first use needs a network connection during setup; operation is local after
that download. `doctor` runs offline by default and reports what would leave
the machine, provider readiness, stored models, source coverage, and embedding
model mismatches. Add `--check-providers` only when you want it to contact
configured services.

## Supported document formats

`osintgpt` reads most formats with its own readers. A few formats it has no
reader for go to a fallback converter,
[markitdown](https://github.com/microsoft/markitdown), as a last resort. Some
formats are better converted before they are added to a project, because the
fallback either fails or keeps their markup as text.

| Format | Extensions | Read by | Notes |
|---|---|---|---|
| Plain text, Markdown | `.txt` `.md` `.markdown` `.rst` `.log` | osintgpt | |
| HTML | `.html` `.htm` | osintgpt | |
| PDF | `.pdf` | osintgpt | Scanned pages need a transcriber (`--vision`) |
| Word | `.docx` | osintgpt | |
| Excel | `.xlsx` `.xlsm` | osintgpt | Needs a content-field mapping |
| CSV, JSON | `.csv` `.json` `.jsonl` `.ndjson` | osintgpt | Needs a content-field mapping |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tif` `.tiff` | Embedding model | Embedded directly when the model supports images; no text is extracted |
| PowerPoint | `.pptx` | markitdown | |
| Outlook email | `.msg` | markitdown | |
| EPUB | `.epub` | markitdown | |
| RSS / Atom feed | `.xml` | markitdown | Feed items are converted to markdown |
| Other XML | `.xml` | — | **Convert first** to `.json`, `.jsonl`, or `.csv` |
| Rich Text | `.rtf` | — | **Convert first** to `.docx` |
| OpenDocument text | `.odt` | — | **Convert first** to `.docx` |
| Legacy Excel | `.xls` | — | **Convert first** to `.xlsx` or `.csv` |

Why convert first:

- **`.rtf` and non-feed `.xml`** pass through the fallback unchanged: control
  words such as `\rtf1\ansi\fonttbl` or tags such as `<record><author>` are
  chunked, embedded, and indexed as if they were content. That adds noise to
  semantic and exact search and spends graph-extraction tokens on markup.
  XML also loses its structure: an `<author>` element becomes text rather than
  a field retrieval can filter on.
- **`.odt`** is not converted at all; the file is skipped with a warning.
- **`.xls`** has no reader and is not listed as readable.

Word processors and spreadsheets export these formats directly. With
LibreOffice installed:

```bash
soffice --headless --convert-to docx report.rtf
soffice --headless --convert-to docx notes.odt
soffice --headless --convert-to xlsx ledger.xls
```

XML has no single conversion, as each schema is different. Export the records
to `.json`, `.jsonl`, or `.csv` from the source system or with a short script,
then register the file with a mapping (`--map content=<field>`, plus
`--map author=<field>` and `--map timestamp=<field>`) so those values stay
fields.

`osintgpt convert <path>` prints the markdown a file would be chunked from,
without registering, embedding, or storing anything. It needs no project and
makes no API call: pages holding no extractable text come out as named gaps.
Pass `--map content=<field>` for a structured file. `--out` writes instead of
printing: converting a single file, it specifies the output filename;
converting a directory, it must be a directory, where the resulting `.md`
files are written with the original structure preserved.
Adding `--vision` (or `--model <name>`, which implies it) is the only way to
spend anything here — it transcribes those pages at one generation call each,
caching into the project's `extracts/` when `--project` is given, so the same
page is not paid for again at index time. The same panel is in the app's
**Material** view.

## Use it as a library

The [CLI quickstart](#quickstart) is the shortest route. The same project and
retrieval APIs are available to Python callers:

```python
from osintgpt import Project, Settings, search_project
from osintgpt.llm import build_embedding_provider

project = Project.load("/path/to/project")
settings = project.settings_for(Settings.from_env(".env"))
embedder = build_embedding_provider(
    project.settings.embedding_provider, settings
)

for hit in search_project(project, "What does the evidence say?", embedder):
    print(hit.score, hit.chunk.citation)
```

Configuration is passed into library calls; importing `osintgpt` does not read
the environment or select a project behind the caller's back.

## Where projects live

The default home is `~/.osintgpt`, with each project under
`projects/<slug>/`. `project.toml` holds non-secret choices, `sources.toml`
records registered locations and field mappings, and `store.sqlite` contains
the default local vector store. Back up the project directory together with
any registered material stored outside it; Qdrant and Postgres stores need
their own backup.

Indexing keeps successful embedding batches in `index-journal.sqlite` until
the file is fully embedded and published to the vector store. If an OpenAI,
Gemini, or Voyage embedding request fails, click **Index now** again or rerun
`osintgpt index` for the same project to reuse the saved batches and process
the remainder. Keep the workbook, field mapping, embedding configuration, and destination the same
when resuming. `--force` discards pending checkpoints and starts a fresh pass;
resume an interrupted forced pass without `--force`.

Pending chunks are not searchable. Successful responses are checkpointed
before a cost-ceiling stop, and the journal is cleared for a file only after
publication and index-state persistence. A response lost before it reaches
the checkpoint may still need repeating. Custom embedding providers retain
whole-call behavior unless they implement `embed_checkpointed`.

New index state records the source hash and fingerprints of the field mapping
(including filters and identity), chunking rules, embedding provider/model/options,
and destination. Changing these makes the next indexing pass rebuild the affected
files automatically. Changing batch size, credentials, or a spending limit does
not invalidate otherwise compatible work. An unchanged pass also checks that the
destination still contains the expected chunks; missing or incomplete files are
scheduled again. The provider selected when the pass starts stays in use for that
pass. Record identities are now stored separately from the file reference.

Older completion records get a one-time local comparison of every stored chunk
against the current source, mapping, text, metadata and model. Matching indexes
using a standard OpenAI embedding model at its default endpoint can be adopted
without embedding calls. This infers the original endpoint and default options:
older stores did not record them. Other providers, custom endpoints, or a mismatch
require an explicit `osintgpt index --force`; the existing chunks are preserved
until rebuilt. Older chunks acquire the new record-identity field on their next
rebuild. Changing a Qdrant or Postgres vector size requires a destination that
supports the new size.

Paid embedding providers now support RPM/TPM pacing. In Streamlit, open
**Settings → Embedding rate limits** after choosing OpenAI, Gemini, or Voyage.
Enter that provider's requests-per-minute and tokens-per-minute quotas, or lower
budgets reserved for this workload. These settings are stored separately for each
provider. For example, an operator with a 2M TPM allocation can configure the
selected project through the CLI:

```bash
osintgpt config set openai_embedding_tpm 2000000
osintgpt config set openai_embedding_rpm 5000
osintgpt config set openai_embedding_batch_tokens 50000
```

Use your actual allocation for each number; these commands are not tier defaults.
Replace `openai` with `gemini` or `voyage` for those providers. `--user` saves user
defaults; `unset` restores inheritance, and `0` removes an operator ceiling.
The matching environment variables are `OSINTGPT_OPENAI_EMBEDDING_TPM`,
`OSINTGPT_OPENAI_EMBEDDING_RPM`, and `OSINTGPT_OPENAI_EMBEDDING_BATCH_TOKENS`.
Explicit environment/library settings take precedence over project and user values.

OpenAI can learn quota ceilings and remaining capacity from
[response headers](https://developers.openai.com/api/docs/guides/rate-limits).
When limits are unknown, its first request embeds one input to obtain those
headers; subsequent batches adapt to the reported limits. A configured lower
ceiling still applies. Set Gemini and Voyage RPM/TPM explicitly from their
[Gemini](https://ai.google.dev/gemini-api/docs/rate-limits) or
[Voyage](https://docs.voyageai.com/docs/rate-limits) dashboard. An unset ceiling
without usable provider headers cannot pace that quota; it does not infer a tier.

The limiter spaces requests and checks a rolling minute of reserved requests and
tokens. Batches respect both input count and token budgets, keeping input order
and leaving text intact. Known OpenAI models use their tokenizer; Gemini, Voyage,
and unknown models use a conservative UTF-8 byte estimate with a framing allowance.
That estimate can reduce throughput and is never substituted for billing usage.
Oversized individual inputs fail with an explanation instead of being truncated
or waiting indefinitely. Provider request ceilings also apply, independently of TPM.

CLI and Streamlit runs using the same osintgpt home coordinate through
`embedding-rates.sqlite`. By default, quotas are grouped by provider, endpoint,
API key and model. Set `openai_embedding_quota_scope` (or the corresponding Gemini
or Voyage field) to the same label across keys/projects/models that share one
provider quota. Only hashes, counts and timestamps are stored in the ledger.
Separate homes can share an explicit `OSINTGPT_EMBEDDING_RATE_STATE_PATH` on the
same machine. Standalone library providers use an in-process ledger unless given
`Settings(embedding_rate_state_path=...)`.

Pacing settings do not invalidate embeddings or checkpoints. Successful vectors
reach their checkpoint before quota updates or cost accounting can stop a run.
SDK retries are disabled for paid embedding calls; osintgpt owns the retry loop.
OpenAI, Gemini, and Voyage automatically retry temporary rate limits, supported
transient HTTP errors (408, 409, 429, 500, 502, 503, 504), and SDK transport failures.
The default is **six total attempts per unfinished batch**, with a **five-minute
retry scheduling budget** starting with the first HTTP attempt, including HTTP
time and waits for retry quota capacity.
Each HTTP attempt has a 60-second SDK timeout; an attempt already in progress can
finish after the scheduling budget. Successful batches start a fresh budget.

Retries follow valid `Retry-After` seconds or HTTP dates, `retry-after-ms`, Gemini
`RetryInfo`, and explicit retry delays in provider messages. The largest valid
hint is a minimum. Exponential backoff starts at one second, grows to 60 seconds,
and adds up to one second of jitter. A provider's longer wait is never shortened
to that backoff cap: if the wait cannot fit the remaining retry budget, indexing
stops with the original provider error. This follows the providers' recovery
guidance for [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits),
[Gemini](https://ai.google.dev/gemini-api/docs/troubleshooting), and
[Voyage](https://docs.voyageai.com/docs/rate-limits).

Every retry obtains a new RPM/TPM reservation. Failed attempts stay charged to the
local rate window, and cooldowns are shared with other runs using the same quota
group. If newly reported TPM requires smaller batches, the unfinished inputs are
repartitioned in order. A single input that cannot fit still stops with an error.
Waits remain interruptible; cancelling does not reserve or send another attempt.

Invalid credentials, invalid requests, recognized billing/credit/spend errors,
and identified daily or zero-quota errors stop immediately. Unknown 429 errors
receive bounded retries; their status alone cannot identify every provider quota
condition. Local storage failures, invalid embedding responses, and cost-ceiling
stops are never retried. Only returned usage is recorded; a transport timeout can
leave provider-side processing and charges uncertain.

Retry limits apply automatically in the app and CLI. Library callers can pass
`retry_policy=RetryPolicy(...)` from `osintgpt.llm.embedding_retry` to
`OpenAICompatEmbedding` or `GeminiEmbedding`; `max_attempts=1` disables automatic
retries. These operational options do not change the index fingerprint.
After retry exhaustion, rerun indexing with the same vector configuration to
resume completed checkpoints. Reserve headroom for other machines or applications
that do not share this local quota ledger.

The app's **Material → Indexing** panel shows the provider/model that will be
used before any client is built. It warns when completed indexes or checkpoints
record another provider/model. Settings changes made in another app session or
through the CLI take effect on the next app rerun.

During indexing, the current file shows chunks saved and pending, fully embedded
records/documents, and reused checkpoints. Structured sources also show kept
records and exclusions for missing content or the configured character minimum.
Totals appear after extraction; file counts and record counts are separate.
Pacing waits display the current delay, retries show their attempt number, and
the last run retains a short activity history and reported usage across reruns.
Failures and skipped files are presented as outcomes needing attention.

**Saved indexing status** reads completed metadata and checkpoints from disk,
so reopening the app shows recovery progress without contacting a provider.
Completed counts describe the last published versions; checkpoints can replace
those versions and become searchable when the entire file is published. Older
runs may lack provider or record-count metadata; missing counts are not zeros.
Indexing verifies the source, configuration and destination before reusing them.

Use **Index now** to update or resume. Keep **Rebuild all files and discard saved
checkpoints** off when resuming; selecting it explicitly recomputes unchanged
files too. To resume work made with a previous provider/model, restore that
configuration in Settings. Use the app's Stop control to interrupt indexing;
successfully checkpointed batches remain available after interruption.

The optional **Stop at estimated run cost (USD)** covers embedding and scanned-page
transcription responses in this run. A response can cross the ceiling; it is
saved before the run stops. Missing token usage or an unknown price also stops an
active ceiling. Usage totals cover returned responses, so failed transport
attempts may have unreported charges. **Preview what would be indexed** describes
the full registered corpus, including already completed material; its cost is
not an estimate of the remaining resume cost.

## Testing

From the active source checkout, use the project's Python environment:

```shell
python -m pytest
python -m pytest -ra --durations=20
```

The first command is serial and shows pytest's normal header, progress and
summary. The second adds skip/failure reasons and slow-test timings. Coverage
and parallel workers are explicit options. See [the test guide](tests/README.md)
for offline isolation, tokenizer preparation, evidence capture and CI profiles.

## Responsible use

`osintgpt` analyzes material supplied by its operator. Use it only with data
you may lawfully process, protect personal and sensitive information, verify
answers against their cited passages, and understand any costs or data
handling terms of the providers you configure. The maintainers are not liable
for misuse or third-party service charges.

## License

Licensed under the [Apache License 2.0](LICENSE).
