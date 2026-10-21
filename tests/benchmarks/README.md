# Graph validation benchmarks

These commands run from the repository root with the development environment.
They create isolated temporary projects; they do not rebuild an operator project.
Phases 8 and 11 currently authorize offline work only. Live commands below are
prepared for a later operator-selected model, budget and reviewed corpus.

## Offline checks

```powershell
python tests/benchmarks/graph_quality.py --output extraction-replay.json
python tests/benchmarks/graph_quality.py --review-template --output annotation-review.json
python tests/benchmarks/graph_answers.py --output answers-offline.json
python tests/benchmarks/graph_answers.py --reuse-evidence answers-offline.json --output answers-reused-offline.json
python tests/benchmarks/graph_scale.py --pages 1000 --output recovery-1000.json
python -m pytest tests/benchmarks -q
```

`graph_quality.json` contains 14 fictional diagnostic cases: English, Spanish,
Chinese and Arabic; aliases and namesakes; allegations, negation, uncertainty,
dates, hypothetical claims and conflicting accounts; CSV/JSON records; and a
scanned PDF with an explicitly seeded transcription cache. The scan tests the
downstream graph pipeline, not OCR accuracy. The corpus is **draft**, not an
independently reviewed or representative research dataset.

Default extraction runs replay the annotations through the real builder,
identity resolver, quotation checks, persistence and scorer. Perfect replay
scores validate that path; they are **not model accuracy**. Tests deliberately
corrupt identities, direction, qualifications, evidence and duplicates to check
that the scorer detects errors.

The paired answer runner also defaults to offline execution. Its native SDK calls
use local HTTP fixtures, synthetic embeddings and scripted answers. No credentials
or real API requests are needed. These runs verify routing, persistence and usage
attribution; their token counts are simulated and actual provider spend is zero.
The script follows current identity checks: names alone cannot merge entities
across documents. Ambiguity and missing connections remain visible in each arm's
`graph_status`. Explicit fixture IDs, such as the selected namesake, resolve only
through a unique matching name and source reference in the extracted candidates.
The earlier passage proxy's expected graph statuses are not live extraction gold.

The scorer reports entity and alias precision/recall, directed relationship
precision/recall, fully qualified supported relationships, quotation validity,
and per-field qualification accuracy. Results include raw denominators, per-case
details, language/tag breakdowns and missing-run failures. Empty denominators are
null rather than a perfect score. Relationship and qualification metrics each
use one-to-one maximum matching so duplicates or prediction order cannot inflate
or hide correct matches. Field-level accuracy is measured independently; only
the fully qualified metric requires every field and supporting excerpt together.
Alignment only accepts a unique annotated
name/alias; namesakes additionally need distinguishing source anchors. Ambiguous
matches remain unresolved. Relations use explicit reviewed equivalents; there
is no translation heuristic or model judge. Qualification strings must match the
annotation. Unlisted valid synonyms or equivalent date formats need annotation
review before comparing runs.

Quotation validity reuses the production Unicode/whitespace-aware matcher; it
does not remove words or infer meaning. The fully qualified score additionally
requires the annotated supporting excerpts to be covered by returned quotations.
Thus an authentic entity-name quotation alone cannot prove the relationship.
This strict diagnostic measure can reject a shorter but semantically sufficient
excerpt; inspect the per-claim results before treating it as an extraction error.

`graph_scale.py` creates a text-bearing PDF with one fictional relationship and
18 filler lines per page. The real PDF loader, extraction windows, native SDK
serialization, rate ledger, checkpoints and publication run against a local
mock HTTP transport. It exercises a 429 with Retry-After, interruption, invalid
JSON structure, a cost stop after response persistence, resume, unchanged reuse,
and failed/successful replacement after a source change. Invalid output is retained
before targeted correction; repeated invalid completion makes no new request on
ordinary resume. The final retry of that failed replacement uses an explicit rebuild.
It checks the final
page location and preservation of the previous publication.

The report records stage latency, peak Python allocations, requests and simulated
usage. Rate waits use a fake clock. Simulated token counts and catalogue costs
are **not provider measurements**; actual spend is zero. Python allocation peaks
exclude native PDF/SDK memory, and tracing affects latency. This synthetic PDF
does not replace a representative 1,000-page operator benchmark.

## Annotation review and later extraction runs

Review the source text, identity anchors, accepted relations, evidence and every
qualification in `graph_quality.json`. Edit ambiguous gold cases before approving
them. Generate the review template again after any corpus change, enter reviewer
and date, and set `approved` only after reviewing every listed case. The corpus
hash invalidates an approval after any source or annotation edit.

Later, once live work is authorized:

```powershell
python tests/benchmarks/graph_quality.py --live --extraction-provider PROVIDER --extraction-model MODEL --max-cost-usd BUDGET --review annotation-review.json --output extraction-live.json
```

Credentials come from the process environment via `Settings.from_env()`; these
standalone development runners do not load a project's stored credentials or
discover `.env` files. One usage recorder covers the extraction run. Cost ceilings
are checked after responses and can be exceeded by the last response. Unknown
pricing/usage stops an active ceiling. Partial results are saved with missing
cases counted as failures. Quotation repair is disabled in this diagnostic run
to measure the initial extraction contract consistently.
Reports identify the extraction model and recipe, with extraction usage separate
from the unused answering and embedding roles. `--provider` and `--model` remain
aliases for the extraction options in this runner.

## Later paired answer evaluation

```powershell
python tests/benchmarks/graph_answers.py --live --answer-provider ANSWER_PROVIDER --answer-model ANSWER_MODEL --extraction-provider GRAPH_PROVIDER --extraction-model GRAPH_MODEL --embedding-provider EMBEDDING_PROVIDER --embedding-model EMBEDDING_MODEL --max-cost-usd BUDGET --context-tokens 2048 --output-tokens 1024 --output answers-live.json
```

This reuses the Phase 5 source/question fixture. It runs real indexing, graph
extraction, hybrid search, graph queries and generation, with one shared cost
recorder. `--live` is required for provider/model or spending options. Without
extraction overrides, extraction inherits the answering provider/model. A model-only
extraction override keeps the answering provider; a provider override requires an
explicit extraction model. The old `--provider` and `--model` flags are aliases for
the **answering** options in this runner. This standalone runner does not read saved
project or app model settings.

Both arms use the same answering model, question, answer prompt and maximum output size.
The paired runner's `--output-tokens` controls answering only. Graph extraction
uses provider-managed output and does not accept an operator output-token cap.
The input limit counts the actual rendered system text plus user question using
the same `cl100k_base` reference tokenizer. Provider message overhead and a
different model's tokenizer can differ; actual reported usage is saved separately.
Whole evidence records are packed without truncating quotations. Actual input
sizes may be below the identical cap, and omissions are reported.

The graph-assisted arm packs returned graph claims before hybrid passages. This
is an explicit benchmark policy, not a change to application ranking. Arm order
alternates across questions. Fixed query identities/terms isolate supplied-context
effects; autonomous model tool selection and identity clarification are not
evaluated. Review/replace the draft keyword-style questions and expected evidence
before a representative live run. Failed/interrupted source processing stops the
comparison before answering. Finished builds with rejected proposals may still be
compared: `complete_with_gaps` and its counters remain explicit, so rejection-heavy
models are not silently excluded from evaluation.

### Hold the comparison variable fixed

To compare extraction models, repeat preparation with the same answering model,
embedding model, corpus, questions, prompts, context/output limits and retrieval
policy; change only the extraction pair. Compare `comparison_controls`, the saved
model selections and extraction recipe before attributing differences to a model.
The recipe includes extraction/repair prompt hashes and window/contract settings.
Extraction includes quotation correction where enabled, with batches sized for
provider capacity and no fixed identity-turn or per-record call ceiling.

To compare answering models, reuse the first report's graph and exact supplied
evidence. Once live runs are authorized:

```powershell
python tests/benchmarks/graph_answers.py --live --answer-provider OTHER_PROVIDER --answer-model OTHER_ANSWER_MODEL --reuse-evidence answers-live.json --max-cost-usd BUDGET --context-tokens 2048 --output-tokens 1024 --output answers-other-model.json
```

Reuse makes only answering calls; it does not extract, embed, or retrieve again.
The evidence bundle contains the graph snapshot, extraction configuration, contexts,
rendered system inputs and questions. Its hash must remain valid, and the fixture,
prompt, retrieval policy and input/output limits must match. Extraction/embedding
flags cannot accompany reuse. The saved `evidence_sha256` should be identical in
the two reports. This freezes supplied context, not autonomous RAG tool planning.

Reports include `models`, `usage_by_role` (extraction, answering, embedding), raw
usage records, tokens, cost completeness and comparison controls. One spending
ceiling covers indexing/query embeddings, extraction/correction, and both answer
arms. It is checked after responses, so the final response can exceed it; missing
billable usage or unknown pricing stops further work. Role totals do not create
separate ceilings. `usage` describes this invocation; `budget_usage` includes any
earlier records on the same recorder when used from Python. During evidence reuse,
`graph_build.usage` is historical preparation usage; current extraction and embedding
usage is zero. Unknown estimated cost is `null`, distinct from zero offline spend.

No string matcher can establish whether arbitrary multilingual prose is supported
or preserves allegations. Copy the report's `review_template` into a separate
review file. Read each answer against its **actual supplied context** and gold
sources, count factual claims and citations, and record supported/correct counts,
expected facts covered and qualification preservation. An abstention has zero
claims; it does not earn perfect precision, and missing answerable facts lower
coverage. Keep the same expected-fact denominator for both arms. Review blinded
copies in randomized order where practical; retain an adjudication note for
ambiguous judgments.

```powershell
python tests/benchmarks/graph_answers.py --score-run answers-live.json --review answer-review.json --output answer-scores.json
```

Reviews are bound to answer/context hashes, including the recorded model, rendered
instructions and output limit. Pairs must use the same answering model and output
limit. Missing, edited, duplicate or unpaired
answers cannot silently receive a score. Citation-number bounds are automatic;
semantic citation correctness and unsupported conclusions require the review.
The resulting report exposes citation accuracy, fact coverage, unsupported claim
counts and qualification preservation per arm. Cost-stopped runs are incomplete
and cannot be scored as paired comparisons.

## Operator exploration measurements

`graph_exploration.py` measures local exploration separately from model quality.
It creates a disposable SQLite graph with real local text sources, parallel and
cyclic assertions, a twelve-step path, an isolated identity, all three relation
polarities, differing qualifications, and a quotation larger than 64 KiB.
Provider construction and socket connections are forbidden by the runner.

```powershell
python tests/benchmarks/graph_exploration.py --identities 1000 --output exploration-1000.json
python tests/benchmarks/graph_exploration.py --identities 1000 --trace-memory --output exploration-memory.json
python -m pytest tests/benchmarks/test_graph_exploration.py
```

The identity count controls the high-branching part of the fixture; path and
isolated identities are additional. An independent adjacency traversal checks
the assertion set. The runner checks pause/resume, every 30-row page, full-quote
search and evidence access, and exact page/diagram/all-matching exports. Its
companion tests compare source and direction scopes, counts, filters, complete
paths, disconnected searches, and retained model response limits.

Reports record initialization, batch and page timings, evidence/search/export
timings, Windows process memory snapshots, and optional Python allocation peaks.
The first batch is at most 30 adjacency entries so pause/resume can be exercised;
later batches use `--batch-size` (default 100). Process peak memory includes
imports and fixture setup. Python tracing excludes native allocations and adds
timing overhead. These observations are not capacity guarantees or pass/fail
latency thresholds. Browser measurements use the existing
[app harness](../app/README.md). See the [Phase 15 report](../../plan/phase15-validation.md)
for measured conditions and limitations.

For repeated discovery/export stage comparisons on both the 3,159-assertion
versioned fixture and the separate 5,005-assertion star fixture:

```powershell
python tests/benchmarks/graph_exploration.py --stages --repeats 3 --output graph-performance.json
```

This uses the same fixture/oracle and records inclusive stage times and call counts,
batch latency, fresh-query versus subsequent page work, each export scope, separate
JSON/CYPHERL serialization, and process memory. Nested stage times must not be added.
Runs are sequential with fresh query caches but shared process/OS caches. Compare
unchanged configurations on the same machine without overlapping validation jobs;
keep cProfile/tracemalloc attribution separate from these lightweight timer runs.
There are no wall-clock pass/fail thresholds. Deterministic operation-count and
evidence/invalidation regressions live in `tests/graph/test_graph_performance.py`
and `tests/graph/test_mapping_cache.py`. Browser timings remain a separate app check.
The [Phase 16 report](../../plan/phase16-validation.md) records matched before/after
measurements, integrity coverage, and remaining scheduling/memory limits.

## Existing validation

`graph_passages.py` retains the original offline source-support proxy with
synthetic embeddings. Its assisted arm has a larger evidence budget; it is not
the new paired live-answer comparison. `graph_adjacency.py` measures adjacency
only. Neither result establishes live answer quality. Current-schema/refusal,
empty graph, concurrent jobs, evidence inspection and export checks live in
`tests/graph`, `tests/app` and `tests/cli`. Explicit supported-schema migration is
also covered, including disabled graphs and inspection without provider calls.
Model-role settings, app/CLI parity, compatible checkpoints and historical model
attribution reuse the Phase 9–10 tests; Phase 11 adds the remaining combinations
and benchmark checks. See [Phase 11 validation](../../plan/phase11-validation.md).
