# osintgpt examples

Run every example from the repository root. Start by generating the invented
case corpus: prose (Markdown, text, HTML) in several languages and scripts,
CSV/JSON/JSONL records, an evaluation question set, and a watchlist. It
writes only under `examples/data/generated/`.

```bash
python examples/data/make_corpus.py
```

| Path under `generated/case/` | Contents | Used in |
|---|---|---|
| `material/prose/`, `material/reports/` | Prose that needs no mapping | CLI 01, 03, 07 |
| `material/records/`, `material/intake/` | CSV, JSONL, nested JSON, with junk rows to filter | CLI 02 |
| `questions.toml` | Eight questions with expected documents and terms | CLI 06, `evaluate_retrieval.py` |
| `watchlist.txt` | Literal identifiers: ids, a handle, a URL, a hash | `watchlist_sweep.py` |

**Cost** in the tables below: *keyless* makes no provider call; *embedding*
and *generation* call the configured providers. Every command and program
that calls a paid provider prints its usage and estimated cost.

## How retrieval works

osintgpt answers from three retrieval legs over one project:

- **Semantic:** passages close in meaning to the query (embedding search).
- **Lexical:** passages containing literal terms such as handles, URLs, hashes,
  and ids (no provider call).
- **Graph:** sourced relationship claims between entities, built explicitly.

`search` runs the passage legs and fuses them by reciprocal rank. `ask` gives
the model all three legs as tools and lets it choose; `ask --static` retrieves
semantic passages once and answers from them.

## CLI walkthroughs — `cli/`

Follow them in order; each continues the project from the one before.

| Step | File | Covers | Cost |
|---|---|---|---|
| 1 | [`01-first-case.md`](cli/01-first-case.md) | Project, sources, credentials, ceiling, `doctor`, index, first answer | embedding, generation |
| 2 | [`02-structured-data.md`](cli/02-structured-data.md) | Field roles, `--dry-run`, `min_chars`, nested JSON | keyless to map; embedding to index |
| 3 | [`03-exact-and-semantic.md`](cli/03-exact-and-semantic.md) | Semantic, exact, fused, and derived-term search | keyless to embedding |
| 4 | [`04-the-graph.md`](cli/04-the-graph.md) | Build, inspect, verify, and export the relationship graph | generation |
| 5 | [`05-running-local.md`](cli/05-running-local.md) | sentence-transformers and Ollama, model switches | local |
| 6 | [`06-evaluating.md`](cli/06-evaluating.md) | Hit rate, MRR, recall, reading misses | embedding |
| 7 | [`07-asking-and-conversations.md`](cli/07-asking-and-conversations.md) | Agentic and static answers, traces, threads, JSON | embedding, generation |
| 8 | [`08-converting-documents.md`](cli/08-converting-documents.md) | `convert`, folders, PDFs and `--vision`, formats | keyless; `--vision` generation |
| 9 | [`09-project-operations.md`](cli/09-project-operations.md) | Projects, settings, credentials, sources, remote stores | keyless |

## Library programs — `library/`

Each program is standalone, accepts `--help`, and takes a project directory
(`~/.osintgpt/projects/<slug>` for CLI-created projects). Configuration is
explicit: credentials come from the process environment, or from a file
passed with `--env-file` (see [`config/.env.template`](config/.env.template)).
Credentials stored with `osintgpt auth set` are read by the CLI and app only.

| Topic | File | What it shows | Cost |
|---|---|---|---|
| Ingest | [`index_a_folder.py`](library/index_a_folder.py) | Create or load a project, register a folder, index with progress and a cost ceiling | embedding |
| Ingest | [`map_structured_records.py`](library/map_structured_records.py) | `describe_fields`, `preview_mapping`, `load_documents`, register | keyless |
| Retrieve | [`retrieval_legs.py`](library/retrieval_legs.py) | Each leg on its own, derived terms, weighted rank fusion | embedding; generation with `--derive-terms` |
| Retrieve | [`across_projects.py`](library/across_projects.py) | One ranked list over compatible projects | embedding |
| Answer | [`answer_with_citations.py`](library/answer_with_citations.py) | Static and agentic answers, live tool calls, trace, follow-ups | embedding, generation |
| Answer | [`conversation_thread.py`](library/conversation_thread.py) | Recorded threads with carried context | embedding, generation |
| Graph | [`build_and_query_graph.py`](library/build_and_query_graph.py) | Build and traverse sourced relationships | generation |
| Measure | [`evaluate_retrieval.py`](library/evaluate_retrieval.py) | Score every retrieval method against a question set | embedding |
| Providers | [`custom_provider.py`](library/custom_provider.py) | Registered or any OpenAI-compatible backend, locality audit, retry policy | embedding, generation |
| Providers | [`migrating_from_0_1.py`](library/migrating_from_0_1.py) | Replace the pre-1.0 compatibility constructors | none until used |

Example run against a CLI project:

```text
python examples/library/retrieval_legs.py ~/.osintgpt/projects/synthetic-relay-case \
    "who financed the relay upgrade" --term KS-1180 --term PO-7731
```

## Operator scripts — `scripts/`

Ready-to-use tools built on the library.

| File | What it does | Cost |
|---|---|---|
| [`dry_run.py`](scripts/dry_run.py) | What a folder would index and cost; which structured files still need a mapping (exits 1 while any do) | keyless |
| [`inspect_chunks.py`](scripts/inspect_chunks.py) | The actual chunk boundaries and size statistics for one document | keyless |
| [`watchlist_sweep.py`](scripts/watchlist_sweep.py) | Every indexed passage mentioning any watchlist term, with context, to screen or CSV/JSONL | keyless |
| [`batch_ask.py`](scripts/batch_ask.py) | Answer a question list into JSONL with sources, trace, and usage; resumable, under one cost ceiling | embedding, generation |

```text
python examples/scripts/watchlist_sweep.py ~/.osintgpt/projects/synthetic-relay-case \
    examples/data/generated/case/watchlist.txt --out hits.csv
python examples/scripts/batch_ask.py ~/.osintgpt/projects/synthetic-relay-case \
    examples/data/generated/case/questions.toml --out answers.jsonl --ceiling 0.50
```
