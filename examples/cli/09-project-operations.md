# Project operations

Day-to-day commands for managing projects, settings, credentials, sources,
and storage. Every project is isolated: its own settings, sources, store,
conversations, and graph.

## Projects

```bash
osintgpt project list
osintgpt project show
osintgpt project use synthetic-relay-case
osintgpt project create "Field case" --path D:/cases/field-case
osintgpt project delete field-case --yes
```

Projects live in `~/.osintgpt/projects/<slug>/` unless created with
`--path`, for example beside the material or on an encrypted or removable
volume. Material inside the project directory is cited by relative path, so
the whole folder can be moved or handed over intact. `delete` requires
`--yes` and removes the entire project directory, including its SQLite
store, conversations, and **any material stored inside it**. Material
registered from elsewhere and remote Qdrant or Postgres collections are left
untouched. `--home PATH` before any command uses a separate osintgpt
home with its own projects and credentials.

## Settings

Settings resolve in this order: environment or library arguments, then the
project, then user defaults (`--user`), then built-in defaults.

```bash
osintgpt config get
osintgpt config get generation_model
osintgpt config set generation_model gpt-5.6-luna
osintgpt config set cost_ceiling_usd 1.00 --user
osintgpt config set cost_ceiling_usd unset
```

`config get` with no key lists every project setting and which secrets are
set. `--user` reads or writes defaults inherited by every project. `unset`
clears an optional value (ceilings, rate limits) so it is inherited again.

| Setting | Use |
|---|---|
| `embedding_provider`, `embedding_model` | Index and query embeddings; changing either rebuilds the index on the next `index` |
| `generation_provider`, `generation_model` | Answers, derived terms, follow-ups, graph extraction |
| `ingestion_model` | Scanned-page transcription; falls back to `generation_model` |
| `cost_ceiling_usd` | Stop any single command above this estimate |
| `conversation_window` | Earlier turns carried by `ask --conversation` |
| `suggest_followups` | Turn the "Ask next" call on or off |
| `graph_enabled` | Graph extraction and the answering model's graph tool |
| `storage_backend` | `sqlite` (default), `qdrant`, or `postgres` |
| `<provider>_embedding_rpm`, `_tpm`, `_batch_tokens`, `_quota_scope` | Embedding pacing for `openai`, `gemini`, `voyage` |

The README's [embedding rate limit section](../../README.md#where-projects-live)
covers pacing, retries, and resuming interrupted runs.

## Credentials

```bash
osintgpt auth set openai
osintgpt auth list
osintgpt auth path
osintgpt auth remove openai
```

Providers: `openai`, `gemini`, `voyage`, `anthropic`, `qdrant`,
`postgres`. An exported environment variable (`OPENAI_API_KEY`, ...) wins
over the stored value; `auth list` shows which source each one comes from.
Library code reads only what it is given: `Settings.from_env()` or an explicit
`.env` file, see [`config/.env.template`](../config/.env.template).

## Sources

```bash
osintgpt sources
osintgpt remove examples/data/generated/case/material/intake/export.json
osintgpt index
```

```text
10 unchanged, 2 removed
```

`remove` unregisters a path and leaves the files on disk. The next `index`
deletes its chunks from the store (here, the two records of `export.json`).

```bash
osintgpt index --force
osintgpt index --json
```

`--force` re-embeds every file and discards saved checkpoints. Without it,
an interrupted run resumes from saved batches.

## Remote storage

SQLite needs no setup. For Qdrant or Postgres (pgvector), set the backend
before the first `index`, then supply the connection:

```bash
osintgpt config set storage_backend qdrant
osintgpt auth set qdrant
osintgpt doctor --check-providers
```

Qdrant reads `QDRANT_URL` (or `QDRANT_HOST` and `QDRANT_PORT`) from the
environment and the API key from `auth set qdrant`. Postgres reads its
connection string from `auth set postgres` or `POSTGRES_DSN`. Each project
uses its slug as the collection name, so projects stay separate on a shared
server. Back up remote stores separately from the project directory.

## Health check

```bash
osintgpt doctor
osintgpt doctor --strict --json
```

`doctor` reports providers, credentials, store contents, stored embedding
models, source coverage, and whether content leaves the machine. `--strict`
exits non-zero on any issue, which suits scheduled jobs.
