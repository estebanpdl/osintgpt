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

Local embeddings are the one opt-in extra because they bring
sentence-transformers and torch:

```bash
pip install osintgpt[local]
```

The current release is 0.3.0. See
[osintgpt on PyPI](https://pypi.org/project/osintgpt/) for package metadata
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

An unindexed project is also a valid state: `ask` explains that nothing was
retrieved and does not call the generation model.

## How it searches

`ask` can draw from semantic similarity, exact text matches, and sourced graph
relationships. By default the model surveys the project, chooses which tools
to use, reads the relevant material, and then answers. Use `--trace` to see
that work, or `--static` when you want the earlier one-pass retrieval behavior.
Graph edges retain their source document and quoted evidence;
`osintgpt graph verify` checks those quotes, while
`osintgpt graph export graph.cypherl` writes CYPHERL for Memgraph or Neo4j
(`.json` is also supported).

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

## What it reads

The 29 readable extensions cover PDF, Word, Excel and CSV, JSON and JSONL,
Markdown and plain text, HTML and XML, and common image formats when the
embedding model supports them. A fallback converter handles formats such as
PowerPoint and EPUB. Structured files need a content-field mapping when they
are registered; scanned PDF pages need a transcriber to recover text.

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

## Responsible use

`osintgpt` analyzes material supplied by its operator. Use it only with data
you may lawfully process, protect personal and sensitive information, verify
answers against their cited passages, and understand any costs or data
handling terms of the providers you configure. The maintainers are not liable
for misuse or third-party service charges.

## License

Licensed under the [Apache License 2.0](LICENSE).
