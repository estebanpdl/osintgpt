# osintgpt examples

Run these examples from the repository root. Start by generating the small,
invented corpus; it contains Markdown, text, CSV, and JSONL records in several
scripts and writes nothing outside `examples/data/generated/`.

```bash
python examples/data/make_corpus.py
```

Choose the half that matches how you use osintgpt:

- [`cli/`](cli/) is for an operator working at a shell. The walkthroughs move
  from a first project through structured data, retrieval, graphs, local
  models, and evaluation.
- [`library/`](library/) is for a developer importing osintgpt. Every file is
  standalone, accepts `--help`, and keeps configuration explicit.

## CLI walkthroughs

| File | What it shows |
|---|---|
| [`01-first-case.md`](cli/01-first-case.md) | Create, inspect, index, and ask |
| [`02-structured-data.md`](cli/02-structured-data.md) | Assign roles to record fields |
| [`03-exact-and-semantic.md`](cli/03-exact-and-semantic.md) | Raw semantic hits and model-directed retrieval |
| [`04-the-graph.md`](cli/04-the-graph.md) | Build through the library, then verify and export |
| [`05-running-local.md`](cli/05-running-local.md) | Sentence-transformers and Ollama |
| [`06-evaluating.md`](cli/06-evaluating.md) | Measure retrieval against known documents |

## Library programs

| File | What it does |
|---|---|
| [`index_a_folder.py`](library/index_a_folder.py) | Creates or loads one project and indexes a folder |
| [`search_without_answering.py`](library/search_without_answering.py) | Compares semantic results with rank fusion |
| [`answer_with_citations.py`](library/answer_with_citations.py) | Contrasts static and agentic answers |
| [`custom_provider.py`](library/custom_provider.py) | Builds providers without a project |
| [`across_projects.py`](library/across_projects.py) | Searches compatible isolated projects together |
| [`build_and_query_graph.py`](library/build_and_query_graph.py) | Builds and traverses sourced relationships |
| [`evaluate_retrieval.py`](library/evaluate_retrieval.py) | Scores known-answer retrieval questions |
| [`migrating_from_0_1.py`](library/migrating_from_0_1.py) | Replaces the compatibility constructors |

The programs read credentials already present in the environment with
`Settings.from_env()`. [`config/.env.template`](config/.env.template) is only
for library code that explicitly passes its path to `Settings.from_env`; the
CLI and browser app do not discover that file.

## Inspection scripts

[`scripts/dry_run.py`](scripts/dry_run.py) reports what a folder would cost
and which structured files still need mappings. [`scripts/inspect_chunks.py`](scripts/inspect_chunks.py)
shows the actual chunk boundaries for one document. Both are keyless and make
no provider calls.
