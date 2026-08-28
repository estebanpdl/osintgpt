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

## Responsible use

`osintgpt` analyzes material supplied by its operator. Use it only with data
you may lawfully process, protect personal and sensitive information, verify
answers against their cited passages, and understand any costs or data
handling terms of the providers you configure. The maintainers are not liable
for misuse or third-party service charges.

## License

Licensed under the [Apache License 2.0](LICENSE).
