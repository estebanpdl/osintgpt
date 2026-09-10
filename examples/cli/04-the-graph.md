# The sourced graph

Graph construction costs one generation call per readable document and never
runs as an indexing or search side effect. Enable it, then build explicitly:

```bash
osintgpt config set graph_enabled true
osintgpt graph build
```

The build command warns before provider calls, prints `position/total ref`
progress, and reports the run's provider usage. Use `--incremental` after the
first build to read only new documents, or `--rebuild` to replace claims from
documents that changed. The two modes cannot be combined.

```bash
osintgpt graph build --incremental --json
osintgpt graph build --rebuild --project CASE_SLUG
```

Traversal is keyless once a graph exists. `neighbors` shows sourced claims
touching an entity; a depth above one walks farther out. `path` finds the
shortest chain within the requested depth. Every row includes its source
document and exact evidence text.

```bash
osintgpt graph neighbors Neral-7 --depth 2 --limit 20
osintgpt graph path Neral-7 "Station Kestrel" --max-depth 4
```

These JSON commands were rerun against a built scratch graph with all provider
keys unset:

```bash
osintgpt graph neighbors Neral-7 --project walkthrough --json
osintgpt graph path Neral-7 "Station Kestrel" \
  --project walkthrough --json
```

```json
{"project": "walkthrough", "entity": "Neral-7", "depth": 1, "results": [{"source": "Neral-7", "target": "Relay Delta", "relation": "operated", "ref": "dispatches.txt", "evidence": "Neral-7 operated Relay Delta.", "depth": 1}]}
{"project": "walkthrough", "source": "Neral-7", "target": "Station Kestrel", "max_depth": 4, "length": 2, "documents": ["dispatches.txt", "ledger.txt"], "edges": [{"source": "Neral-7", "target": "Relay Delta", "relation": "operated", "ref": "dispatches.txt", "evidence": "Neral-7 operated Relay Delta.", "depth": 1}, {"source": "Relay Delta", "target": "Station Kestrel", "relation": "reported to", "ref": "ledger.txt", "evidence": "Relay Delta reported to Station Kestrel.", "depth": 2}]}
```

Verification reopens source documents and checks that every evidence quotation
is still present. `--strict` makes any finding a non-zero exit. Export keeps
the same sourced assertions for other tools:

```bash
osintgpt graph verify --strict
osintgpt graph export graph.cypherl
osintgpt graph export graph.json
```

CYPHERL is ready for line-oriented import into Memgraph or Neo4j. An exported
relationship is a sourced assertion from the corpus, not proof that the
assertion is true.
