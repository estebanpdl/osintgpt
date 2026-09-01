# The sourced graph

Graph construction costs one generation call per readable document and never
runs as an indexing side effect. Enable it for the selected project and use
`project show` to copy the project path:

```bash
osintgpt config set graph_enabled true
osintgpt project show
```

There is currently no CLI graph-build command. Build and inspect it through
the library example, supplying a generation model through project settings,
the environment, or the explicit option:

```bash
python examples/library/build_and_query_graph.py PROJECT_PATH Neral-7 \
  --generation-provider PROVIDER --generation-model MODEL_NAME
```

Each relationship stores the document and exact evidence text that asserted
it. Verification reopens those documents and checks that every quotation is
still present; `--strict` also makes any failure a non-zero exit.

```bash
osintgpt graph verify --strict
osintgpt graph export graph.cypherl
osintgpt graph export graph.json
```

CYPHERL is ready for line-oriented import into Memgraph or Neo4j. JSON keeps
the entity and edge records for other tools. An exported relationship is a
sourced assertion from the corpus, not proof that the assertion is true.
