# Exact and semantic retrieval

`search` returns raw semantic matches without asking a generation model to
write an answer. It is the direct choice when you want passages ranked by
meaning and intend to read them yourself.

```bash
osintgpt search "acknowledged transmission" --top-k 5
```

`ask` is model-directed. The model may survey documents, use semantic search
for concepts, use exact search for literal identifiers, query a built graph,
and read a source before answering. `--trace` prints those choices and result
counts so you can see whether the retrieval path fits the question.

```bash
osintgpt ask "Where does identifier LX-204 appear?" --trace
osintgpt ask "Where does identifier LX-204 appear?" --static --passages 5
```

The second command deliberately uses the one-pass semantic path. For direct
library access to both semantic and exact legs without generating an answer,
run:

```bash
python examples/library/search_without_answering.py \
  PROJECT_PATH "acknowledged transmission" --term LX-204 --top-k 5
```
