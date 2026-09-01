# First case

Run from the repository root. Generate the invented corpus, create an isolated
project, select it, and register only the prose folder first.

```bash
python examples/data/make_corpus.py
osintgpt project create "Synthetic relay case"
osintgpt project use synthetic-relay-case
osintgpt add examples/data/generated/case/material/prose
python examples/scripts/dry_run.py examples/data/generated/case/material/prose
```

The generator and dry run above produced this output on Windows:

```text
Wrote 4 documents and 3 questions to examples\data\generated\case
examples\data\generated\case\material\prose
2 files, 2 documents, 2 chunks, 158 tokens, ~$0.0000 to embed

would index
  relay-ledger.md                       1 docs       1 chunks         84 tokens
  dispatches.txt                        1 docs       1 chunks         74 tokens
```

Read one document's chunk statistics before embedding it:

```bash
python examples/scripts/inspect_chunks.py \
  examples/data/generated/case/material/prose/relay-ledger.md --stats
```

That command produced:

```text
examples\data\generated\case\material\prose\relay-ledger.md  —  1 documents, 1 chunks

documents      1
chunks         1
size           min 317  median 317  max 317  cap 1500
at the cap     0 (>93% of ceiling)
under 200      0 (0%) — short chunks carry little for a vector to match on
context        1 of 1 (100%) carry a heading or section path
```

Indexing needs an embedding provider. Store its credential once, or export the
variable yourself — the environment is read first, then the stored value:

```bash
osintgpt auth set openai
osintgpt auth list
```

`auth set` prompts rather than taking the key as an argument, which keeps it
out of the shell history. The fully local setup, needing no credential at all,
is in [`05-running-local.md`](05-running-local.md).

```bash
osintgpt index
osintgpt ask "Which node acknowledged sequence LX-204?"
```

`ask` prints its answer followed by the source documents. A bracketed marker
such as `[1]` refers to the first retrieved passage; open the listed source and
check the claim against it rather than treating the generated answer as the
source of record.
