# Evaluating retrieval

An evaluation set pairs questions with the documents known to answer them,
so a retrieval change can be measured instead of argued about. The generator
writes one with eight questions to `examples/data/generated/case/questions.toml`:

```toml
[[question]]
text = "Who paid for the hardware upgrade at the northern relay?"
expected = [
    "D:/i/tools/osintgpt/examples/data/generated/case/material/reports/field-note-kestrel.md",
]
terms = [
    "Kestrel",
]
note = "paraphrase: the note says underwrote and refit"
```

`expected` holds refs exactly as the store records them. Material outside
the project is stored under its absolute path, so the generator resolves
them for your checkout; rerun it if the repository moves. `terms` is used
only by hybrid evaluation, and `note` records what the question probes.

## Score the project

After indexing as in [`01`](01-first-case.md) and
[`02`](02-structured-data.md):

```bash
osintgpt evaluate examples/data/generated/case/questions.toml --top-k 3
```

```text
┌─────────────────┬────────────────────────┐
│ Retrieval       │ semantic               │
│ Embedding model │ text-embedding-3-small │
│ Top k           │ 3                      │
│ Scored          │ 8                      │
│ Found           │ 8                      │
│ Hit rate        │ 100%                   │
│ MRR             │ 0.875                  │
│ Recall          │ 100%                   │
└─────────────────┴────────────────────────┘

Misses
None

Unscorable
None
Usage: 8 calls, 107 tokens, ~$0.000002
```

- **Hit rate:** questions with at least one expected document in the top k.
- **MRR:** mean of 1/rank of the first expected document; 1.0 means it was
  always first.
- **Recall:** share of expected documents retrieved within the top k.

Evaluation embeds each question with the project's embedding model, so it
needs the same credential as `index`.

## Read the misses

Tighten `--top-k` to 1 and measure hybrid retrieval, which adds each
question's `terms` as an exact leg:

```bash
osintgpt evaluate examples/data/generated/case/questions.toml --top-k 1 --retrieval hybrid
```

```text
│ Retrieval       │ hybrid                 │
│ Top k           │ 1                      │
│ Found           │ 6                      │
│ MRR             │ 0.750                  │
...
Misses
Did the Koru-5 operator deny relaying packet AR-12?
  Expected:
D:/i/tools/osintgpt/examples/data/generated/case/material/reports/interview-transcript.txt
  Retrieved: D:/i/tools/osintgpt/examples/data/generated/case/material/records/events.csv
¿Por qué se cerró la ruta MX-3?
  Expected: D:/i/tools/osintgpt/examples/data/generated/case/material/reports/nota-operativa.md
  Retrieved: D:/i/tools/osintgpt/examples/data/generated/case/material/records/messages.jsonl
```

Both misses are short records that repeat the question's identifier
(`AR-12`, `MX-3`), so they outrank the document that actually answers. Exact
terms cannot fix that, because the records contain the same terms. A miss
like this is a reason to look at the records and the question, not only at
the aggregate.

Hybrid evaluation uses only the terms recorded in the set; it never asks a
generation model for terms, so runs are reproducible.

## Limits

These scores measure whether retrieval finds the expected documents for
these questions. They do not measure whether a generated answer is correct.
Eight questions over a dozen documents say little; build a set of real
questions for your own corpus, and save new ones with `save_questions` from
Python. [`library/evaluate_retrieval.py`](../library/evaluate_retrieval.py)
runs every method in one pass.
