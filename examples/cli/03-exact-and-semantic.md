# Exact and semantic retrieval

`search` returns passages without generating an answer. It runs the two
passage legs of the tri-retrieval engine; the third leg, the relationship
graph, is in [`04-the-graph.md`](04-the-graph.md).

| Leg | Finds | Cost per search |
|---|---|---|
| Semantic | Passages close in meaning, even with no shared words | One embedding call |
| Lexical (exact) | Passages containing a literal term: handles, URLs, hashes, ids | None; reads the store |
| Fused | Both lists combined by reciprocal rank fusion | As for the legs used |

This uses the project indexed in [`01`](01-first-case.md) and
[`02`](02-structured-data.md). Each result prints its passage below a header
line; only the headers are shown here, and `…/case/` stands for
`<repo>/examples/data/generated/case/`.

## Semantic

```bash
osintgpt search "who financed the relay upgrade" --top-k 4
```

```text
1. 0.453  …/case/material/prose/relay-ledger.md  [semantic #1]
2. 0.441  …/case/material/reports/field-note-kestrel.md  [semantic #2]
3. 0.385  …/case/material/intake/posts.csv#p4  [semantic #3]
4. 0.376  …/case/material/records/events.csv#EV-001  [semantic #4]
Usage: 1 call, 5 tokens, ~$0
```

The field note never says "financed"; it says Kestrel "underwrote the
refit", and semantic search still ranks it second. The relay ledger ranks
first while saying nothing about money: a score measures closeness to the
query wording, not relevance or truth. Read the passages.

## Exact

Repeat `--exact` for each literal term. Matching is a substring match over
indexed text and the author field, so it needs no API key and makes no
provider call. The query argument is optional when only exact terms are
given:

```bash
osintgpt search --exact KS-1180 --exact PO-7731
```

```text
1. 0.500  …/case/material/intake/export.json#t3  [lexical #1]
2. 0.500  …/case/material/intake/posts.csv#p4  [lexical #2]
3. 0.500  …/case/material/reports/field-note-kestrel.md  [lexical #3]
4. 0.500  …/case/material/reports/procurement-memo.md  [lexical #4]
```

The score is the share of terms a passage contains: each of these holds one
of the two. Terms shorter than two characters are ignored. To sweep a whole
watchlist and export the hits, see
[`scripts/watchlist_sweep.py`](../scripts/watchlist_sweep.py).

## Fused

Add `--semantic` and query text to run both legs. Fusion ranks by position,
not by score, so a passage found by both legs rises above one found by
either alone:

```bash
osintgpt search "who financed the relay upgrade" --exact KS-1180 --semantic --top-k 4
```

```text
1. 0.032  …/case/material/intake/posts.csv#p4  [lexical #1, semantic #3]
2. 0.032  …/case/material/reports/field-note-kestrel.md  [semantic #2, lexical #2]
3. 0.016  …/case/material/prose/relay-ledger.md  [semantic #1]
4. 0.016  …/case/material/records/events.csv#EV-001  [semantic #4]
Usage: 1 call, 5 tokens, ~$0
```

The two passages naming the account now lead, and the ledger drops to
third. Fused scores are rank-fusion values, not similarities.

`--derive-terms` asks the generation model to pick the literal terms, one
extra generation call, then fuses as above:

```bash
osintgpt search "Which reserve paid for purchase order PO-7731?" --derive-terms --top-k 3
```

```text
1. 0.033  …/case/material/intake/export.json#t3  [semantic #1, lexical #1]
2. 0.032  …/case/material/reports/procurement-memo.md  [semantic #2, lexical #2]
3. 0.016  …/case/material/intake/posts.csv#p4  [semantic #3]
Usage: 2 calls, 359 tokens, ~$0.000117
```

`library/retrieval_legs.py --derive-terms` prints the chosen terms; here
`gpt-5.6-luna` picked `PO-7731`. Term choice is model-driven, with no
language-specific stop-word list. For a question naming no identifier, such
as "Which purchase order covered the replacement antennas?", the model may
return no terms, and the search runs semantic only.

## JSON for scripts

`--json` returns every result with its `legs` and `ranks`, plus a usage
record. `--full` prints complete passages instead of trimming them.

```bash
osintgpt search --exact KS-1180 --json
```

Lexical-only JSON reports `"calls": 0` in its usage record.

## From Python

[`library/retrieval_legs.py`](../library/retrieval_legs.py) runs each leg on
its own (`search_project`, `lexical_search`, `derive_search_terms`) and fuses
them with `reciprocal_rank_fusion`, including a per-leg weight.

Next: [`07-asking-and-conversations.md`](07-asking-and-conversations.md)
shows how `ask` chooses among these legs.
