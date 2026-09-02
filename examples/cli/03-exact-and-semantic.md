# Exact and semantic retrieval

`search` returns passages without asking a generation model to write an
answer. With no retrieval flags it keeps the semantic-only default:

```bash
osintgpt search "acknowledged transmission" --top-k 5
```

Use repeatable `--exact` flags for literal identifiers. This path reads the
indexed text and needs no API key or embedding provider. The query argument is
optional when exact matching is the only leg:

```bash
osintgpt search --exact LX-204 --exact QN-88 --full
```

Add `--semantic` to fuse semantic and exact ranks. `--derive-terms` instead
asks the generation model to choose literal terms, costing one generation
call, and then fuses them with semantic results:

```bash
osintgpt search "Which node acknowledged the sequence?" \
  --exact LX-204 --semantic
osintgpt search "Which node acknowledged the sequence?" --derive-terms
```

Each result names the leg or legs that found it. This keyless JSON output was
rerun against an indexed scratch project with all provider keys unset:

```bash
osintgpt search --exact LX-204 --project walkthrough --json
```

```json
{"results": [{"rank": 1, "score": 1.0, "citation": "dispatches.txt › Dispatch 4", "ref": "dispatches.txt", "text": "Operator Neral-7 acknowledged transmission LX-204.", "path": "Dispatch 4", "timestamp": "", "author": "", "legs": ["lexical"], "ranks": {"lexical": 1}}, {"rank": 2, "score": 1.0, "citation": "ledger.txt › Ledger", "ref": "ledger.txt", "text": "Account LX-204 was assigned to relay Delta.", "path": "Ledger", "timestamp": "", "author": "", "legs": ["lexical"], "ranks": {"lexical": 2}}], "usage": {"calls": 0, "billable_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": null, "complete": true, "unpriced_calls": 0, "uncounted_calls": 0, "by_model": {}, "ceiling_usd": null}}
```

`ask` remains model-directed. It may use exact, semantic, and graph tools and
can show those decisions with `--trace`; `--static` deliberately uses one
semantic retrieval pass.

```bash
osintgpt ask "Where does identifier LX-204 appear?" --trace
osintgpt ask "Where does identifier LX-204 appear?" --static --passages 5
```
