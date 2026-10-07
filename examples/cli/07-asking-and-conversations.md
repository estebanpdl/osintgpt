# Asking and conversations

`ask` answers from the project and lists its sources. It has two paths:

| Path | What happens | Calls |
|---|---|---|
| Default (agentic) | The model is given retrieval tools (semantic, exact, snowball, document listing, source fetch, and graph when enabled) and chooses which to run, round by round | One generation call per round, plus embeddings for semantic searches |
| `--static` | One semantic retrieval of `--passages` passages, then one answer | One embedding call and one answer call |

Both paths finish with suggested follow-up questions, which cost one more
generation call (turn off with `config set suggest_followups false`), and a
usage line. This uses the project from
[`01`](01-first-case.md) and [`02`](02-structured-data.md), with
`generation_model` set to `gpt-5.6-luna`.

## Read the trace

```bash
osintgpt ask "Who paid for the Neral-7 refit, and through which account?" --trace --new-conversation
```

```text
Station Kestrel paid for—or underwrote—the Neral-7 refit. The payment went through account
**KS-1180**, listed in the ledger as the **“maintenance reserve”** rather than capital spending.
((D:/i/tools/osintgpt/examples/data/generated/case/material/reports/field-note-kestrel.md);
(D:/i/tools/osintgpt/examples/data/generated/case/material/intake/posts.csv))

Sources
• D:/i/tools/osintgpt/examples/data/generated/case/material/intake/posts.csv
• D:/i/tools/osintgpt/examples/data/generated/case/material/prose/relay-ledger.md
• D:/i/tools/osintgpt/examples/data/generated/case/material/records/events.csv
• D:/i/tools/osintgpt/examples/data/generated/case/material/reports/field-note-kestrel.md
...

Ask next
1. Which nodes had the shared key that could write to the event log in cycle four?
2. What does the corpus record about the conflicting claims that Koru-5 did not relay packet AR-12 while Neral-7 acknowledged receiving AR-12?
3. Who opened route MX-3 after the route had been closed, and what does the interview transcript say about its reopening?

Trace
round 1
  exact_search(terms=['Neral-7'], mode='refs', limit=50) — 6 documents, 0.00s
  semantic_search(query='Who paid for the Neral-7 refit and which account the payment came through', limit=20) — 17 passages, 0.95s
  list_documents(limit=100) — 11 documents, 0.00s
Usage: 4 calls, 13,773 tokens, ~$0.003342
```

In one round the model ran an exact search for the identifier, a semantic
search for the meaning, and listed the documents. `Sources` lists every
document the tools returned, not only the ones the answer cites. The trace
shows each call's arguments and result count, so you can see what the model
actually looked at.

## Check negative and conflicting answers

The agentic path depends on the generation model choosing useful tool calls.
Ask a question whose answer sits in one document:

```bash
osintgpt ask "Who edited the Sorin-3 status page during the outage?" --trace
```

```text
The status page was edited by **@vela_ops**. It was edited twice during the outage, and the
second edit removed the mention of the modem.
Source: `D:/i/tools/osintgpt/examples/data/generated/case/material/reports/incident-amber.html`
...
Trace
round 1
  exact_search(terms=['Sorin-3'], mode='refs', limit=50) — 5 documents, 0.00s
  list_documents(limit=100) — 11 documents, 0.00s
round 2
  exact_search(terms=['Sorin-3'], mode='snippets', limit=20) — 5 passages, 0.00s
Usage: 4 calls, 9,639 tokens, ~$0.002329
```

With `gpt-4.1-mini`, the same question searched for the phrase
`'Sorin-3 status page'` as one exact term, found nothing, and answered that
the corpus does not contain the information. Before accepting "not found",
read the trace: an empty exact search on a long phrase proves nothing.
Rephrase, search for an identifier yourself, try another model, or use the
static path, which always retrieves semantic passages:

```bash
osintgpt ask "Who edited the Sorin-3 status page during the outage?" --static --passages 3
```

```text
The status page was edited by **@vela_ops**. [1]

Sources
• D:/i/tools/osintgpt/examples/data/generated/case/material/reports/incident-amber.html
• D:/i/tools/osintgpt/examples/data/generated/case/material/intake/export.json#t1
• D:/i/tools/osintgpt/examples/data/generated/case/material/reports/nota-operativa.md
...
Usage: 3 calls, 1,578 tokens, ~$0.000573
```

Conflicting sources should come back as a conflict, not a verdict:

```bash
osintgpt ask "Did the Koru-5 operator deny relaying packet AR-12?" --trace
```

```text
Yes. In the interview, the Koru-5 operator denied relaying AR-12, stating: “Koru-5 did not relay
AR-12” and that the node was under maintenance. However, the event log records that Koru-5
relayed AR-12 to Neral-7, so the accounts conflict. ( `.../reports/interview-transcript.txt`;
`.../records/events.csv#EV-001` )
...
Trace
round 1
  exact_search(terms=['Koru-5', 'AR-12'], mode='refs', limit=50) — 3 documents, 0.00s
  semantic_search(query='whether the Koru-5 operator denied relaying packet AR-12', limit=10) — 10 passages, 1.48s
Usage: 4 calls, 10,373 tokens, ~$0.00274
```

Compare models on your own questions before relying on one.

## Continue a conversation

`--new-conversation` recorded the first answer in a thread. List threads and
continue one with `--conversation ID`; the last `conversation_window` turns
(default 3) are sent with the new question, so "that account" resolves:

```bash
osintgpt conversations list --json
osintgpt ask "Did that account also pay for the antennas?"   --conversation 2026-10-07T18-11-04-who-paid-for-the-neral-7-refit-and-through --trace
```

```text
Yes—**the antennas were paid from the same maintenance reserve** that covered the Neral-7 refit.
The antenna purchase was **PO-7731**, and the account ledger identifies the reserve used for the
refit as **KS-1180**. However, the documents do not explicitly print “KS-1180” alongside PO-7731;
they connect them by calling it the same reserve. (...procurement-memo.md; ...export.json;
...field-note-kestrel.md)
...
Trace
round 1
  exact_search(terms=['KS-1180', 'antennas'], mode='refs', limit=50) — 3 documents, 0.00s
  semantic_search(query='whether the maintenance reserve account KS-1180 paid for the Neral-7 antennas or antenna equipment', limit=10) — 10 passages, 0.88s
  list_documents(limit=50) — 11 documents, 0.00s
round 2
  fetch_source(ref='…/reports/procurement-memo.md', offset=0, limit=30, source_version='9b9fa0ad…') — 18 lines, 0.04s
  fetch_source(ref='…/intake/export.json', offset=0, limit=30, source_version='da7e1e0c…') — 5 lines, 0.01s
  fetch_source(ref='…/reports/field-note-kestrel.md', offset=0, limit=30, source_version='a0893090…') — 21 lines, 0.02s
Usage: 5 calls, 19,817 tokens, ~$0.004842
```

In round 2 the model read the three source documents in full with
`fetch_source`. The answer separates what the documents state (the same
reserve) from what they do not (`KS-1180` never appears beside `PO-7731`).
That distinction is what to check against the cited sources.

```bash
osintgpt conversations show 2026-10-07T18-11-04-who-paid-for-the-neral-7-refit-and-through
osintgpt conversations delete 2026-10-07T18-11-04-who-paid-for-the-neral-7-refit-and-through
```

Threads are JSONL files under the project's `conversations/` directory,
holding each answer with its full trace. The app's conversation view reads
the same files.

## JSON output

```bash
osintgpt ask "Who edited the Sorin-3 status page during the outage?" --json
```

The object carries `answer`, `sources`, `followups`, `degraded` (the reason
the static fallback was used, if it was), `trace`, and `usage`, plus
`conversation` when the answer was recorded in a thread. Each trace call
records `tool`, `arguments`, `results`, `documents`, `seconds`, and `error`.
To answer a list of questions into one JSONL file, use
[`scripts/batch_ask.py`](../scripts/batch_ask.py).

## Cost control

`cost_ceiling_usd` stops any single command once its estimated spend passes
the ceiling. For a one-off limit on a command, set it for the project and
remove it afterwards:

```bash
osintgpt config set cost_ceiling_usd 0.05
osintgpt ask "Which node acknowledged sequence LX-204?"
osintgpt config set cost_ceiling_usd unset
```

`unset` returns the project to the user default (`--user`) or to no ceiling.
