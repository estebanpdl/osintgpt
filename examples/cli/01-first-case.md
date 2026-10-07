# First case

Create one isolated project, register prose material, check the setup,
index it, and ask a question. Run from the repository root.

In the output below, paths under the repository are printed as
`D:/i/tools/osintgpt/...`; yours show your own checkout.

## 1. Generate the corpus and create a project

```bash
python examples/data/make_corpus.py
osintgpt project create "Synthetic relay case"
osintgpt project use synthetic-relay-case
```

```text
Wrote 11 material files, 8 questions, and a watchlist to examples\data\generated\case
```

`project use` selects the project for every later command. Any command also
accepts `--project SLUG` to target another project without changing the
selection. `osintgpt project list` and `osintgpt project show` print what
exists and the selected project's settings.

## 2. Preview, then register material

`dry_run.py` reads and chunks a folder without registering or embedding
anything, and estimates the embedding cost:

```bash
python examples/scripts/dry_run.py examples/data/generated/case/material/reports
```

```text
examples\data\generated\case\material\reports
5 files, 5 documents, 5 chunks, 764 tokens, ~$0.0000 to embed

would index
  field-note-kestrel.md                 1 docs       1 chunks        195 tokens
  interview-transcript.txt              1 docs       1 chunks        174 tokens
  incident-amber.html                   1 docs       1 chunks        141 tokens
  nota-operativa.md                     1 docs       1 chunks        134 tokens
  procurement-memo.md                   1 docs       1 chunks        120 tokens
```

Register both prose folders. Registering records a location; files stay
where they are and nothing is copied:

```bash
osintgpt add examples/data/generated/case/material/prose
osintgpt add examples/data/generated/case/material/reports
osintgpt sources
```

```text
Synthetic relay case sources
D:/i/tools/osintgpt/examples/data/generated/case/material/prose
D:/i/tools/osintgpt/examples/data/generated/case/material/reports
canon
7 files currently covered
```

`canon` is the project's own Markdown directory and is always included.
Files added to a registered folder later are picked up by the next `index`.

## 3. Configure providers and a cost ceiling

Indexing needs an embedding credential and `ask` needs a generation model.
Store the credential once; `auth set` prompts for it, which keeps the key out
of shell history. A variable already exported in the environment wins over
the stored value.

```bash
osintgpt auth set openai
osintgpt config set generation_model gpt-5.6-luna
osintgpt config set cost_ceiling_usd 0.50
osintgpt doctor
```

```text
                        Providers
┌────────────┬─────────┬────────────────────────┬───────┐
│ Role       │ Backend │ Model                  │ Ready │
├────────────┼─────────┼────────────────────────┼───────┤
│ embedding  │ openai  │ text-embedding-3-small │ True  │
│ generation │ openai  │ gpt-5.6-luna           │ True  │
└────────────┴─────────┴────────────────────────┴───────┘
Locality: not local: embedding (openai) sends content to a hosted API; generation (openai) sends
content to a hosted API
...
Findings
OK embedding provider: ready to build
OK generation provider: ready to build
ISSUE store: store file does not exist
OK sources: 3 registered
```

`doctor` makes no network call unless given `--check-providers`. The store
issue clears after the first `index`. Use any model your provider offers;
the fully local setup needing no credential is in
[`05-running-local.md`](05-running-local.md).

The ceiling applies to each command run separately, not to the project's
lifetime. A provider reports usage only after a call, so a run can cross the
ceiling by that last call, then stops non-zero. If a model has no known
price or a provider omits usage, the run stops rather than continuing with
an incomplete estimate.

## 4. Index and ask

```bash
osintgpt index
```

```text
1/7 D:/i/tools/osintgpt/examples/data/generated/case/material/prose/dispatches.txt
2/7 D:/i/tools/osintgpt/examples/data/generated/case/material/prose/relay-ledger.md
...
7/7 D:/i/tools/osintgpt/examples/data/generated/case/material/reports/procurement-memo.md
7 documents, 7 chunks
Usage: 7 calls, 927 tokens, ~$0.000019
```

Running `index` again embeds only new or changed files. An interrupted run
keeps completed files and saved batches; rerun `index` to continue.

```bash
osintgpt ask "Which node acknowledged sequence LX-204?" --static --passages 3
```

```text
Vela-9 acknowledged sequence LX-204. [1][2]

Sources
• D:/i/tools/osintgpt/examples/data/generated/case/material/prose/relay-ledger.md
• D:/i/tools/osintgpt/examples/data/generated/case/material/prose/dispatches.txt
• D:/i/tools/osintgpt/examples/data/generated/case/material/reports/interview-transcript.txt

Ask next
1. What packet did Vela-9 forward to Sorin-3 after acknowledging sequence LX-204?
2. How does dispatches.txt’s report of Sorin-3’s green status in the sixth cycle compare with relay-ledger.md’s report of an amber status after packet QN-88?
3. Which nodes had the shared key in cycle four, according to the operator of Koru-5?
Usage: 3 calls, 1,760 tokens, ~$0.000745
```

`[1]` refers to the first listed source. Open it and check the claim; the
generated answer is not the source of record. Without `--static`, `ask` lets
the model choose its own retrieval tools; see
[`07-asking-and-conversations.md`](07-asking-and-conversations.md).

Next: [`02-structured-data.md`](02-structured-data.md) adds CSV and JSON
records to the same project.
