# Evaluating retrieval

An evaluation set pairs questions with the documents known to answer them.
The generated `questions.toml` contains three such questions. After creating
a project for the generated case and indexing its prose, score it directly
from the command line:

```bash
osintgpt evaluate examples/data/generated/case/questions.toml \
  --project case --top-k 10
```

The command uses the project's configured embedding provider and model, so it
needs the same credentials as semantic indexing and search. The report names
the retrieval method and embedding model, then prints hit rate (questions with
any expected document), mean reciprocal rank (how early the first expected
document appeared), and recall (how many expected documents appeared within
`--top-k`, which defaults to 10). It also lists every miss and unscorable
question.

Evaluation also reports its embedding calls and estimated cost. JSON includes
whether every billable call was counted and priced, so a partial estimate
cannot be mistaken for a complete one.

To measure hybrid retrieval, add literal `terms` to the relevant questions in
the TOML set and run:

```bash
osintgpt evaluate examples/data/generated/case/questions.toml \
  --project case --retrieval hybrid --json
```

Hybrid evaluation uses only the terms recorded in the question set. It does
not ask a generation model to derive extra terms, keeping runs reproducible.

These scores measure retrieval against this question set. They do not measure
whether a generated answer is factual or complete, and they are only as useful
as the expected documents and questions chosen. Read the misses and unscorable
questions instead of treating one aggregate as a quality verdict.
