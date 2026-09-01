# Evaluating retrieval

An evaluation set pairs questions with the documents known to answer them.
The generated `questions.toml` contains three such questions. Create or load a
project inside the generated case, index its prose, and score it:

```bash
python examples/library/index_a_folder.py \
  examples/data/generated/case \
  examples/data/generated/case/material/prose \
  --embedding-provider PROVIDER --embedding-model MODEL_NAME
python examples/library/evaluate_retrieval.py \
  examples/data/generated/case \
  examples/data/generated/case/questions.toml \
  --embedding-provider PROVIDER --embedding-model MODEL_NAME
```

Evaluation is currently a library operation; there is no `osintgpt evaluate`
command. The report prints hit rate (questions with any expected document),
mean reciprocal rank (how early the first expected document appeared), and
recall (how many expected documents appeared within `--top-k`, which defaults
to 10).

These scores measure retrieval against this question set. They do not measure
whether a generated answer is factual or complete, and they are only as useful
as the expected documents and questions chosen. Read the misses and unscorable
questions instead of treating one aggregate as a quality verdict.
