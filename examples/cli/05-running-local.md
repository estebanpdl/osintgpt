# Running locally

Fully local operation has two separate pieces: sentence-transformers embeds
inside the Python process, and Ollama serves generation on the machine. The
default SQLite store is already local.

```bash
pip install "osintgpt[local]"
osintgpt config set embedding_provider sentence-transformers
osintgpt config set generation_provider ollama
osintgpt config set generation_model MODEL_NAME
osintgpt doctor
```

Install Ollama separately and replace `MODEL_NAME` with a model already
pulled there. `OLLAMA_BASE_URL` may point at a different endpoint; the
default is the loopback server. `doctor` states the verdict on its
`Locality:` line, for example
`local: nothing leaves this machine at query time`, and names the provider
that breaks it otherwise.

A model downloaded on first use needs the network during setup only.

## Re-index after changing the embedding model

Vectors from different models are not comparable. Changing the embedding
provider or model makes the next `index` rebuild every file with the new
model; search only reads chunks from the configured model.

```bash
osintgpt index
osintgpt index --purge-other-models
osintgpt ask "Which node acknowledged sequence LX-204?" --trace
```

`--purge-other-models` deletes the old model's vectors, and only when the
pass ends with no failed or skipped files. Without it they stay, unused, so switching back
costs nothing.

## What changes

Local provider calls print no dollar estimate. Their JSON usage record still
names the calls and models, with `estimated_cost_usd` set to null rather than
`0`, so a non-billable run is distinct from a measured remote bill.

Local is a data-boundary choice, not a promise of identical answers. Model
size and tool-calling support affect answer quality and whether `ask` can use
its retrieval tools. A model without tool calling falls back to the static
retrieval-and-answer path, and `--trace` says so.

Any other OpenAI-compatible server (vLLM, LM Studio, a gateway) can be used
from Python; see [`library/custom_provider.py`](../library/custom_provider.py).
