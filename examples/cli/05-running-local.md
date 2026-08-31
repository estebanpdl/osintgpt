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
osintgpt index
osintgpt ask "Which node acknowledged sequence LX-204?" --trace
```

Install Ollama separately and replace `MODEL_NAME` with a model already pulled
there. `OLLAMA_BASE_URL` may point at a different endpoint; the default is the
loopback server. With a loopback Ollama URL, sentence-transformers, and SQLite,
document content does not leave the machine at query time.

Local is a data-boundary choice, not a promise of identical answers. Model
size and tool-calling ability affect answer quality and whether the agentic
loop can use retrieval tools well. If tool calling is unavailable, osintgpt
falls back to the static retrieval-and-answer path and says so in the trace.
