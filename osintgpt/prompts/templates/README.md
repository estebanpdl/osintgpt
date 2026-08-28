# Prompts — index

Every instruction osintgpt sends to a model lives here as a `.md.j2`
template. **Read the files** — this page maps each one to where it is used,
not what it says. A copy of the text here would drift from the text itself.

Loaded through `templates.py`:

| Helper | Use |
|---|---|
| `prompt(name, **context)` | Templates with variables. |
| `static_prompt(name)` | No variables; cached, since the result cannot differ. |

`StrictUndefined` is on: a template referencing a variable the caller forgot
raises at render rather than shipping a sentence with a hole in it.

Jinja rather than `str.format`, because a prompt that specifies JSON output
contains literal `{` and `}` and `.format` would require doubling every one.

## The templates

| Template | Sent by | Variables |
|---|---|---|
| `summarize` | `prompts.basic_summarization` → `SemanticOperations.summarize_content` | — |
| `topic_modeling` | `prompts.topic_modeling_summarization` | — |
| `sentence_details` | `OpenAIGPT.analyze_sentence_details` (deprecated) | — |
| `answer` | `answering.build_prompt` → `answer_question` | `question`, `passages` |
| `search_terms` | `lexical.derive_search_terms` | `max_terms` |

## Two kinds of prompt

The distinction is not enforced anywhere, so it has to be understood.

**Contract prompts** specify an output shape something downstream depends on.
`sentence_details` must return that JSON object, because its caller reads
`Subject or topics` out of the reply to decide what to embed. Rewording the
instructions is fine; changing the shape breaks the caller **silently** — the
failure looks like "search results got worse", not like an exception.

**Voice prompts** shape tone and behaviour with nothing parsing them.
`summarize` and `topic_modeling` are read by a person, so they are safe to
edit freely.

`answer` is a voice prompt by that test — nothing parses the reply — but it
is the one where editing carries real cost. Its rules are what keep an answer
inside the retrieved passages, and weakening them produces confident text that
cites nothing and reads exactly like the grounded version.

If per-project prompt overrides are ever built, only voice prompts are safe to
expose.

`search_terms` is a **contract** prompt: `derive_search_terms` parses a JSON
array out of the reply. Its failure is soft — an unparseable reply means the
lexical leg sits out and the semantic leg still answers — but a reworded
prompt that stops returning an array turns the exact-match leg off silently,
which looks like retrieval getting slightly worse.

## What is on its way out

`sentence_details` asks the model to reduce a question to its subject before
that subject is embedded — reformulation, and the right instinct: request words
like "summarize" pollute a similarity search. It survives because its caller is
part of the deprecated `OpenAIGPT` surface and someone's script depends on it.

Agentic retrieval absorbs it. A model that calls `semantic_search(query=...)`
composes that phrase itself, inside a call it was already making, so the
separate pre-flight stops being cheaper-and-better and simply stops having a
caller. The instinct survives as a standing instruction in the retrieval
prompt; the round trip does not. Explicit reformulation remains only in the
static fallback, where no model is choosing the query.

Read this template as a record of the current API, not a pattern to extend.

## What is deliberately not here

**Operator-facing messages** — the missing-key notices, the unmapped-source
error, the placeholder for a PDF page that could not be read. No model ever
sees them, so they live with the code that decides to emit them.

**Assembly logic** — how retrieved excerpts are framed, how a breadcrumb is
joined to a chunk, how history is truncated. That is code operating on runtime
data, not authored text.
