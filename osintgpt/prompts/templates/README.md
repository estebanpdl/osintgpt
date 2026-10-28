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
| `summarize` | `prompts.basic_summarization`; explicit provider generation | — |
| `topic_modeling` | `prompts.topic_modeling_summarization` | — |
| `sentence_details` | Public template for explicit generation; no automatic caller | — |
| `answer` | `answering.build_prompt` → `answer_question` | `question`, `passages` |
| `search_terms` | `lexical.derive_search_terms` | `max_terms` |
| `graph_extraction` | `graph.extract_document` | —; source and bounded entity context are user-message JSON |
| `graph_quote_repair` | `graph.extraction._repair` | —; fixed candidates and source ranges are user-message JSON |
| `agentic` | `agentic.agentic_answer` | `today` |
| `followups` | `followups.suggest_followups` | `n`, `question`, `answer`, `passages`, `asked` |
| `transcription` | `ingestion.transcription.transcriber_for` | `page` |

## Two kinds of prompt

The distinction is not enforced anywhere, so it has to be understood.

**Contract prompts** specify an output shape something downstream depends on.
`sentence_details` retains its historical JSON shape for direct template users.
Its old facade caller has been removed. Applications that parse its
`Subject or topics` field must keep their parser aligned with the template;
changing the shape can silently change which text they embed.

**Voice prompts** shape tone and behaviour with nothing parsing them.
`summarize` and `topic_modeling` are read by a person, so they are safe to
edit freely.

`agentic` is a voice prompt too, and the highest-leverage one here. Nothing
parses it, but it is where the survey-before-drill habit is taught — the tools
enforce no order, so the wording is the only thing making a cheap survey the
obvious first move. It also carries today's date, which is how the model
resolves "last week" without anything here parsing a phrase.

`answer` is a voice prompt by that test — nothing parses the reply — but it
is the one where editing carries real cost. Its rules are what keep an answer
inside the retrieved passages, and weakening them produces confident text that
cites nothing and reads exactly like the grounded version.

If per-project prompt overrides are ever built, only voice prompts are safe to
expose.

`transcription` is a **contract** prompt with no parser behind it, which makes
it the most dangerous one here. Its output is not checked against anything —
it *becomes* the page, and every leg then treats it as the source document.
The rules that matter are the ones forbidding improvement: do not translate,
do not correct spelling, do not guess at a digit. A model that tidies a
scanned page produces text that reads better than the original and no longer
matches it, and nothing downstream can tell.

`graph_extraction` is a **contract** prompt. Its reply declares local entity IDs,
supported aliases/identifiers, and qualified claims. Every relationship needs
quotations located in the supplied source ranges. Matching normalizes NFC and
whitespace only, without language-specific stop words. Relation polarity (`relation_polarity`), assertion status,
attribution, and dates preserve the model's interpretation in any source language;
matching a quotation alone cannot prove that interpretation correct.

`graph_quote_repair` is a **contract** prompt used at most once per window to
correct unlocated quotations for bounded candidates. It cannot change the claim's
endpoints, relation, or qualifications. Rejected candidates and correction outcomes
are retained in the graph audit, while only validated claims enter traversal.

`followups` is a **contract** prompt — the reply is parsed as a JSON array —
and its instructions carry a second contract nothing can check: every
suggestion must stand alone. A CLI prints them numbered and an interface makes
each a button, so a question containing "it" or "that answer" arrives
meaningless when the analyst sends it. Weakening that line produces
suggestions that read well and cannot be used.

`search_terms` is a **contract** prompt: `derive_search_terms` parses a JSON
array out of the reply. Its failure is soft — an unparseable reply means the
lexical leg sits out and the semantic leg still answers — but a reworded
prompt that stops returning an array turns the exact-match leg off silently,
which looks like retrieval getting slightly worse.

## Standalone templates after legacy removal

`sentence_details` asks the model to reduce a question to its subject before
that subject is embedded. The legacy `OpenAIGPT` caller has been removed, but
the public template remains available through `static_prompt('sentence_details')`.
It performs no work by itself. Agentic retrieval composes its own tool queries;
`search_project` embeds the query it is given without this preprocessing step.

The `summarize` and `topic_modeling` templates also remain accessible through
their public prompt helpers. Use the resulting text as the system argument to
`generator.generate(system, user)`. This does not recreate the removed
`SemanticOperations` recursion, chunk handling, or result shape. See the
[migration guide](../../../docs/migration-legacy-api.md).

## What is deliberately not here

**Operator-facing messages** — the missing-key notices, the unmapped-source
error, the placeholder for a PDF page that could not be read. No model ever
sees them, so they live with the code that decides to emit them.

**Assembly logic** — how retrieved excerpts are framed, how a breadcrumb is
joined to a chunk, how history is truncated. That is code operating on runtime
data, not authored text.
