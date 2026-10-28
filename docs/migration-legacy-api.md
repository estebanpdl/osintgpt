# Migrating from the legacy API

The next minor release after 0.7.1 removes `osintgpt.llms`,
`osintgpt.embeddings`, `osintgpt.semantic_operations`, and `osintgpt.databases`.
This is a **breaking change in a 0.x minor release**. Earlier warnings and the
migration example promised removal in 1.0; that promise has changed. There is
no intervening notice-only release. These imports now raise `ModuleNotFoundError`;
there are no compatibility aliases. This document describes the unreleased
removal; the source version remains 0.7.1 until release preparation.

Scripts that still require these APIs can retain the last compatible series:

```shell
python -m pip install "osintgpt<0.8"
```

Use a separate environment for those scripts. Pinning retains old behavior;
it does not migrate a script or its data. Modern code uses `osintgpt.llm`
(singular), project retrieval/answering, and project conversations.

## Removed imports and methods

The package imports below and their implementation submodules are removed:
`llms.openai_gpt`, `llms.search`, `llms.completions`, `llms.conversation`,
`embeddings.openai_embeddings`, `semantic_operations.operations`, and
`databases.sql_manager`.

| Removed API | Replacement or explicit limitation |
|---|---|
| `llms.OpenAIGPT(...)` | `llm.build_generation_provider('openai', settings, model=...)`. It returns a generation provider, not a combined search/logging facade. |
| `OpenAIGPT.client` proxy; `get_openai_api_key()` | Supply credentials through `Settings` to the factory. SDK client attributes are provider-specific; there is no portable client/key accessor on the provider interface. |
| `OpenAIGPT.count_tokens(prompt)` | `utils.count_tokens(prompt, model=generator.model)`; an approximation, especially for unknown models. |
| `OpenAIGPT.estimated_prompt_cost(prompt)` | No equivalent prompt-only facade method. Use `llm.UsageRecorder` for reported call usage and available price estimates. |
| `CompletionsMixin.get_model_completion(...)`, `get_model_completion_using_system_role(...)` | `generator.generate(system, user)` returns text. Use `generate_response` for the modern structured response API. Arbitrary SDK kwargs and old message-list arguments are not accepted by `generate`. |
| `CompletionsMixin.interactive_completion(...)` | No equivalent blocking console method. Use the CLI/app, or manage your own input loop and thread context. |
| `CompletionsMixin.analyze_sentence_details(...)` | No equivalent facade method. `prompts.templates.static_prompt('sentence_details')` remains available for explicit generation; no automatic subject extraction or embedding is performed. |
| `SearchMixin.load_embeddings_from_csv(...)`, `load_embeddings_from_dataframe(...)`, `get_embeddings()`, `get_embeddings_dataframe()` | No mutable dataframe/CSV facade equivalent. Register source documents, configure structured tabular ingestion where needed, and call `index_project`. Existing embedding CSVs are not automatically imported. |
| `SearchMixin.search_results_from_vector(...)`, `search_results_from_dataframe(...)`, `_relatedness_fn(...)` | `search_project(project, query, embedder)`, `hybrid_search(...)`, or a modern store's `search(vector, embedding_model, ...)`. These return typed results, not the old dictionary of vector/text/score tuples. No dataframe or scalar-cosine helper equivalent. |
| `embeddings.OpenAIEmbeddingGenerator(...)` | `llm.build_embedding_provider('openai', settings, model=...)`. An embedding provider does not require a chat model. |
| `OpenAIEmbeddingGenerator.client`, `get_openai_api_key()`, `get_openai_gpt_model()`, `get_openai_embedding_model()` | Configure through `Settings` and the factory's `model=`; read `embedder.model`. No portable SDK client/key/chat-model accessor. |
| `load_text(...)`, `calculate_embeddings()`, cached `embeddings`, `generate_embedding(text)` | `embedder.embed(texts)` or `embedder.embed([text])[0]`. Input is explicit on every call; no mutable loaded-text state or automatic result cache. |
| `OpenAIEmbeddingGenerator.count_tokens()`, `calculate_embeddings_estimated_cost()` | Sum `utils.count_tokens(text, embedder.model)` for an approximate token count. No equivalent loaded-text cost method; use `UsageRecorder` for calls. |
| `OpenAIEmbeddingGenerator.load_embeddings_from_csv(...)` | No CSV loader on the provider. Read data separately or re-index registered source material. |
| `semantic_operations.SemanticOperations(...)`, `summarize_content(...)` | No replacement class. Use explicit `generator.generate(system, user)` for summarization (example below). No automatic splitting, recursive aggregation, or dataframe conversion is implied. |
| `SemanticOperations.semantic_similarity_search(...)` | Use project search or `answer_question`/`agentic_answer`. There is **no equivalent** for the old recursive search/summary algorithm or its result shape. |
| `ConversationLogMixin.insert_system_prompt_into_sql_database(...)`, `insert_user_prompt_into_sql_database(...)`, `insert_completion_response_into_sql_database(...)` | `projects.conversations.start_conversation`, `append_turn`, and `open_conversation` persist question/answer turns. They do not log every system/user/SDK message or preserve legacy SQL IDs. |
| `ConversationLogMixin._get_completion_response_id(...)`, `_get_completion_response_usage(...)`, `_get_completion_response_role_and_message(...)`, `_generate_unique_id(...)` | Removed internal response/SQL helpers. Use provider response/usage interfaces and the thread's `id`; no legacy ID or role-message mapping equivalent. |
| `databases.SQLDatabaseManager(...)`, `create_connection(...)`, `get_connection()`, `_create_chat_gpt_index_table()`, `_create_chat_gpt_conversations_table()`, `insert_data_to_chat_gpt_index(...)`, `insert_data_to_chat_gpt_conversations(...)`, `load_messages_from_chat_gpt_conversations(...)` | No SQL logging manager replacement. Use modern project threads for new conversations and standard SQLite tools for old logs, as below. |

The three mixin classes themselves are also removed. Do not subclass them.

## Direct embedding and generation

These examples can make provider calls when executed. Configure credentials
and model names explicitly before using them; `--help` on the accompanying
[factory example](../examples/library/migrating_from_0_1.py) makes no calls.

```python
from osintgpt import Settings
from osintgpt.llm import (
    UsageRecorder, build_embedding_provider, build_generation_provider,
)
from osintgpt.prompts import basic_summarization

settings = Settings.from_env()  # Reads environment variables, not an implicit .env.
usage = UsageRecorder()
embedder = build_embedding_provider('openai', settings, recorder=usage)
generator = build_generation_provider('openai', settings, recorder=usage)

vectors = embedder.embed(['First document', 'Second document'])
reply = generator.generate('Answer clearly.', 'What is an embedding?')
summary = generator.generate(basic_summarization(), 'Text to summarize')
print(usage.total_tokens, usage.estimated_cost)
```

Model selection, provider response formats, batching, pacing, and usage reporting
follow the modern provider implementation. Usage recording is opt-in; missing
provider usage and unknown prices are not proof of zero usage or zero cost.
Direct generation does not write legacy SQL logs or create a project thread.

## Project retrieval, answers, and threads

Create/register/index the corpus through the CLI or the project APIs first.
For tabular sources, use the structured ingestion mapping in
[the examples](../examples/README.md); vector CSV loading is not a migration.
The embedding model used for search must match the indexed model.

```python
from osintgpt import Project, answer_question, search_project
from osintgpt.projects.conversations import (
    append_turn, open_conversation, start_conversation,
)

project = Project.load('path/to/project')
question = 'Which records mention Alpha?'
results = search_project(project, question, embedder, top_k=5)
for result in results:
    print(result.ref, result.score, result.chunk.text)

answer = answer_question(project, question, embedder, generator)
print(answer.text)
thread = start_conversation(project, question)
append_turn(thread, question, {'answer': answer.text})
reopened = open_conversation(project, thread.id)
```

This example uses the project's default store. For a remote backend, open
`vector_store.store_for(project, settings)` and pass `store=store` to retrieval
and answering. `index_project(project, embedder, config=settings)` indexes the
registered corpus; it does not ingest a legacy embedding cache automatically.
`hybrid_search` returns fused ranks rather than raw cosine scores.
`answer_question` returns an `Answer`, while `agentic_answer` uses tools and
returns an `AgenticAnswer`; neither recreates the recursive legacy helper.

Threads are append-only `conversations/*.jsonl` files under each project, with
metadata and question/answer turn records. They are **not Markdown transcripts
or the old SQLite schema**. `append_turn` returns `None` on a write failure;
check its return value if durable history is required. The minimal example
above stores only answer text; applications needing passages, tool traces and
other answer fields must serialize those fields explicitly (the agentic API
provides `answer_to_dict`).

## Retaining, reading, and exporting old SQL logs

Removing Python code does not delete, upgrade, or overwrite any database.
Keep your historical database and a backup. The two tables are:

| Table | Columns |
|---|---|
| `chat_gpt_index` | `id`, `created_at` |
| `chat_gpt_conversations` | `id`, `ref_id`, `chat_id`, `role`, `message` |

`ref_id` joins a message to `chat_gpt_index.id`. Order messages by the integer
`chat_gpt_conversations.id`; grouping only by `chat_id` can split a conversation
across completions. Modern threads do not open or import these tables.

Close the old application and obtain a consistent SQLite snapshot before using
the export example. A copy of just the main file while WAL writes are active
may omit messages; use SQLite's backup facility or a checkpointed, closed
database. Keep the original. The example treats its input as immutable, so it
must not be used on a live database or a copy with uncheckpointed WAL.

```shell
python examples/library/export_legacy_logs.py closed-snapshot.sqlite new-export
```

The [export example](../examples/library/export_legacy_logs.py) uses standard
library SQLite read-only access, creates a new directory, and writes both
tables as UTF-8 CSV with headers. Existing output directories are rejected.
It preserves Unicode and embedded newlines. It does not convert logs to modern
threads. Keep exported CSVs as data; spreadsheet software may interpret cells
starting with formula characters when opening them.

To read one old thread without exporting, use a closed snapshot:

```python
from contextlib import closing
from pathlib import Path
import sqlite3

uri = Path('closed-snapshot.sqlite').resolve(strict=True).as_uri()
with closing(sqlite3.connect(uri + '?mode=ro&immutable=1', uri=True)) as connection:
    rows = connection.execute(
        'SELECT role, message FROM chat_gpt_conversations '
        'WHERE ref_id = ? ORDER BY id', ('your-thread-id',),
    ).fetchall()
```

There is no automatic SQL-to-thread migration, ID translation, or promise that
modern retrieval/answering reproduces an old result.

## APIs and assets that remain

All modern top-level exports, `llm` factories, SQLite/Qdrant/Postgres stores,
`BaseVectorEngine`, the low-level Qdrant facade, structured ingestion, shared
settings, cost utilities, and prompt infrastructure remain. `sql_db_file_path`
and its project override remain accepted compatibility settings; they do not
turn direct generation into SQL logging. `utils.create_unique_id`, both
summarization prompt helpers, and the `sentence_details`, `summarize`, and
`topic_modeling` templates remain separately accessible public helpers/assets.
Their preservation does not restore the removed facade methods.
