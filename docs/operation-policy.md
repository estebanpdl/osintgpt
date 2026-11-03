# Operation policy

Snowball's library, CLI, app, and agentic-tool entry points use
`osintgpt.operations.OperationContext`. Other operations still use their
existing adapters. The context currently resolves embedding and store
dependencies; generation roles are not yet part of this interface.

Create a fresh context per invocation. Project settings inherit unset values
from user defaults, and nonempty invocation `overrides` take precedence. Model
values from explicit `Settings` or the environment fill only choices left empty
by those sources. Connection settings carry credentials separately from project
choices. The resolved settings and configuration are snapshots: later changes
to saved defaults do not alter an active operation. A resolved project copy
carries inherited storage choices to the store factory without saving them.

```python
from osintgpt.config import Settings
from osintgpt.operations import OperationContext
from osintgpt.llm.base import EmbeddingPurpose

# project is an explicitly selected Project; no UI selection is consulted.
with OperationContext(project, config=Settings(openai_api_key="...")) as operation:
    vectors = operation.embed(["research question"], purpose=EmbeddingPurpose.QUERY)
    hits = operation.store.search(
        vectors[0], embedding_model=operation.embedder.model, top_k=5,
    )
print(operation.outcome.status, operation.outcome.usage)
```

Providers and stores are constructed only on use. A context without work opens
neither. Injected resources remain caller-owned; their recorders are never
replaced. Factory-created resources belong to the operation and are closed on
success, failure, cancellation, or a cost stop when they expose `close()`.
Cleanup does not mask the original failure. A context cannot be reused, and its
dependencies cannot be accessed outside its `with` block. Factories must return
fresh resources rather than cached clients from another operation.

Built-in providers emit usage events. The context observes these without also
attaching its recorder to the provider. Each response reaches its recorder and
any enclosing observer once. `operation.embed()` records a successful custom
provider call without events as unknown usage; it does not infer zero tokens or
zero cost. Injected models are checked against explicit or inherited model
choices before work. This does not inspect or validate every model in an index.

Ceilings are optional. The lower nonempty project/user or invocation ceiling
applies, and zero is an explicit ceiling. A run override cannot raise a saved
ceiling. Existing usage rules check after a response: the final call can exceed
the ceiling. Unknown usage or prices stop the run when a ceiling is configured,
because the limit cannot be enforced with incomplete accounting.
No ceiling means no accounting stop. Calls billed by an enclosing provider's
recorder retain that caller's policy as well.

`OperationOutcome` contains a status (`completed`, `failed`, `cancelled`, or
`cost_limit`), a generic error category, and a usage snapshot. Unknown totals are
`None`; the priced subtotal remains separate as `known_cost_usd`. The optional
`on_event` callback receives `usage` and `outcome` events with these safe fields,
never queries, credentials, connection settings or raw exception messages.
Ordinary display failures are isolated; cancellation and hard stops propagate.
`CostLimitReached.operation_outcome` retains accounting for adapters that handle
the stop. Other exceptions propagate unchanged. Snowball does not persist partial
chains on failure or cancellation.
