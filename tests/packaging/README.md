# Package and optional-backend checks

Core includes the provider SDKs, Streamlit, document readers, NumPy, Jinja2 and
httpx. NumPy performs SQLite vector search;
Jinja2 renders prompts; httpx supplies the Gemini transport error contract.
These are direct dependencies even when another package also installs them.
httpx2 is inspected conditionally for SDK-wrapped protocol errors. SDKs that
use that transport own its installation; SDKs using httpx need no httpx2.

Runtime extras remain `qdrant`, `postgres` and `local`; `all` combines them.
`dev` installs test tools. The local encoder imports sentence-transformers on
construction, which in turn owns torch. Importing a backend class does not
establish a connection or download a model.

```shell
python -m pytest tests/packaging tests/vector_store/test_optional_qdrant.py tests/llm/test_transport_imports.py -ra
```

`test_dependency_contract.py` inventories direct Python import statements and
checks their declared owners. Its Python 3.10 grammar check is static; runtime
and dependency-resolution support still need the supported-interpreter CI
matrix. The import finder in `test_optional_backends.py` simulates all runtime
extras being absent, including on a development machine that has them installed.
Factory tests inject Postgres drivers and a local encoder. Existing Qdrant
tests inject a client to check lazy exports and factory selection. These fakes
check routing and errors, not server behavior or model quality. With
qdrant-client installed, `tests/vector_store/test_store_contract.py` also exercises
its actual in-memory backend. Postgres contract tests need a separately
authorized service profile; ordinary tests clear service selectors.

## Built-wheel smoke

Commit the candidate first. Run with the prepared core-only project Python,
which must have setuptools >=77 and pip, but none of the runtime extras:

```shell
python ci/wheel_smoke.py --scratch /absolute/path/outside/checkout/new-wheel-check
```

On Windows, use an absolute Windows path for `--scratch`. The path must not
already exist. The command archives Git HEAD, builds from that clean export
with `--no-index --no-deps --no-build-isolation`, creates a
`--system-site-packages` venv, and installs only the wheel with
`--no-index --no-deps --ignore-installed`. It never installs into the parent
environment. It retains the archive, wheel hash/content manifest, logs, venv
and command receipts outside the checkout for inspection or later cleanup.

The smoke requires the optional distributions to be genuinely absent, asserts
that package and metadata originate inside the venv, runs the installed CLI
help, imports modern APIs and the app module, rejects retired modules/exports, renders all
packaged prompts, verifies app assets, persists/reopens/searches/deletes a
synthetic SQLite record, and checks missing-extra guidance. Copied offline
guards isolate user state and block outbound Python connections; the source
package is never placed on the installed probe's import path.

This is an isolated wheel installation **over inherited dependencies**. It is
not a fresh dependency solve or a proof that every supported Python version
can install the declared lower bounds. A clean-environment core/per-extra CI
profile is a separate check. No server, provider request or model download is
part of this recipe.

## Removed APIs and distributions

`test_legacy_removal.py` checks absent legacy imports/exports and surviving
source/example imports. `test_distribution_contents.py` uses the development
setuptools dependency to build an sdist and then rebuild a wheel from it,
entirely offline in pytest scratch. Both archives must exclude retired code;
the sdist includes migration/changelog documents, and the wheel must retain
every surviving package source and prompt/app asset byte-for-byte.
`test_legacy_log_export.py` exercises the read-only export example and modern
thread creation beside a synthetic old database, verifying the database bytes
and previously exported CSVs are preserved. No private logs are read.
