# Running the tests

Run from the source checkout in the project's prepared Python environment.
For Conda, prefix the commands below with `conda run -n osintgpt`.
`python -m pytest` loads the checkout's package; an installed `osintgpt`
console command can load a different copy. Distribution metadata still comes
from the installed distribution, so keep its version aligned with the source.

```shell
python -m pytest
python -m pytest -ra --durations=20
python -m pytest tests/config -ra --durations=20
```

The default is serial, with normal pytest output, strict markers and the
`tests/` discovery root. Warnings remain visible; narrow filters already used
by individual tests concern the behavior they explicitly exercise. There is
no global warning suppression. `PYTEST_ADDOPTS` and third-party plugins can
alter pytest behavior; record or clear inherited overrides when collecting
comparable evidence. Installing xdist or pytest-cov does not enable them.

## Offline boundaries

Before importing application or test modules, `tests/conftest.py` activates
`ci/sitecustomize.py` and `ci/offline_environment.py`. They block non-loopback
Python TCP connections, clear provider settings and service selectors, and
redirect home/configuration directories. The autouse `isolated_environment`
fixture repeats configuration clearing and gives each test a temporary home.
Python subprocesses inherit the source path and startup guard, including xdist
workers. Mock transports and in-memory stores remain available.

Python file/SQLite opens into the original `~/.osintgpt` are refused.
Explicit `.env` and `credentials.toml` files must be inside the current test
sandbox. Tests may create synthetic configuration there; no developer keys
or credential files are required. These checks and home redirection are not
an OS sandbox: native clients, subprocesses that discard the environment,
metadata-only filesystem operations and every possible path alias are not
universally intercepted. Loopback remains available for controlled fakes and
Python's event-loop machinery; it is not permission to call a local provider.

The store contract contains dormant Postgres and remote-Qdrant branches.
Ordinary pytest clears `OSINTGPT_TEST_POSTGRES_DSN` and
`OSINTGPT_TEST_QDRANT_HOST`: Postgres cases skip, and Qdrant uses its in-memory
client when installed. Missing optional dependencies produce attributed skips.
The reserved `integration` marker has no selection or opt-in implementation.
Before adding service/live tests, introduce an explicit opt-in harness with
default exclusion and isolated endpoints. A marker alone never grants network
or paid-provider permission. Benchmark scripts that explicitly use models are
separate from the offline `test_*.py` cases and must not be run as ordinary tests.

Token-count tests require public tiktoken encoding assets already cached.
Prepare them separately before enabling the socket guard, if they are absent:

```shell
python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('cl100k_base', 'o200k_base')]"
```

This downloads static data, not model inference. Set `TIKTOKEN_CACHE_DIR` to a
dedicated cache if needed and retain that value for pytest. A missing cache
fails visibly under the guard; tests never silently fetch assets. Session
scratch homes use the system temporary directory (or the configured
`TEMP`/`TMPDIR`); per-test homes follow pytest's `tmp_path`/`--basetemp`.

## Evidence and CI profiles

For a full regression gate, retain the 25 slowest timings:

```shell
python ci/run_tests.py --log test-results/serial.log -- -ra --durations=25
```

The wrapper preserves combined stdout/stderr even when pytest fails, writes
a sibling JSON receipt with the command, working directory, exit code and
wall time, then returns pytest's exit code. The runner duration is the number
in pytest's final summary; receipt wall time includes process startup. The
harness regression verifies capture using an intentionally failing synthetic
suite; that failure is expected only inside the regression's child process.

Compare serial and two-worker pass/fail/skip counts on the same commit and
environment before adopting this explicit parallel profile:

```shell
python -m pytest -n 2 --dist=loadfile -ra --durations=20
```

File grouping keeps each module's fixtures and state in one worker. Local
default runs and CI use the same selection and guards. A designated CI job
can add `--cov=osintgpt --cov-report=term-missing`; coverage measures executed
lines, not investigative accuracy. CI needs the same prepared tokenizer cache.

Record interpreter and package origins, metadata version/origin, loaded
plugins, command, worktree/commit, worker count, collected/run/pass/fail/skip/
deselect totals, warning/skip reasons, exit code and runner duration. Also
record concurrent heavy work; timing on a shared workstation is not an
isolated performance benchmark. Optional-dependency skips do not qualify those
backends, and offline stubs do not establish live-provider compatibility.

See [package checks](packaging/README.md) for the direct-import inventory,
optional-backend simulations and an installed-wheel smoke that keeps all
build and venv scratch outside the checkout.
