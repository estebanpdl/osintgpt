# Changelog

## 0.9.0 — 2026-10-29

### Added

- Snowball semantic chains: start from a query, take the closest passage as
  the next query, and repeat. Available as an app page (between Graph and
  Settings), `osintgpt snowball` and `from osintgpt import snowball`; the
  agentic chat tool uses the same operation.
- Defaults: depth 25 and an inclusive 0.5 similarity threshold (agentic tool:
  5 hits). Text deduplication after NFKC, case and whitespace normalization;
  `--dedupe chunk` keeps file/chunk identities.
- Each hop reports the seed, full hit text, score, source, date and similarity
  to the initial query. Optional date grouping by day, week, month or year.
- CSV/XLSX export with one row schema; formula-like text is escaped and
  existing files are not overwritten.
- Embedding usage and cost are reported per run.

### Notes

- A chain is a similarity path, not evidence of causation or propagation.
- No breaking changes, new dependencies or stored-data changes.
