# Changelog

## 0.8.0 — 2026-10-28

### Breaking changes

- Remove `osintgpt.llms`, `osintgpt.embeddings`, `osintgpt.semantic_operations`,
  and `osintgpt.databases`, including their classes, mixins, dataframe search,
  mutable embedding helpers, and SQL conversation logging APIs.
- This removal changes the previously published **removed in 1.0** promise.
  It ships directly in the next **0.x minor**, with no notice-only release
  first. It is not a patch-compatible change.
- Use modern provider factories, project search/answers, and JSONL project
  conversations. Historical SQL databases remain untouched; no automatic
  conversion is performed. Scripts requiring the old API can pin
  `osintgpt<0.8`. See the [migration guide](docs/migration-legacy-api.md) for
  every removed surface, behavioral differences, unsupported equivalents,
  and read/export examples.

### Testing

- Replace legacy-only tests with removed-import/export and built wheel/sdist
  checks, retaining modern project isolation and provider-boundary coverage.
- Filter only the Google GenAI `'_UnionGenericAlias'` deprecation message from
  `google.genai.types`; other warnings remain visible.
