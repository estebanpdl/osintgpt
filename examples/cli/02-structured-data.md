# Structured data

CSV, JSON, JSONL, and spreadsheets are record sets. osintgpt refuses to guess
which field contains searchable material because a full index of identifiers
and repeated field names looks healthy while retrieving badly.

After completing the first walkthrough, try the CSV without a mapping:

```bash
osintgpt add examples/data/generated/case/material/records/events.csv
```

The command refuses and lists the available fields. Register it again with
the roles chosen explicitly, then do the same for the JSONL file:

```bash
osintgpt add examples/data/generated/case/material/records/events.csv \
  --map content=summary \
  --map metadata=record_id,script \
  --map timestamp=reported_at \
  --map identity=record_id
osintgpt add examples/data/generated/case/material/records/messages.jsonl \
  --map content=message \
  --map metadata=script,sequence \
  --map identity=sequence
osintgpt sources
osintgpt index
```

`content` is chunked and embedded. `metadata` travels with the record but is
not embedded, so repeated labels do not make unrelated records appear alike.
`timestamp` is kept separately for retrieval filters, while `identity` keeps a
record's citation stable when rows are reordered. `--map` is repeatable, and
the content and metadata roles accept comma-separated field names.
