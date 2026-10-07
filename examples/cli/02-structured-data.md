# Structured data

CSV, XLSX, JSON, and JSONL files are record sets: each row has many fields
and only some are worth searching. osintgpt does not guess which ones. A
wrong guess still indexes, but retrieval gets worse with no visible error.
You assign field roles with `--map`.

Continue in the project from [`01-first-case.md`](01-first-case.md).

## Field roles

| Key | Role |
|---|---|
| `content` | Text that is chunked, embedded, and searched. **Required.** Several fields are joined in the order given; the first is primary, and a record whose primary field is empty is skipped. |
| `metadata` | Kept with the record and shown with results, never embedded. Ids, URLs, counts. |
| `timestamp` | When the record was made. Kept separately and returned with results. |
| `author` | Who wrote it. Exact search matches it alongside the text. |
| `identity` | Unique record id. Becomes the citation (`posts.csv#p4`) and keeps citations stable when rows move. |
| `records` | JSON only: dotted path to the list of records, e.g. `data.items`. |
| `min_chars` | Skip records whose content is shorter than this. |

`content` and `metadata` take comma-separated lists; the others take one
field. Nested fields use dots: `user.handle`. `--map` is repeatable.

## 1. Let the file describe itself

```bash
osintgpt add examples/data/generated/case/material/intake/posts.csv
```

Without a mapping, `add` refuses and reports each field (trimmed here):

```text
posts.csv needs a content field mapping
fields: {"post_id": {"filled": 4, "sampled": 4, "average_length": 2, "unique": true, "example": "p1"},
 "channel": {"filled": 4, "sampled": 4, "average_length": 10, "unique": false, "example": "vela_watch"},
 "message": {"filled": 3, "sampled": 4, "average_length": 56, "unique": true, "example": "Vela-9 is back online ..."},
 ...}
```

`message` is filled in 3 of 4 rows and averages 56 characters: that is the
body. `post_id` is unique and 2 characters long: that is an identity.
`python examples/scripts/dry_run.py <folder>` prints the same field
description for every unmapped file in a folder.

## 2. Test a mapping with `--dry-run`

`--dry-run` reads every record and registers nothing:

```bash
osintgpt add examples/data/generated/case/material/intake/posts.csv \
  --map content=message --dry-run
```

```text
posts.csv — nothing registered
3 of 4 records, 1 with no content, shortest 9 chars
shortest kept record:
@vela_ops
```

The empty row was dropped, but the shortest kept record is a bare handle.
Add a floor and check again:

```bash
osintgpt add examples/data/generated/case/material/intake/posts.csv \
  --map content=message --map min_chars=30 --dry-run
```

```text
posts.csv — nothing registered
2 of 4 records, 1 with no content, 1 under the floor, shortest 78 chars
shortest kept record:
Station Kestrel paid for the Neral-7 refit; the transfer used account KS-1180.
```

Judge a mapping by the shortest kept record, not the kept count. Mapping
every field as content keeps all four rows, including one made only of a
channel name and a URL:

```bash
osintgpt add examples/data/generated/case/material/intake/posts.csv \
  --map content=channel,message,url --dry-run
```

```text
posts.csv — nothing registered
4 of 4 records, shortest 37 chars
shortest kept record:
vela_watch

https://relay.example/p/2
```

Across thousands of rows, records like that look alike to an embedding model
and crowd out real material. Put ids, URLs, and channel names in `metadata`,
`author`, or `identity` instead.

## 3. Nested JSON

`export.json` wraps its records under `data.items`. Without `records`, the
whole file is one record with no `text` field:

```bash
osintgpt add examples/data/generated/case/material/intake/export.json \
  --map content=text --dry-run
```

```text
export.json — nothing registered
0 of 1 records, 1 with no content
```

Point `records` at the list and reach nested fields with dots:

```bash
osintgpt add examples/data/generated/case/material/intake/export.json \
  --map records=data.items --map content=text --map author=user.handle \
  --map timestamp=created --map identity=id --dry-run
```

```text
export.json — nothing registered
2 of 3 records, 1 with no content, shortest 51 chars
shortest kept record:
Sorin-3 reported amber again after the power fault.
```

JSONL and NDJSON need no `records`: each line is a record. XLSX reads the
first sheet, with the first row as the header.

## 4. Register and index

Drop `--dry-run` and give every role:

```bash
osintgpt add examples/data/generated/case/material/intake/posts.csv \
  --map content=message --map timestamp=published_at --map author=channel \
  --map identity=post_id --map metadata=url,views --map min_chars=30
osintgpt add examples/data/generated/case/material/intake/export.json \
  --map records=data.items --map content=text --map author=user.handle \
  --map timestamp=created --map identity=id
osintgpt add examples/data/generated/case/material/records/events.csv \
  --map content=summary --map metadata=record_id,script \
  --map timestamp=reported_at --map identity=record_id
osintgpt add examples/data/generated/case/material/records/messages.jsonl \
  --map content=message --map metadata=script,sequence --map identity=sequence
osintgpt index
```

```text
1/4 D:/i/tools/osintgpt/examples/data/generated/case/material/records/events.csv
2/4 D:/i/tools/osintgpt/examples/data/generated/case/material/records/messages.jsonl
3/4 D:/i/tools/osintgpt/examples/data/generated/case/material/intake/posts.csv
4/4 D:/i/tools/osintgpt/examples/data/generated/case/material/intake/export.json
4 documents, 10 chunks, 7 unchanged
Usage: 4 calls, 174 tokens, ~$0.000003
```

Registering the same path again replaces its mapping, so a mapping can be
corrected without unregistering first. Changing a mapping makes the next
`index` rebuild that file. `osintgpt sources` lists each file's mapping.

Each record is its own document. An exact search shows the roles at work:
the citation carries the identity, and timestamp and author come back with
the text:

```bash
osintgpt search --exact KS-1180 --json
```

```text
{"results": [{"rank": 1, "score": 1.0,
  "citation": "D:/i/tools/osintgpt/examples/data/generated/case/material/intake/posts.csv#p4",
  "text": "Station Kestrel paid for the Neral-7 refit; the transfer used account KS-1180.",
  "timestamp": "2042-05-06 12:30", "author": "sorin_notes", "legs": ["lexical"], ...}, ...]}
```

## Same workflow elsewhere

- **App:** `osintgpt app` → **Material**. Point it at a folder; files sharing
  one set of columns share one mapping panel, with the same field table and a
  live kept/dropped preview. The app and CLI write the same project files.
- **Python:** [`library/map_structured_records.py`](../library/map_structured_records.py)
  runs `describe_fields`, `preview_mapping`, and `load_documents`, then
  registers the file.

## Checklist

- Run `--dry-run` first and read the shortest kept record.
- Map one body field as `content`, usually.
- Always set `identity` and `timestamp`.
- Put ids and links in `metadata`.
- Use `min_chars` to drop signature-only or handle-only rows.
