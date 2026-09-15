# Mapping structured data

CSV, XLSX and JSON files are record sets. Each row has many fields, and only
some of them are worth searching. osintgpt will not guess which — a wrong
guess indexes cleanly and answers badly, and nothing tells you it happened.

So you say it yourself, with `--map`.

## The seven keys

| Key | What it does |
|---|---|
| `content` | The text that gets embedded and searched. **Required.** |
| `timestamp` | When the record was made. Date filters read this. |
| `author` | Who wrote it. Exact search matches it alongside the text. |
| `identity` | The record's id. Becomes its citation, and keeps re-indexing from duplicating it. |
| `metadata` | Kept with the record, never embedded. For provenance. |
| `records` | JSON only: where the list of records sits inside the file. |
| `min_chars` | Skip records with less content than this. Optional. |

`content` and `metadata` take several fields, comma-separated. The others take
one. `--map` can be repeated.

## An example file

Say you have `path/to/posts.csv` — 4 rows, 6 fields:

| post_id | channel | message | published_at | url | views |
|---|---|---|---|---|---|
| p1 | alpha_news | Report on the shipment arriving at the northern port this week. | 2026-03-01 09:12 | https://… | 482 |
| p2 | alpha_news | *(empty)* | 2026-03-01 11:40 | https://… | 150 |
| p3 | beta_watch | @alpha_news | 2026-03-02 08:05 | https://… | 97 |
| p4 | beta_watch | Longer analysis of the same shipment, with figures and sources. | 2026-03-02 12:30 | https://… | 1203 |

Two rows are real posts. One is empty. One is just a handle.

---

## 1. Ask the file what is in it

Add it with no mapping at all:

```bash
osintgpt add path/to/posts.csv --project demo-case
```

**Outcome** — it refuses, and lists every field with how often it is filled,
how long it is, and an example value:

```
posts.csv needs a content field mapping
fields: {"post_id": {"filled": 4, "sampled": 4, "average_length": 2, "unique": true, "example": "p1"},
         "channel": {"filled": 4, "sampled": 4, "average_length": 10, "unique": false, "example": "alpha_news"},
         "message": {"filled": 3, "sampled": 4, "average_length": 46, "unique": true, "example": "Report on the shipment…"},
         ...}
try: osintgpt add "path/to/posts.csv" --project demo-case --map content=<field> --map timestamp=<field> ...
```

Read those numbers. `message` is filled 3 of 4 and averages 46 characters —
that is the body. `post_id` is unique and 2 characters long — that is an id,
not content.

## 2. Test the mapping before you keep it

`--dry-run` reports what would happen and saves nothing:

```bash
osintgpt add path/to/posts.csv --project demo-case \
  --map content=message --dry-run
```

**Outcome:**

```
posts.csv — nothing registered
3 of 4 records, 1 with no content, shortest 11 chars
shortest kept record:
@alpha_news
```

Three records kept. The empty row was dropped for you. But the shortest kept
record is `@alpha_news` — a handle with no text. That is not worth indexing.

Every record is counted, so the numbers are the ones you will actually get. A
large spreadsheet takes a few seconds to read; a CSV is instant.

## 3. Set a floor

```bash
osintgpt add path/to/posts.csv --project demo-case \
  --map content=message --map min_chars=30 --dry-run
```

**Outcome:**

```
posts.csv — nothing registered
2 of 4 records, 1 with no content, 1 under the floor, shortest 63 chars
shortest kept record:
Report on the shipment arriving at the northern port this week.
```

Now both junk rows are gone and the shortest surviving record is a real post.
Raise or lower the number and run it again until the shortest kept record
looks like something you would want an answer to cite.

Add `--json` if you want the numbers as data:

```bash
osintgpt add path/to/posts.csv --project demo-case \
  --map content=message --map min_chars=30 --dry-run --json
```

```json
{"file": "posts.csv", "records": 4, "kept": 2, "without_content": 1,
 "too_short": 1, "shortest": 63, "summary": "2 of 4 records, …"}
```

## 4. Register it

Drop `--dry-run` and fill in the other roles:

```bash
osintgpt add path/to/posts.csv --project demo-case \
  --map content=message \
  --map timestamp=published_at \
  --map author=channel \
  --map identity=post_id \
  --map metadata=url,views \
  --map min_chars=30
```

**Outcome:**

```
Source registered
project   demo-case
path      path/to/posts.csv
fields    {"content": ["message"], "metadata": ["url", "views"],
           "timestamp": "published_at", "author": "channel",
           "identity": "post_id", "min_chars": 30}
```

Then check and index:

```bash
osintgpt sources --project demo-case
osintgpt index --project demo-case
```

Each record now becomes one document, cited as `posts.csv#p1`, filterable by
date, and findable by channel name.

---

## Spreadsheets

XLSX works exactly the same. The first sheet is read, the first row is the
header:

```bash
osintgpt add path/to/dataset.xlsx --project demo-case \
  --map content=message --map timestamp=date --map author=channel_name \
  --map identity=signature --map min_chars=40 --dry-run
```

## Nested JSON

JSON usually wraps its records in something. Say `path/to/export.json` looks
like this:

```json
{"meta": {"collected": "2026-03-05"},
 "data": {"items": [
   {"id": "t1", "text": "First captured post about the port.",
    "user": {"handle": "alpha_news"}, "created": "2026-03-01T09:12:00Z"},
   ...
 ]}}
```

Map it without saying where the records are, and you get this:

```bash
osintgpt add path/to/export.json --project demo-case \
  --map content=text --dry-run
```

**Outcome:**

```
export.json — nothing registered
0 of 1 records, 1 with no content
```

One record, nothing kept. The whole file was treated as a single record, and
it has no top-level `text` field. Point `records` at the list, and use dots to
reach into nested fields:

```bash
osintgpt add path/to/export.json --project demo-case \
  --map records=data.items \
  --map content=text \
  --map author=user.handle \
  --map timestamp=created \
  --map identity=id --dry-run
```

**Outcome:**

```
export.json — nothing registered
2 of 3 records, 1 with no content, shortest 35 chars
shortest kept record:
First captured post about the port.
```

JSONL and NDJSON need no `records` — one record per line already.

## Two content fields

Sometimes the text really is split in two, like a headline and a body:

```bash
osintgpt add path/to/articles.csv --project demo-case \
  --map content=title,body --map timestamp=published --dry-run
```

They are joined with a blank line between them, in the order you name them.

**The first one is special.** It is the primary field: if it is empty, the
record is skipped, no matter what the others hold. So name the real body
first. Here `title` is first, which means an article with a title but no body
still gets indexed — probably not what you want. Better:

```bash
  --map content=body,title
```

---

## Mistakes to avoid

### Putting everything in `content`

It is tempting to select every field. Look at what happens:

```bash
osintgpt add path/to/posts.csv --project demo-case \
  --map content=channel,message,url --dry-run
```

**Outcome:**

```
posts.csv — nothing registered
4 of 4 records, shortest 35 chars
shortest kept record:
alpha_news

https://example.org/a/2
```

**4 of 4 kept looks better than 2 of 4. It is worse.** The empty-message row
survived because `channel` is filled, so it became a "document" made of a
channel name and a link. Index a hundred thousand rows like that and you get
thousands of near-identical records that crowd out real material in every
search.

A higher kept count is not the goal. Read the shortest kept record instead.

### Putting identifiers in `content`

Ids, urls and channel names repeat across every record. Embedding them makes
every record look a little more like every other record. Put them in
`metadata`, or in `author` and `identity` where they belong.

### Skipping `identity`

Without it, a record is cited by its row number. Reorder or re-export the
file and every citation moves. Name a unique field.

### Skipping `timestamp`

Without it, nothing can filter by date. "What was said last week" cannot be
answered. This is the one most often forgotten.

### Bad values

```bash
--map min_chars=forty
```

```
--map min_chars='forty' needs a whole number
```

---

## From Python

The same thing, as a library. Nothing here reads your environment — you pass
what you want.

### Look at the fields

```python
from osintgpt.ingestion import describe_fields

for name, facts in describe_fields('path/to/posts.csv').items():
    print(f'{name:14} filled {facts["filled"]}/{facts["sampled"]}  '
          f'avg {facts["average_length"]:>4}  unique {facts["unique"]}')
```

```
post_id        filled 4/4  avg    2  unique True
channel        filled 4/4  avg   10  unique False
message        filled 3/4  avg   46  unique True
published_at   filled 4/4  avg   25  unique True
url            filled 4/4  avg   23  unique True
views          filled 4/4  avg    3  unique True
```

### Test a mapping

```python
from osintgpt.ingestion import FieldMapping, preview_mapping

mapping = FieldMapping(
    content=('message',),
    timestamp='published_at',
    author='channel',
    identity='post_id',
    metadata=('url', 'views'),
    min_chars=30
)

preview = preview_mapping('path/to/posts.csv', mapping)
print(preview.summary)
print(preview.records, preview.kept, preview.without_content, preview.too_short)
```

```
2 of 4 records, 1 with no content, 1 under the floor, shortest 63 chars
4 2 1 1
```

`preview_mapping` reads every record in the file.

For several files that share a schema, `preview_files` gives one set of totals
across all of them, and the shortest record from whichever file holds it:

```python
from osintgpt.ingestion import preview_files

preview = preview_files(
    ['path/to/part1.csv', 'path/to/part2.csv'], mapping
)
print(preview.summary)
```

### See the documents

```python
from osintgpt.ingestion import load_documents

for document in load_documents('path/to/posts.csv', mapping):
    print(document.ref)
    print('  text    ', document.text[:50])
    print('  author  ', document.author)
    print('  time    ', document.timestamp)
    print('  metadata', document.metadata)
```

```
path/to/posts.csv#p1
  text     Report on the shipment arriving at the northern po
  author   alpha_news
  time     2026-03-01 09:12:00+00:00
  metadata {'url': 'https://example.org/a/1', 'views': '482'}
path/to/posts.csv#p4
  text     Longer analysis of the same shipment, with figures
  author   beta_watch
  time     2026-03-02 12:30:00+00:00
  metadata {'url': 'https://example.org/b/2', 'views': '1203'}
```

### Register and index

```python
from osintgpt import Project, Settings, index_project
from osintgpt.ingestion import Corpus
from osintgpt.llm import build_embedding_provider

project = Project.load('path/to/project')

Corpus.load(project.paths.sources).register('path/to/posts.csv', mapping)

config = project.settings_for(Settings.from_env())
embedder = build_embedding_provider(
    project.settings.embedding_provider,
    config,
    model=project.settings.embedding_model or None
)
report = index_project(project, embedder, config=config)
print(report.summary)
```

Registering the same path again replaces the old mapping rather than adding a
second one, so you can fix a mapping and re-register without cleaning up.

---

## The same thing in the app

Run `osintgpt app` and open **Material**.

1. **Point it at a folder.** Type or browse to the directory holding your
   files.

2. **It tells you what needs mapping.** A warning names how many structured
   files were found. Prose files — PDFs, Word, text — need nothing and are not
   listed.

3. **One panel per file shape.** If you have forty CSVs that all share the
   same columns, you fill this in once, not forty times. A file with different
   columns gets its own panel.

4. **The field table.** Each panel opens with every field listed — how often
   it is filled, how long it is, whether it is unique, and an example value.
   This is the same information the CLI prints when it refuses. Use it to pick.

5. **Assign the roles.** One control each:
   - *Content — the body*: the main text. Required.
   - *Also embed*: extra text fields, joined after the body.
   - *Timestamp*, *Author*, *Identity*: pick a field or leave blank.
   - *Metadata*: fields to keep but not embed.
   - *Minimum characters*: the floor.

6. **Watch the line underneath.** As soon as you pick a content field, it
   shows what the mapping would do — `2 of 4 records, 1 with no content,
   1 under the floor` — and prints the shortest record that would
   survive. Change a field or the floor and it updates immediately. If nothing
   would survive, it says so in red.

   This is the same check as `--dry-run`, over every file in the panel — files
   that share columns do not share contents, so all of them are counted. Use
   it the same way: adjust until the shortest kept record is something real.

   The files are re-read when you change the content field or the floor, the
   only two things that can move the numbers. Picking a timestamp or an author
   is instant. A panel holding several large spreadsheets takes a few seconds
   per file when you change one of those two.

7. **Register this folder.** Each file gets the mapping for its shape. The
   folder itself stays registered too, so prose files and anything added later
   are still picked up.

8. **Preview, then Index now.** The preview shows totals and the estimated
   cost before anything is embedded.

The app and the CLI write to the same place, so you can map a file in the app
and index it from the terminal, or the reverse.

---

## Short version

- Never guess. Run `--dry-run` first.
- Read the **shortest kept record**, not the kept count.
- One content field, usually. The body.
- Always set `identity` and `timestamp`.
- Ids and links go in `metadata`, not `content`.
- Use `min_chars` to cut signature-only rows.
