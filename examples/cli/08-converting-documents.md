# Converting documents

`convert` prints the Markdown a file is chunked from. It needs no project,
registers nothing, embeds nothing, and makes no network call unless
`--vision` is given. Use it to check what a reader extracts before paying to
index it.

## One file

```bash
osintgpt convert examples/data/generated/case/material/reports/incident-amber.html
```

```text
Incident report: Sorin-3 status amber

This report is invented and identifies no real system.

Timeline

At the end of cycle five, Sorin-3 stopped answering health checks for forty minutes. ...

Indicators

Status page: relay.example/status/88

Firmware image hash: 9f2c11ab

Account that edited the page: @vela_ops
incident-amber.html: read by text, 1 document(s)
```

The last line names the reader and the document count. Markup is gone, and
headings and list items survive as separate lines.

## Structured files

Structured files need the same `--map` roles as `add`; see
[`02-structured-data.md`](02-structured-data.md). Each kept record becomes a
section headed by its citation:

```bash
osintgpt convert examples/data/generated/case/material/intake/posts.csv \
  --map content=message --map identity=post_id --map min_chars=30
```

```text
## examples/data/generated/case/material/intake/posts.csv#p1

Vela-9 is back online after the cabinet repair; route MX-3 carries traffic again.

## examples/data/generated/case/material/intake/posts.csv#p4

Station Kestrel paid for the Neral-7 refit; the transfer used account KS-1180.
posts.csv: read by tabular, 2 document(s)
```

## A whole folder

With a directory, `--out` must name a directory; it is created if absent.
Each file is written as `<original name>.md`, keeping the folder structure:

```bash
osintgpt convert examples/data/generated/case/material/reports --out converted
```

```text
5 file(s) written to converted
```

This writes `converted/field-note-kestrel.md.md`,
`converted/incident-amber.html.md`, and so on. For a single file, `--out`
names the output file instead.

`--json` reports, per file, the reader, document count, PDF page counts
(`pages`, `transcribed_pages`, `empty_pages`), and the Markdown itself.

## PDFs and scanned pages

PDF text is extracted locally. Pages with no extractable text, usually
scans, come out as named gaps. `--vision` transcribes those pages with a
vision-capable generation model, one call per page:

```bash
osintgpt convert report.pdf --vision --project synthetic-relay-case
osintgpt convert report.pdf --model gpt-5.6-luna
```

`--model` picks the transcription model and implies `--vision`. Otherwise the
model is `ingestion_model`, falling back to `generation_model`, read from the
project given with `--project` or from user defaults. With `--project`, each
transcribed page is cached in the project's `extracts/` folder, so `index`
reuses it instead of paying for the page again; without it the cache is in
the osintgpt home. Set the model once with
`osintgpt config set ingestion_model MODEL_NAME`.

## Formats

Most formats are read by osintgpt's own readers. PowerPoint, Outlook `.msg`,
EPUB, and RSS/Atom feeds go through markitdown. Convert `.rtf`, `.odt`,
`.xls`, and non-feed `.xml` to a supported format before adding them; the
[README format table](../../README.md#supported-document-formats) explains why.
Images are embedded directly when the embedding model accepts images;
`convert` extracts no text from them.
