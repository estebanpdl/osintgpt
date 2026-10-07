"""Generate the synthetic, multi-format corpus used by the examples.

Usage:
    python examples/data/make_corpus.py
    python examples/data/make_corpus.py --output /tmp/osintgpt-example
"""

import argparse
import csv
import json
from pathlib import Path

from osintgpt import Question, save_questions


MARKDOWN = """# Relay ledger

The relay nodes in this corpus are invented and identify no real system.

## Cycle four

Neral-7 sent sequence LX-204 to Vela-9.

### Acknowledgement

Vela-9 acknowledged sequence LX-204 and forwarded packet QN-88 to Sorin-3.

## Cycle five

Sorin-3 returned status amber after receiving packet QN-88.
"""

TEXT = """Станция Вела-9 подтвердила последовательность LX-204.
أرسلت المحطة نيرال-7 الحزمة QN-88 في الدورة الخامسة.
节点索林-3在第六周期报告绿色状态。
"""

EVENTS = [
    {
        "record_id": "EV-001",
        "summary": "Koru-5 relayed packet AR-12 to Neral-7.",
        "script": "Latin",
        "reported_at": "2042-05-04T09:15:00Z",
    },
    {
        "record_id": "EV-002",
        "summary": "Узел Сорин-3 получил пакет AR-12.",
        "script": "Cyrillic",
        "reported_at": "2042-05-04T09:18:00Z",
    },
    {
        "record_id": "EV-003",
        "summary": "أكدت العقدة نيرال-7 استلام الحزمة AR-12.",
        "script": "Arabic",
        "reported_at": "2042-05-04T09:21:00Z",
    },
]

MESSAGES = [
    {"sequence": 1, "script": "Latin", "message": "Vela-9 opened route MX-3."},
    {"sequence": 2, "script": "Cyrillic", "message": "Кору-5 закрыл маршрут MX-3."},
    {"sequence": 3, "script": "Arabic", "message": "أعاد سورين-3 فتح المسار MX-3."},
]

# Analyst reports written so that meaning and literal identifiers point to
# different documents: semantic, exact, and fused retrieval each have
# something only they find.
REPORTS = {
    "field-note-kestrel.md": """# Field note: Station Kestrel

Everything in this note is invented and identifies no real place or person.

## Who paid for the northern refit

Station Kestrel underwrote the refit of the northern relay Neral-7 during
cycle three. The money moved through account KS-1180, which the ledger lists
under "maintenance reserve" rather than under capital spending. Two operators
described the arrangement as a favour returned for earlier routing support.

## How the money was described

Nobody at Kestrel used the word funding. Internal messages call the transfer
"line support" and "a loan of capacity". The amounts match the refit invoice
to the unit, which is why the analyst treats the two as the same payment.

## Open questions

- Whether account KS-1180 also covered the antenna work at Vela-9.
- Whether the handle @kestrel_ops belongs to the station or to one operator.
""",
    "incident-amber.html": """<html><head><title>Incident report: Sorin-3</title></head>
<body>
<h1>Incident report: Sorin-3 status amber</h1>
<p>This report is invented and identifies no real system.</p>
<h2>Timeline</h2>
<p>At the end of cycle five, Sorin-3 stopped answering health checks for \
forty minutes. The node came back reporting status amber and a queue of \
unsent packets, including QN-88.</p>
<h2>Cause</h2>
<p>The outage followed a power fault in the cabinet that also houses the \
backup modem. The status page relay.example/status/88 was edited twice \
during the outage; the second edit removed the mention of the modem.</p>
<h2>Indicators</h2>
<ul>
<li>Status page: relay.example/status/88</li>
<li>Firmware image hash: 9f2c11ab</li>
<li>Account that edited the page: @vela_ops</li>
</ul>
</body></html>
""",
    "interview-transcript.txt": """Interview transcript, operator of Koru-5 (invented)

Q: The event log says Koru-5 relayed packet AR-12 to Neral-7. Is that right?
A: No. Koru-5 did not relay AR-12. Our node was in maintenance that morning.
The entry was written by someone else's console.

Q: Who could write to that log?
A: Anyone with the shared key. Vela-9 and Sorin-3 both had it in cycle four.

Q: Did you close route MX-3?
A: I closed it after the second warning. It was reopened later without
asking me, and I learned about it from the message board.

Q: Have you been paid by Station Kestrel?
A: Never directly. I heard they paid for the Neral-7 refit.
""",
    "procurement-memo.md": """# Procurement memo: replacement antennas

This memo is invented and identifies no real supplier or order.

## Request

Vela-9 asked for three replacement antennas after the storm in cycle two.
The request was approved under purchase order PO-7731.

## Supplier

The order went to a single supplier without a second quote. The memo gives
urgency as the reason. Delivery took eleven days, and the invoice was paid
from the same reserve that later covered the Neral-7 refit.

## Follow-up

Check whether PO-7731 appears in the Kestrel accounts.
""",
    "nota-operativa.md": """# Nota operativa: ruta MX-3

Esta nota es inventada y no identifica ningún sistema real.

## Cierre de la ruta

El operador de Koru-5 cerró la ruta MX-3 después de dos avisos de
congestión. Según la nota, el cierre buscaba proteger la cola de Sorin-3
mientras se reparaba el gabinete.

## Reapertura

La ruta MX-3 se reabrió desde Sorin-3 sin consultar al operador de Koru-5.
Nadie dejó constancia de quién autorizó la reapertura.
""",
}

POSTS = [
    {
        "post_id": "p1",
        "channel": "vela_watch",
        "message": "Vela-9 is back online after the cabinet repair; route MX-3 "
                   "carries traffic again.",
        "published_at": "2042-05-05 08:10",
        "url": "https://relay.example/p/1",
        "views": "482",
    },
    {
        "post_id": "p2",
        "channel": "vela_watch",
        "message": "",
        "published_at": "2042-05-05 09:02",
        "url": "https://relay.example/p/2",
        "views": "150",
    },
    {
        "post_id": "p3",
        "channel": "sorin_notes",
        "message": "@vela_ops",
        "published_at": "2042-05-05 10:44",
        "url": "https://relay.example/p/3",
        "views": "97",
    },
    {
        "post_id": "p4",
        "channel": "sorin_notes",
        "message": "Station Kestrel paid for the Neral-7 refit; the transfer "
                   "used account KS-1180.",
        "published_at": "2042-05-06 12:30",
        "url": "https://relay.example/p/4",
        "views": "1203",
    },
]

EXPORT = {
    "meta": {"collected": "2042-05-07", "tool": "invented-collector"},
    "data": {
        "items": [
            {
                "id": "t1",
                "text": "Sorin-3 reported amber again after the power fault.",
                "user": {"handle": "sorin_notes"},
                "created": "2042-05-06T07:12:00Z",
            },
            {
                "id": "t2",
                "text": "",
                "user": {"handle": "vela_watch"},
                "created": "2042-05-06T07:40:00Z",
            },
            {
                "id": "t3",
                "text": "Purchase order PO-7731 was paid from the maintenance "
                        "reserve.",
                "user": {"handle": "kestrel_ops"},
                "created": "2042-05-06T09:05:00Z",
            },
        ]
    },
}

WATCHLIST = """# One literal term per line. Blank lines and lines starting with # are skipped.
LX-204
QN-88
KS-1180
PO-7731
@vela_ops
relay.example/status/88
9f2c11ab
"""

# Expected files are relative to the case directory; notes say what each probes.
QUESTIONS = [
    ("Which node acknowledged sequence LX-204?",
     ["material/prose/relay-ledger.md"], ["LX-204"],
     "identifier and wording both match"),
    ("Какая станция подтвердила последовательность LX-204?",
     ["material/prose/dispatches.txt"], ["LX-204"],
     "Cyrillic question, Cyrillic answer"),
    ("أي محطة أرسلت الحزمة QN-88؟",
     ["material/prose/dispatches.txt"], ["QN-88"],
     "Arabic question, Arabic answer"),
    ("Who paid for the hardware upgrade at the northern relay?",
     ["material/reports/field-note-kestrel.md"], ["Kestrel"],
     "paraphrase: the note says underwrote and refit"),
    ("Why did Sorin-3 stop answering health checks?",
     ["material/reports/incident-amber.html"], ["Sorin-3"],
     "HTML source"),
    ("Which purchase order covered the replacement antennas?",
     ["material/reports/procurement-memo.md"], ["PO-7731"],
     "identifier the question does not contain"),
    ("Did the Koru-5 operator deny relaying packet AR-12?",
     ["material/reports/interview-transcript.txt"], ["AR-12"],
     "negated claim"),
    ("¿Por qué se cerró la ruta MX-3?",
     ["material/reports/nota-operativa.md"], ["MX-3"],
     "Spanish question, Spanish answer"),
]


def question_set(root: Path):
    """Resolve expected files to the refs indexing stores for them.

    Material outside a project is stored under its absolute path, so the
    question set is only valid for the directory it was generated into.
    """
    return [
        Question(
            text=text,
            expected=[(root / ref).resolve().as_posix() for ref in expected],
            terms=terms,
            note=note
        )
        for text, expected, terms, note in QUESTIONS
    ]


def write_corpus(root: Path) -> int:
    """Write every generated document beneath ``root``; return the file count."""
    prose = root / "material" / "prose"
    records = root / "material" / "records"
    reports = root / "material" / "reports"
    intake = root / "material" / "intake"
    for folder in (prose, records, reports, intake):
        folder.mkdir(parents=True, exist_ok=True)

    (prose / "relay-ledger.md").write_text(MARKDOWN, encoding="utf-8")
    (prose / "dispatches.txt").write_text(TEXT, encoding="utf-8")

    with (records / "events.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENTS[0])
        writer.writeheader()
        writer.writerows(EVENTS)

    with (records / "messages.jsonl").open("w", encoding="utf-8") as stream:
        for message in MESSAGES:
            stream.write(json.dumps(message, ensure_ascii=False) + "\n")

    for name, text in REPORTS.items():
        (reports / name).write_text(text, encoding="utf-8")

    with (intake / "posts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=POSTS[0])
        writer.writeheader()
        writer.writerows(POSTS)

    (intake / "export.json").write_text(
        json.dumps(EXPORT, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (root / "watchlist.txt").write_text(WATCHLIST, encoding="utf-8")
    save_questions(root / "questions.toml", question_set(root))

    return 6 + len(REPORTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic corpus used by the examples."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/data/generated/case"),
        help="destination directory",
    )
    arguments = parser.parse_args()
    files = write_corpus(arguments.output)
    print(
        f"Wrote {files} material files, {len(QUESTIONS)} questions, "
        f"and a watchlist to {arguments.output}"
    )


if __name__ == "__main__":
    main()
