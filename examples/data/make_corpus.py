"""Generate the synthetic, multi-format corpus used by the examples.

Usage:
    python examples/data/make_corpus.py
    python examples/data/make_corpus.py --output /tmp/osintgpt-example
"""

import argparse
import csv
import json
from pathlib import Path


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

QUESTIONS = """# Questions whose answer documents are known.

[[question]]
text = "Which node acknowledged sequence LX-204?"
expected = ["material/prose/relay-ledger.md"]

[[question]]
text = "Какая станция подтвердила последовательность LX-204?"
expected = ["material/prose/dispatches.txt"]

[[question]]
text = "أي محطة أرسلت الحزمة QN-88؟"
expected = ["material/prose/dispatches.txt"]
"""


def write_corpus(root: Path) -> None:
    """Write every generated document beneath ``root``."""
    prose = root / "material" / "prose"
    records = root / "material" / "records"
    prose.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)

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

    (root / "questions.toml").write_text(QUESTIONS, encoding="utf-8")


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
    write_corpus(arguments.output)
    print(f"Wrote 4 documents and 3 questions to {arguments.output}")


if __name__ == "__main__":
    main()
