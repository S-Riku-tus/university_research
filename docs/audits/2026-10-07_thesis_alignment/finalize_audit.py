"""Export the task ledger and check the audit's local source references."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / "docs/research_plan/2026-10-07_thesis_alignment_and_priority_audit.md"


def main():
    report = REPORT.read_text(encoding="utf-8")
    rows = []
    for line in report.splitlines():
        if re.match(r"\| R\d{2} \|", line):
            values = [value.strip() for value in line.strip().strip("|").split("|")]
            if len(values) != 6:
                raise ValueError(f"Unexpected task fields: {values[0]}")
            rows.append(values)
    if [row[0] for row in rows] != [f"R{i:02}" for i in range(1, 19)]:
        raise ValueError("Task IDs do not cover R01 to R18 exactly once")
    with (OUT / "research_task_ledger.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "issue", "confirmed_status", "remaining_work", "priority_and_exit", "evidence"])
        writer.writerows(rows)

    linked = set(re.findall(r"\]\(([^)\n]+)\)", report))
    inventory, missing = [], []
    for target in sorted(linked):
        if "://" in target:
            continue
        path = (REPORT.parent / target.split("#", 1)[0]).resolve()
        exists = path.is_file()
        record = {"path": path.relative_to(ROOT).as_posix(), "role": "referenced_by_audit", "exists": exists}
        if exists:
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            missing.append(record["path"])
        inventory.append(record)
    primary = json.loads((OUT / "primary_source_inventory.json").read_text(encoding="utf-8"))
    changed_originals = []
    for record in primary:
        path = ROOT / record["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            changed_originals.append(record["path"])
    checks = {"tasks": len(rows), "local_references": len(inventory),
        "missing_references": missing, "changed_primary_sources": changed_originals,
        "coverage": "goal-indexed audit of relevant recorded sources; not a full data/log reanalysis"}
    (OUT / "source_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=True))
    if missing or changed_originals:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
