"""Index research reports and reuse unchanged existing Office text extracts.

Only changed/new Office files are parsed; images and duplicate PDF exports
are not loaded. Date folders are source labels, not inferred event dates.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REPORTS = ROOT / "研究進捗報告"
PRIOR = ROOT / "docs/audits/2026-09-15_document_consistency/after_final"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def office_text(path):
    blocks = []
    with zipfile.ZipFile(path) as archive:
        if path.suffix == ".docx":
            xml = ET.fromstring(archive.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for i, paragraph in enumerate(xml.findall(".//w:p", ns), 1):
                text = "".join(t.text or "" for t in paragraph.findall(".//w:t", ns)).strip()
                if text:
                    blocks.append({"location": "paragraph " + str(i), "text": text})
        else:
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            names = sorted((name for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                key=lambda name: int(re.search(r"slide(\d+)", name).group(1)))
            for name in names:
                xml = ET.fromstring(archive.read(name))
                paragraphs = ["".join(t.text or "" for t in p.findall(".//a:t", ns))
                    for p in xml.findall(".//a:p", ns)]
                text = "\n".join(p for p in paragraphs if p)
                blocks.append({"location": name, "text": text})
    return "\n".join(b["location"] + "\n" + b["text"] for b in blocks), blocks


def main():
    cache = {row["path"]: row["text"] for row in
        (json.loads(line) for line in (PRIOR / "extracted_text.jsonl").open(encoding="utf-8"))}
    with (PRIOR / "document_catalog.csv").open(encoding="utf-8-sig", newline="") as stream:
        hashes = {row["path"]: row["sha256"] for row in csv.DictReader(stream)}
    inventory, records = [], []
    allowed_roots = {"2024", "2025", "2026", "原子力学会"}
    formal_names = {"2025_AE＿植木研＿8125517＿柴崎陸.docx",
        "2026_AE＿植木研＿8125517＿柴崎陸.docx", "研究計画の基本となる考え方_柴崎陸.docx",
        "卒論_柴崎陸.pdf", "8121048_柴崎陸_最終投稿卒論.pdf"}
    for path in sorted(REPORTS.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".docx", ".pptx", ".pdf", ".md", ".txt"}:
            continue
        rel = path.relative_to(REPORTS)
        if rel.parts[0] not in allowed_roots and path.name not in formal_names:
            continue
        key = path.relative_to(ROOT).as_posix()
        current_hash = digest(path)
        role = "research_report_or_presentation"
        if "論文輪講" in key or "自己紹介" in key:
            role = "literature_review_or_self_introduction"
        elif path.name in formal_names:
            role = "formal_plan_or_thesis"
        if path.suffix.lower() == ".pdf":
            mode = "duplicate_pdf_export" if path.with_suffix(".docx").exists() else "pdf_index_only"
            # Reuse the existing PDF extraction when the exact source is unchanged.
            if mode != "duplicate_pdf_export" and key in cache and hashes.get(key) == current_hash:
                mode = "cached_pdf_text"
                records.append({"path": key, "role": role, "extraction": mode,
                    "sha256": current_hash, "text": cache[key]})
        elif path.suffix.lower() in {".md", ".txt"}:
            mode = "local_text"
            records.append({"path": key, "role": role, "extraction": mode,
                "sha256": current_hash, "text": path.read_text(encoding="utf-8-sig", errors="replace")})
        else:
            mode = "cached_office_text"
            blocks = None
            if key in cache and hashes.get(key) == current_hash:
                text = cache[key]
            else:
                mode = "new_or_changed_office_xml"
                text, blocks = office_text(path)
            records.append({"path": key, "role": role, "extraction": mode,
                "sha256": current_hash, "text": text, "blocks": blocks})
        inventory.append({"path": key, "role": role, "format": path.suffix.lower(),
            "bytes": path.stat().st_size, "sha256": current_hash, "inspection": mode})
    with (OUT / "source_inventory.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    with (OUT / "source_text.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    topics = {
        "ONB_and_early_detection": ["ONB", "沸騰開始", "核沸騰", "発生検知", "早期検知"],
        "noise_anomaly_and_robustness": ["ノイズ", "雑音", "SNR", "noise", "頑健"],
        "ensemble_and_complementarity": ["アンサンブル", "統合", "多様性", "補完", "ensemble"],
        "physical_features_and_explanations": ["説明性", "物理", "周波数", "SHAP", "Grad", "Attention", "PCA"],
        "evaluation_scope_and_labels": ["別日", "汎化", "一般化", "ラベル", "正規化", "回帰", "R2", "R^2", "決定係数"],
        "future_candidates": ["CWT", "SWT", "WaveNet", "wavenet", "Efficient", "サブクール", "細線", "実時間", "リアルタイム"]}
    evidence = []
    for record in records:
        normalized = re.sub(r"\s+", "", record["text"])
        for topic, terms in topics.items():
            pattern = re.compile("|".join(map(re.escape, terms)), re.IGNORECASE)
            matches = list(pattern.finditer(normalized))
            if matches:
                windows = []
                for match in matches[:3]:
                    window = normalized[max(0, match.start() - 90):match.end() + 180]
                    if window not in windows:
                        windows.append(window)
                evidence.append({"path": record["path"], "role": record["role"],
                    "topic": topic, "keyword_hits": len(matches), "source_excerpt": " | ".join(windows)})
    with (OUT / "topic_evidence.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    counts = {mode: sum(row["inspection"] == mode for row in inventory)
        for mode in sorted({row["inspection"] for row in inventory})}
    summary = {"record_date_jst": "2026-10-14", "inventory_count": len(inventory),
        "text_record_count": len(records), "extraction_counts": counts,
        "topic_evidence_rows": len(evidence), "all_extracted_texts_keyword_surveyed": True,
        "scope": "2024-2026 dated research folders, authored conference material, formal plans and graduation thesis; course material and unrelated papers excluded.",
        "limitations": "Office text/notes and existing PDF text only; embedded plots and visual layout are not re-rendered. Folder years/dates are not assumed to be event dates. Report text does not establish submission or teacher approval."}
    (OUT / "survey_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
