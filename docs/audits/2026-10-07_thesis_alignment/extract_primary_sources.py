"""Read only selected original planning/report paragraphs and slide text."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = [
    ("研究進捗報告/研究計画の基本となる考え方_柴崎陸.docx", "planning", None),
    ("研究進捗報告/2026_AE＿植木研＿8125517＿柴崎陸.docx", "formal_plan", None),
    ("研究進捗報告/2026/925/研究進捗報告_柴崎陸_9月25日.docx", "submitted_report", None),
    ("研究進捗報告/2026/918（研究計画発表）/研究進捗_柴崎陸_0918.pptx", "presented_plan", [3, 4, 5, 8]),
    ("研究進捗報告/2026/724（中間発表）/研究進捗_柴崎陸_0724.pptx", "interim_plan", [3, 4, 5, 11, 12]),
]
KEYWORDS = re.compile("目的|題目|課題|目標|今後|計画|アンサンブル|沸騰|ノイズ|説明|検知|研究内容|ONB|精度|回帰|学習|分割|重み")


def main():
    records, text = [], ["# 研究目的・発表・提出報告の原資料抜粋", "", "2026-10-07。対象を索引から選び、計画・目的・課題に関係する本文/指定スライドだけを抽出。教授の口頭発言の記録とは区別する。", ""]
    for relative, role, slides in SOURCES:
        path = ROOT / relative
        record = {"path": relative, "role": role, "exists": path.is_file()}
        if not path.is_file():
            records.append(record)
            continue
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            text.extend(["## "+relative, ""])
            if slides is None:
                root = ET.fromstring(archive.read("word/document.xml"))
                namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", namespace)).strip()
                    for p in root.findall(".//w:p", namespace)]
                matched = [(i+1, value) for i, value in enumerate(paragraphs) if value and KEYWORDS.search(value)]
                record.update(total_paragraphs=len(paragraphs), selected_paragraph_numbers=[i for i, _ in matched])
                text += [f"- 段落{i}: {value}" for i, value in matched]
            else:
                namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                existing = []
                for page in slides:
                    member = f"ppt/slides/slide{page}.xml"
                    if member not in archive.namelist():
                        continue
                    existing.append(page)
                    root = ET.fromstring(archive.read(member))
                    paragraphs = ["".join(t.text or "" for t in p.findall(".//a:t", namespace)).strip()
                        for p in root.findall(".//a:p", namespace)]
                    text.extend([f"### スライド{page}", "", *["- "+value for value in paragraphs if value]])
                record["selected_slide_numbers"] = existing
            text.append("")
        records.append(record)
    (OUT / "primary_source_extracts.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    (OUT / "primary_source_inventory.json").write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in record.items() if k != "sha256"} for record in records], ensure_ascii=True))


if __name__ == "__main__":
    main()
