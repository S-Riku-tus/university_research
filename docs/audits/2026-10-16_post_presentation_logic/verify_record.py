"""Record selected purpose evidence and verify the post-presentation documents."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    primary = [
        ("研究進捗報告/研究計画の基本となる考え方_柴崎陸.docx", [11]),
        ("研究進捗報告/2026_AE＿植木研＿8125517＿柴崎陸.docx", [38, 39, 40, 41]),
        ("研究進捗報告/2026/522（研究計画発表）/研究進捗_柴崎陸_0522.pptx", [6, 7]),
        ("研究進捗報告/2026/619（研究計画発表）/研究進捗_柴崎陸_0619.pptx", [4, 5]),
        ("研究進捗報告/2026/724（中間発表）/研究進捗_柴崎陸_0724.pptx", [4, 5, 11]),
    ]
    evidence = []
    for raw, positions in primary:
        path = ROOT / raw
        before = sha(path)
        excerpts = []
        with zipfile.ZipFile(path) as z:
            if path.suffix == ".docx":
                doc = ET.fromstring(z.read("word/document.xml"))
                paras = ["".join(t.text or "" for t in q.findall(".//w:t", NS))
                         for q in doc.findall(".//w:p", NS)]
                excerpts = [{"paragraph": i, "text": paras[i - 1]} for i in positions]
            else:
                doc = ET.fromstring(z.read("ppt/presentation.xml"))
                rels = {e.get("Id"): e.get("Target") for e in
                        ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))}
                ids = doc.findall("p:sldIdLst/p:sldId", NS)
                for i in positions:
                    target = rels[ids[i - 1].get("{" + NS["r"] + "}id")]
                    part = posixpath.normpath(posixpath.join("ppt", target))
                    page = ET.fromstring(z.read(part))
                    text = "\n".join("".join(t.text or "" for t in q.findall(".//a:t", NS))
                                     for q in page.findall(".//a:p", NS))
                    excerpts.append({"slide": i, "text": text})
        assert sha(path) == before
        evidence.append({"path": raw, "sha256": before, "unchanged": True,
                         "selected_excerpts": excerpts})
    secondary = [
        "docs/research_plan/2026-05-22_after_presentation_concise.md",
        "docs/research_plan/2026-06-19_after_presentation.md",
        "docs/research_plan/2026-07-24_current_progress_and_priorities.md",
        "docs/research_plan/2026-09-18_professor_discussion.md",
        "experiments/2026-10-15_clean_only_result_analysis/README.md",
        "experiments/2026-10-14_error_feature_linkage/README.md",
        "experiments/2026-10-07_onb_input_and_integration/README.md",
    ]
    evidence.extend({"path": p, "sha256": sha(ROOT / p),
                     "role": "prior interpretation or results; not assumed verbatim professor speech"}
                    for p in secondary)
    (OUT / "source_evidence.json").write_text(
        json.dumps({"primary_and_supporting_sources": evidence}, ensure_ascii=False, indent=2)
        + "\n", encoding="utf-8")
    record = ROOT / "docs/research_plan/2026-10-16_after_presentation_professor_discussion.md"
    draft = ROOT / "docs/thesis/2026-10-16_purpose_and_research_logic_draft.md"
    notes = [
        "実験を出ていないなら二章は書くべきではないので，付録で書くとか誰かのデータを使ったとかを明記する．そこら辺のスタンスを理解する",
        "予測誤差で見るのは良いが，目的から考えてやりたい",
        "目的から考えたほうが良い．流れを整理してほしい．夏休み前にやっている気がしているのでそれも踏まえて",
        "代わり映えしてるのかが分からなかったのでその理由とかロジックを把握したい",
    ]
    text = record.read_text(encoding="utf-8")
    assert all("> " + note in text for note in notes)
    checked = 0
    missing = []
    for doc in [record, draft]:
        for raw in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8-sig")):
            if raw.startswith(("http:", "https:", "#")):
                continue
            checked += 1
            if not (doc.parent / unquote(raw.split("#")[0])).exists():
                missing.append([str(doc.relative_to(ROOT)), raw])
    assert not missing, missing
    verification = {
        "checked_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "presentation_completed": "reported by user",
        "professor_notes_verbatim_preserved": 4,
        "primary_documents_checked": 5,
        "primary_documents_unchanged": True,
        "purpose_history_checked": ["2026-05-22", "2026-06-19", "2026-07-24", "2026-09-18"],
        "new_documents": [str(p.relative_to(ROOT)).replace("\\", "/") for p in [record, draft]],
        "new_document_links_checked": checked,
        "broken_document_links": 0,
        "data_acquisition_roles": {
            "user_involved_in_2025_june11_june18_experiment_or_recording": False,
            "experimenter_known_to_user": False,
            "acquisition_source_identity": "unconfirmed",
            "analysis_roles_not_inferred_from_this_reply": True,
        },
        "research_narrative_still_requires_revision": True,
        "completed_numerical_analyses_remain_complete": True,
        "new_model_fits": 0,
        "new_inference": 0,
        "new_input_interventions": 0,
        "pptx_edited": False,
        "old_raw_materials_preserved": True,
    }
    (OUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    for path in [record, draft]:
        print("\nREADBACK", str(path.relative_to(ROOT)))
        print("\n".join(path.read_text(encoding="utf-8").splitlines()[:5]))


if __name__ == "__main__":
    main()
