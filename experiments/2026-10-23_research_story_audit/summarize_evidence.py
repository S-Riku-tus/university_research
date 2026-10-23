"""Read existing summaries for the research explanation; no model fit/inference."""
from __future__ import annotations

import csv
import hashlib
import json
import posixpath
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / "experiments/2026-10-15_clean_only_result_analysis"
CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metrics_path = MAIN / "pooled_metrics.csv"
    onset_path = MAIN / "recorded_onb_point_detection.csv"
    metrics = read_csv(metrics_path)
    onsets = read_csv(onset_path)
    by_key = {(r["noise"], r["model_key"]): r for r in metrics}
    onset_by_key = {(r["noise"], r["model_key"]): r for r in onsets}
    assert len(metrics) == len(by_key) == 77
    assert len(onsets) == len(onset_by_key) == 77
    weight_rows = read_csv(MAIN / "weights.csv")
    saved_weights = {(r["strategy_name"], r["model_key"]): float(r["weight"]) for r in weight_rows}
    core_members = ("randomforest", "conformer", "alexnet")
    assert all(abs(saved_weights["original3_hgb_extra_trees", k] - .25 * saved_weights["original3_mse", k]) < 1e-12
               for k in core_members)
    clean_gap = float(by_key["clean", CORE]["rmse_all_kW_m2"]) - float(
        by_key["clean", FIVE]["rmse_all_kW_m2"])
    comparisons = []
    for noise in NOISES:
        core, five = by_key[noise, CORE], by_key[noise, FIVE]
        for row in (core, five):
            assert [int(float(row[k])) for k in ("n", "n_pre_onb", "n_post_onb", "n_onb", "n_source_wavs")] == [540, 225, 315, 30, 36]
        result = {"noise": noise}
        for label, key, row in (("core3", CORE, core), ("five", FIVE, five)):
            result[label] = {
                "rmse_kW_m2": float(row["rmse_all_kW_m2"]),
                "near_rmse_kW_m2": float(row["rmse_onb_kW_m2"]),
                "fn": int(float(row["fn"])),
                "fp": int(float(row["fp"])),
                "recorded_onb_detected": int(onset_by_key[noise, key]["detected_onb_chunks"]),
                "recorded_onb_n": int(onset_by_key[noise, key]["n_at_recorded_onb"]),
                "degradation_from_clean_kW_m2": float(row["rmse_all_kW_m2"]) - float(by_key["clean", key]["rmse_all_kW_m2"]),
            }
        gap = result["core3"]["rmse_kW_m2"] - result["five"]["rmse_kW_m2"]
        degradation_reduction = result["core3"]["degradation_from_clean_kW_m2"] - result["five"]["degradation_from_clean_kW_m2"]
        assert abs(degradation_reduction - (gap - clean_gap)) < 1e-12
        result.update(absolute_rmse_reduction_kW_m2=gap,
                      degradation_reduction_kW_m2=degradation_reduction)
        comparisons.append(result)

    previous_path = ROOT / "experiments/2026-10-21_research_logic_audit/verification.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    cnn_path = ROOT / "experiments/2026-10-22_cnn_three_tuning_progress/status_135525.json"
    cnn = json.loads(cnn_path.read_text(encoding="utf-8"))
    selection_path = ROOT / "experiments/2026-10-22_resnet50_parameter_selection/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    deck = ROOT / "研究進捗報告/2026/1016（研究進捗報告）/研究進捗_1016_編集用_修正版.pptx"
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
          "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    with zipfile.ZipFile(deck) as archive:
        slide_names = [s for s in archive.namelist() if s.startswith("ppt/slides/slide") and s.endswith(".xml")]
        slides = sorted(slide_names, key=lambda s: int(s.rsplit("slide", 1)[1].split(".")[0]))
        assert len(slides) == 14
        extracted_text = []
        for i, slide in enumerate(slides, 1):
            text = " | ".join(t.text or "" for t in ET.fromstring(archive.read(slide)).findall(".//a:t", ns))
            extracted_text.append({"slide": i, "text": text})
        relations = ET.fromstring(archive.read("ppt/slides/_rels/slide13.xml.rels"))
        image_by_id = {r.attrib["Id"]: posixpath.normpath(posixpath.join("ppt/slides", r.attrib["Target"]))
                       for r in relations if r.attrib["Type"].endswith("/image")}
        pictures = []
        for picture in ET.fromstring(archive.read("ppt/slides/slide13.xml")).findall(".//p:pic", ns):
            ext = picture.find(".//a:xfrm/a:ext", ns)
            blip = picture.find(".//a:blip", ns)
            if ext is not None and blip is not None:
                target = image_by_id[blip.attrib["{" + ns["r"] + "}embed"]]
                pictures.append((int(ext.attrib["cx"]) * int(ext.attrib["cy"]), target))
        assert pictures
        largest_area = max(area for area, _ in pictures)
        outline_images = []
        for area, target in pictures:
            if area < largest_area / 4:
                continue
            output = OUT / (f"slide13_outline_{len(outline_images) + 1}" + Path(target).suffix)
            output.write_bytes(archive.read(target))
            outline_images.append(output.relative_to(ROOT).as_posix())

    sources = [metrics_path, onset_path, previous_path, cnn_path, selection_path, deck,
               MAIN / "conditions.json", MAIN / "weights.csv",
               ROOT / "code/utils/dataloading/acoustic_summary_features.py",
               ROOT / "code/utils/ensemble/fixed_core_stacking.py",
               ROOT / "LaTeX/master_thesis/chapters/01_introduction.tex",
               ROOT / "LaTeX/master_thesis/chapters/06_conclusions.tex",
               ROOT / "LaTeX/master_thesis/references.tex"]
    result = {
        "created_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "scope": "Existing summary CSV reading, explanatory arithmetic, and current slide XML/image extraction. Prior raw-prediction audit is cited, not rerun.",
        "new_model_fit_or_inference": False,
        "raw_prediction_recomputation": False,
        "raw_arrays_loaded": False,
        "main_run": {k: previous[k] for k in ("run_hash", "fit_id", "train_n", "evaluation_n", "source_wavs", "shared_source_wavs", "train_evaluation_same_chunk_overlap", "onset_evaluation_n")},
        "comparisons": comparisons,
        "checks": {
            "all_7_rmse_lower_than_core3": all(r["absolute_rmse_reduction_kW_m2"] > 0 for r in comparisons),
            "all_7_fn_lower_than_core3": all(r["five"]["fn"] < r["core3"]["fn"] for r in comparisons),
            "all_7_fp_same_as_core3": all(r["five"]["fp"] == r["core3"]["fp"] for r in comparisons),
            "near_rmse_improved_conditions": [r["noise"] for r in comparisons if r["five"]["near_rmse_kW_m2"] < r["core3"]["near_rmse_kW_m2"]],
            "all_6_noise_degradation_greater_than_core3": all(r["degradation_reduction_kW_m2"] < 0 for r in comparisons[1:]),
            "gap_degradation_identity_verified_conditions": len(comparisons),
            "saved_five_core_weights_equal_0p25_times_core3_mse": True,
        },
        "cnn_status_is_historical_snapshot": True,
        "cnn_status_at": cnn["checked_at_jst"],
        "cnn_saved_completed_counts": {k: v["completed"] for k, v in cnn["models"].items()},
        "resnet_selection_at": selection["checked_at_jst"],
        "resnet_selected_parameters": selection["selected"]["parameters"],
        "slide_count": len(slides),
        "slide13_images": outline_images,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
    }
    (OUT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "slide_text.json").write_text(json.dumps(extracted_text, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": result["checks"], "slide13_images": result["slide13_images"],
                      "clean_gap": comparisons[0]["absolute_rmse_reduction_kW_m2"],
                      "minus20_gap": comparisons[-1]["absolute_rmse_reduction_kW_m2"],
                      "minus20_degradation_reduction": comparisons[-1]["degradation_reduction_kW_m2"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
