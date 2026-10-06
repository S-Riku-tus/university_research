"""Freeze model inventory and verify the selected three-model integration."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / "experiments/2026-10-06_chunk_complementary_model_pilots"
FOLLOWUP = ROOT / "experiments/2026-10-06_hgb_followup_validation"


def read(name):
    with name.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    pilot = {(r["method"], r["noise"]): r for r in read(PRIOR / "outer_metrics.csv") if r["day"] == "two_day"}
    oof = {(r["method"], r["noise"]): r for r in read(PRIOR / "candidate_metrics_all.csv") if r["day"] == "two_day"}
    inventory = []
    for representation in ["band10", "frequency34", "temporal46"]:
        for learner in ["SVR", "HGB"]:
            method = f"{representation}_{learner}_rmse"
            clean, noisy = pilot[(method, "clean")], pilot[(method, "-20")]
            inventory.append({"method": method, "evidence_stage": "seed42 developmental pilot; RMSE selection",
                **{f"clean_{key}": clean[key] for key in ["rmse", "mae", "fn", "fp"]},
                **{f"minus20_{key}": noisy[key] for key in ["rmse", "fn", "fp"]},
                "clean_oof_rmse": oof[(method, "clean")]["rmse"], "minus20_oof_fp": oof[(method, "-20")]["fp"],
                "paired_seeds43_44_available": method == "frequency34_HGB_rmse"})
    write("pilot_model_inventory.csv", inventory)
    summaries = read(FOLLOWUP / "replication_summary.csv")
    write("paired_method_profiles.csv", summaries)
    matrices = {}
    with (FOLLOWUP / "outer_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["method"] in ["ca_hgb_clean_mse", "hgb4_clean_mse"]:
                key = (row["seed"], row["noise"], row["source_wav_id"], row["chunk_index"])
                matrices.setdefault(key, {})[row["method"]] = (float(row["prediction_kW_m2"]), float(row["onb_kW_m2"]))
    differences = []
    for predictions in matrices.values():
        primary, threshold = predictions["ca_hgb_clean_mse"]
        four, other_threshold = predictions["hgb4_clean_mse"]
        assert threshold == other_threshold
        assert (primary >= threshold) == (four >= threshold)
        differences.append(abs(primary - four))
    assert len(differences) == 2 * 7 * 540
    assert max(differences) < .002
    config = ROOT / "configs/experiments/2026-10-06_model_adoption_decision.json"
    decision = json.loads(config.read_text(encoding="utf-8"))
    assert decision["primary"]["method"] in {r["method"] for r in summaries}
    verification = {"status": "passed", "primary_vs_four_model_prediction_pairs": len(differences),
        "max_prediction_difference_kW_m2": max(differences), "binary_decisions_identical": True,
        "pilot_inventory_rows": len(inventory), "paired_method_profile_rows": len(summaries),
        "new_model_training": False, "manual_evidence_requests_removed": True,
        "decision_config": str(config.relative_to(ROOT))}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(verification)


if __name__ == "__main__":
    main()
