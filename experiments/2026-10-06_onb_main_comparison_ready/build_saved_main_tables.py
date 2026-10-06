"""Finalize the main tables from fixed saved seed43/44; never retrain."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.calculation.source_day_metrics import source_day_metric_rows
from utils.calculation.onb_comparison_report import write_comparison_report

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/2026-10-06_fixed_three_additions"
PAIRED = ROOT / "experiments/2026-10-06_hgb_followup_validation"
PROTOCOL = ROOT / "configs/experiments/2026-10-06_onb_main_comparison_protocol.json"
MAPPING = {"original3_mse": "existing3_clean_mse", "original3_performance": "existing_performance",
           "original3_hgb": "existing3_clean_mse__hgb4_anchor",
           "original3_extra_trees": "existing3_clean_mse__extra_trees4_anchor",
           "original3_hgb_extra_trees": "existing3_clean_mse__selected5",
           "simple_equal": "hgb_extra_trees5_equal"}
LABELS = {"randomforest": "元RF (PCA＋XGBRF)", "conformer": "CNN＋Transformer", "alexnet": "AlexNet派生CNN",
          "hgb": "HGB34単体", "extra_trees": "ExtraTrees34単体", "original3_mse": "元3 MSE",
          "original3_performance": "元3 従来performance", "original3_hgb": "元3＋HGB (4)",
          "original3_extra_trees": "元3＋ExtraTrees (4)", "original3_hgb_extra_trees": "元3＋HGB＋ExtraTrees (5)",
          "simple_equal": "5等平均"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    protocol = read_json(PROTOCOL)
    rows, weights, inputs = [], [], []
    max_difference = 0.0
    for seed in [43, 44]:
        folder = SOURCE/f"seed{seed}"
        paths = [folder/"predictions.npz", folder/"outer_integrated.npz", folder/"frozen_integration.json",
                 PAIRED/f"seed{seed}/manifest.json"]
        inputs.extend({"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths)
        manifest, frozen = read_json(paths[-1]), read_json(paths[-2])
        methods = {r["method"]: np.asarray(r["weights"]) for r in frozen["methods"]}
        with np.load(paths[0]) as saved:
            keys, noises = saved["keys"].tolist(), saved["noises"].tolist()
            outer, indices = saved["outer"].copy(), saved["test_indices"].copy()
        with np.load(paths[1]) as saved:
            prior, names = saved["predictions"].copy(), saved["methods"].tolist()
            np.testing.assert_array_equal(indices, saved["indices"])
        metadata = [manifest["samples"][int(i)] for i in indices]
        targets = np.asarray([float(r["heat_flux"]) for r in metadata])
        for strategy, method in MAPPING.items():
            for key in protocol["model_keys"]:
                weights.append({"seed": seed, "fold": 1, "strategy_name": strategy,
                                "model_key": key, "weight": float(methods[method][keys.index(key)]),
                                "fit_scope": "clean training OOF; fixed across all noise"})
        for n, noise in enumerate(noises):
            predictions = {key: outer[n, :, keys.index(key)]*1000 for key in protocol["model_keys"]}
            for strategy, method in MAPPING.items():
                reconstructed = outer[n] @ methods[method]
                max_difference = max(max_difference, float(np.max(np.abs(reconstructed-prior[n, :, names.index(method)]))))
                np.testing.assert_allclose(reconstructed, prior[n, :, names.index(method)], rtol=1e-12, atol=1e-10)
                predictions["ensemble__"+strategy] = reconstructed*1000
            labels = {k: LABELS[k.removeprefix("ensemble__")] for k in predictions}
            current = source_day_metric_rows(targets, predictions, metadata, protocol["source_day_thresholds_W_m2"], 1, labels,
                                             protocol["near_onb_band_fraction"])
            rows.extend({"seed": seed, "noise": noise, "source_result": str(paths[0].relative_to(ROOT)), **r} for r in current)
    # Match every historical field for all overlapping production rows.
    historical_path = ROOT/"experiments/2026-10-06_onb_five_model_integration/metrics_recomputed.csv"
    with historical_path.open(encoding="utf-8-sig") as stream:
        historical = list(csv.DictReader(stream))
    lookup = {(r["seed"], r["noise"], r["source_day"], r["model_key"]): r for r in rows}
    compared = 0
    for before in historical:
        after = lookup[int(before["seed"]), before["noise"], before["source_day"], before["model_key"]]
        for key, value in after.items():
            if key in before and isinstance(value, (float, int, np.integer)):
                np.testing.assert_allclose(value, float(before[key]), rtol=1e-10, atol=1e-6, equal_nan=True)
        compared += 1
    verification = {"status": "passed", "scope": "saved paired 150-epoch development study, not a new normal-main run",
                    "new_model_fits": 0, "new_model_predictions": 0,
                    "normal_entrypoint_150_epoch_run_executed": False,
                    "source_rows": len(rows), "historical_metric_rows_matched": compared,
                    "max_reconstructed_prediction_difference_kW_m2": max_difference, "inputs": inputs}
    result = write_comparison_report(OUT/"saved_main_tables", rows, protocol,
        {"seeds": [43, 44], "source": "fixed-three additions study", "source_protocol": str(PROTOCOL.relative_to(ROOT)),
         "evaluation": "same known-WAV unused chunks; 540 per seed; source-day ONB thresholds",
         "new_standard_run": False}, weights, verification)
    print(json.dumps({"status": "passed", "main_report": str(result.relative_to(ROOT)), "rows": len(rows), "matched": compared}))


if __name__ == "__main__":
    main()
