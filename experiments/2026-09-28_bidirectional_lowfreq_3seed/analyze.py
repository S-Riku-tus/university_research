from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "0611_to_0618": {
        42: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_160352",
        43: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_171715",
        44: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_180134",
    },
    "0618_to_0611": {
        42: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
        43: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/28/onb_xd-t0618-v0611_iw3-nc_s0_e150_102124",
        44: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/28/onb_xd-t0618-v0611_iw3-nc_s0_e150_132256",
    },
}
FREQUENCIES = ["3kHz", "5kHz"]
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
MODEL_KEYS = {
    "randomforest": "RandomForest",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "performance_kfold",
    "ensemble__simple_equal": "simple_equal",
}
SUMMARY_LABELS = {
    "randomforest": "RandomForest",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "Ensemble individual-performance KFold",
    "ensemble__simple_equal": "Ensemble simple equal",
}
SINGLES = {"RandomForest", "Conformer", "AlexNet"}
VALUE_FIELDS = ["r2", "rmse_kw_m2", "bias_kw_m2", "onb_rmse_kw_m2", "pre_onb_fpr", "post_onb_recall"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], groups: list[str], values: list[str] = VALUE_FIELDS) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[g] for g in groups)].append(row)
    output = []
    for key, selected in buckets.items():
        out = dict(zip(groups, key)); out["n"] = len(selected)
        for field in values:
            array = np.asarray([float(row[field]) for row in selected])
            out[field + "_mean"] = np.mean(array)
            out[field + "_sd"] = np.std(array, ddof=1) if len(array) > 1 else 0.0
        output.append(out)
    return output


metrics: list[dict[str, object]] = []
weights: list[dict[str, object]] = []
correlations: list[dict[str, object]] = []
for direction, seeded_runs in RUNS.items():
    for seed, run in seeded_runs.items():
        for frequency in FREQUENCIES:
            fit_ids = set()
            for noise in NOISES:
                noise_dir = "heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}"
                condition = run / f"maxfreq={frequency}" / noise_dir
                fit_ids.update(json.loads((condition / "completed.json").read_text())["fit_ids"])
                manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
                if manifest["validation_config"]["run"]["random_seed"] != seed:
                    raise RuntimeError(f"Seed mismatch: {condition}")
                threshold = float(manifest["dataset"]["threshold"])
                suffix = "no_noise" if noise == "clean" else noise
                predictions = read_csv(condition / "fold_pred" / f"pred_f1_{suffix}.csv")
                summary = {row["model"]: row for row in read_csv(condition / f"metrics_summary_{suffix}.csv")}
                y = np.asarray([float(row["y_true"]) for row in predictions])
                before, after = y < threshold, y >= threshold
                residuals = {}
                for key, model in MODEL_KEYS.items():
                    pred = np.asarray([float(row[key]) for row in predictions])
                    residual = pred - y; residuals[model] = residual
                    metrics.append({
                        "direction": direction, "seed": seed, "frequency": frequency, "snr": noise, "model": model,
                        "r2": 1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2),
                        "rmse_kw_m2": np.sqrt(np.mean(residual**2)) / 1000,
                        "bias_kw_m2": np.mean(residual) / 1000,
                        "onb_rmse_kw_m2": float(summary[SUMMARY_LABELS[key]]["rmse_onb_mean"]) / 1000,
                        "pre_onb_fpr": np.mean(pred[before] >= threshold),
                        "post_onb_recall": np.mean(pred[after] >= threshold),
                    })
                correlations.append({
                    "direction": direction, "seed": seed, "frequency": frequency, "snr": noise,
                    "deep_residual_correlation": np.corrcoef(residuals["Conformer"], residuals["AlexNet"])[0, 1],
                })
                if noise == "clean":
                    learned_weights = read_csv(condition / "ensemble_weights_no_noise.csv")[0]
                    internal = json.loads((condition / "internal_validation_fold1.json").read_text(encoding="utf-8"))
                    for key in ("randomforest", "conformer", "alexnet"):
                        external = next(row for row in metrics if row["direction"] == direction and row["seed"] == seed and row["frequency"] == frequency and row["snr"] == "clean" and row["model"] == MODEL_KEYS[key])
                        weights.append({
                            "direction": direction, "seed": seed, "frequency": frequency, "model": MODEL_KEYS[key],
                            "internal_oof_r2": 1 - float(internal["individual_errors"][key]),
                            "weight": float(learned_weights[key]), "external_r2": external["r2"],
                            "external_rmse_kw_m2": external["rmse_kw_m2"],
                        })
            if len(fit_ids) != 1:
                raise RuntimeError(f"Expected one fit: {direction}, seed={seed}, {frequency}, {fit_ids}")

write_csv(Path(__file__).with_name("metrics_by_seed.csv"), metrics)
write_csv(Path(__file__).with_name("weights.csv"), weights)
write_csv(Path(__file__).with_name("deep_residual_correlations.csv"), correlations)

clean = aggregate([row for row in metrics if row["snr"] == "clean"], ["direction", "frequency", "model"])
write_csv(Path(__file__).with_name("clean_aggregate.csv"), clean)

# Noise values are averaged within each run before seed/direction summaries.
noise_by_run_raw = aggregate([row for row in metrics if row["snr"] != "clean"], ["direction", "seed", "frequency", "model"])
noise_by_run = [{**{k: row[k] for k in ("direction", "seed", "frequency", "model")}, **{v: row[v + "_mean"] for v in VALUE_FIELDS}} for row in noise_by_run_raw]
noise_direction = aggregate(noise_by_run, ["direction", "frequency", "model"])
noise_combined = aggregate(noise_by_run, ["frequency", "model"])
write_csv(Path(__file__).with_name("noise_direction_aggregate.csv"), noise_direction)
write_csv(Path(__file__).with_name("noise_bidirectional_aggregate.csv"), noise_combined)

snr_direction = aggregate(metrics, ["direction", "frequency", "snr", "model"])
snr_combined = aggregate(metrics, ["frequency", "snr", "model"])
write_csv(Path(__file__).with_name("snr_direction_aggregate.csv"), snr_direction)
write_csv(Path(__file__).with_name("snr_bidirectional_aggregate.csv"), snr_combined)

success: list[dict[str, object]] = []
for direction_scope in ("0611_to_0618", "0618_to_0611", "both"):
    directions = list(RUNS) if direction_scope == "both" else [direction_scope]
    for frequency in FREQUENCIES:
        for ensemble in ("performance_kfold", "simple_equal"):
            beat_best = within2 = beat_rf = 0; delta_best = []; delta_rf = []
            for direction in directions:
                for seed in (42, 43, 44):
                    for noise in NOISES:
                        cell = [row for row in metrics if row["direction"] == direction and row["seed"] == seed and row["frequency"] == frequency and row["snr"] == noise]
                        best = min(float(row["rmse_kw_m2"]) for row in cell if row["model"] in SINGLES)
                        rf = next(float(row["rmse_kw_m2"]) for row in cell if row["model"] == "RandomForest")
                        value = next(float(row["rmse_kw_m2"]) for row in cell if row["model"] == ensemble)
                        beat_best += value < best; within2 += value <= 1.02 * best; beat_rf += value < rf
                        delta_best.append(value - best); delta_rf.append(value - rf)
            success.append({
                "scope": direction_scope, "frequency": frequency, "ensemble": ensemble, "cells": len(delta_best),
                "beat_best_single": beat_best, "within_2pct_best_single": within2, "beat_rf": beat_rf,
                "mean_delta_best_kw_m2": np.mean(delta_best), "mean_delta_rf_kw_m2": np.mean(delta_rf),
            })
write_csv(Path(__file__).with_name("ensemble_success.csv"), success)

transfer: list[dict[str, object]] = []
for scope in ("0611_to_0618", "0618_to_0611", "both"):
    directions = list(RUNS) if scope == "both" else [scope]
    matches = 0; rank_correlations = []
    for direction in directions:
        for seed in (42, 43, 44):
            for frequency in FREQUENCIES:
                selected = [row for row in weights if row["direction"] == direction and row["seed"] == seed and row["frequency"] == frequency]
                internal = np.asarray([float(row["internal_oof_r2"]) for row in selected])
                external = np.asarray([float(row["external_r2"]) for row in selected])
                matches += np.argmax(internal) == np.argmax(external)
                rank_correlations.append(np.corrcoef(np.argsort(np.argsort(internal)), np.argsort(np.argsort(external)))[0, 1])
    transfer.append({
        "scope": scope, "cells": len(rank_correlations), "best_match": matches,
        "mean_spearman": np.mean(rank_correlations), "sd_spearman": np.std(rank_correlations, ddof=1),
    })
write_csv(Path(__file__).with_name("weight_transfer_summary.csv"), transfer)

print("Noise averages by direction (mean +/- seed SD)")
for direction in RUNS:
    for frequency in FREQUENCIES:
        print(f"\n{direction} {frequency}")
        for row in noise_direction:
            if row["direction"] == direction and row["frequency"] == frequency:
                print(f"{row['model']:20s} RMSE={row['rmse_kw_m2_mean']:6.1f}+/-{row['rmse_kw_m2_sd']:4.1f} R2={row['r2_mean']:.3f} FPR={row['pre_onb_fpr_mean']:.3f}")
print("\nBidirectional noise macro-average")
for frequency in FREQUENCIES:
    print(f"\n{frequency}")
    for row in noise_combined:
        if row["frequency"] == frequency:
            print(f"{row['model']:20s} RMSE={row['rmse_kw_m2_mean']:6.1f}+/-{row['rmse_kw_m2_sd']:4.1f} R2={row['r2_mean']:.3f} FPR={row['pre_onb_fpr_mean']:.3f}")
print("\nSuccess", success)
print("\nTransfer", transfer)
