from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUNS = {
    "0611_to_0618": {
        "matched": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nm_s0_e150_194237",
        "clean_only": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_160352",
    },
    "0618_to_0611": {
        "matched": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/28/onb_xd-t0618-v0611_iw3-nm_s0_e150_205608",
        "clean_only": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
    },
}
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
MODEL_KEYS = {
    "randomforest": "RandomForest",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "performance_kfold",
    "ensemble__simple_equal": "simple_equal",
}
SINGLES = ("RandomForest", "Conformer", "AlexNet")
VALUE_FIELDS = (
    "r2", "rmse_kw_m2", "mae_kw_m2", "bias_kw_m2", "pre_onb_rmse_kw_m2",
    "post_onb_rmse_kw_m2", "onb_rmse_kw_m2", "pre_onb_fpr", "post_onb_recall",
    "accuracy", "precision", "f1",
    "roc_auc_cont", "pr_auc_cont",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, pred: np.ndarray, threshold: float, band: float) -> dict[str, float]:
    residual = pred - y
    before, after = y < threshold, y >= threshold
    near = np.abs(y - threshold) <= abs(threshold) * band
    predicted_after = pred >= threshold
    tp = np.sum(predicted_after & after)
    fp = np.sum(predicted_after & before)
    tn = np.sum(~predicted_after & before)
    fn = np.sum(~predicted_after & after)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0

    def rmse(mask: np.ndarray) -> float:
        return float(np.sqrt(np.mean(residual[mask] ** 2)) / 1000) if np.any(mask) else np.nan

    return {
        "r2": float(1 - np.sum(residual ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "rmse_kw_m2": float(np.sqrt(np.mean(residual ** 2)) / 1000),
        "mae_kw_m2": float(np.mean(np.abs(residual)) / 1000),
        "bias_kw_m2": float(np.mean(residual) / 1000),
        "pre_onb_rmse_kw_m2": rmse(before),
        "post_onb_rmse_kw_m2": rmse(after),
        "onb_rmse_kw_m2": rmse(near),
        "pre_onb_fpr": float(np.mean(pred[before] >= threshold)),
        "post_onb_recall": recall,
        "accuracy": float((tp + tn) / (tp + fp + tn + fn)),
        "precision": precision,
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "roc_auc_cont": float(roc_auc_score(after, pred)),
        "pr_auc_cont": float(average_precision_score(after, pred)),
    }


def rank_correlation(a: dict[str, float], b: dict[str, float]) -> float:
    keys = list(SINGLES)
    ar = np.argsort(np.argsort([a[key] for key in keys])).astype(float)
    br = np.argsort(np.argsort([b[key] for key in keys])).astype(float)
    return float(np.corrcoef(ar, br)[0, 1])


rows: list[dict[str, object]] = []
weights: list[dict[str, object]] = []
correlations: list[dict[str, object]] = []
predictions: dict[tuple[str, str, str, str], np.ndarray] = {}
targets: dict[tuple[str, str], np.ndarray] = {}
sample_groups: dict[tuple[str, str], np.ndarray] = {}
audits: list[dict[str, object]] = []

for direction, policies in RUNS.items():
    for policy, run in policies.items():
        fit_ids = []
        for noise in NOISES:
            noise_dir = "heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}"
            condition = run / "maxfreq=3kHz" / noise_dir
            required = [
                condition / "completed.json", condition / "run_manifest.json",
                condition / "internal_validation_fold1.json",
            ]
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(f"Incomplete condition: {condition}")
            completed = json.loads(required[0].read_text(encoding="utf-8"))
            manifest = json.loads(required[1].read_text(encoding="utf-8"))
            context = manifest["validation_config"]["learning_policy"]
            if context["training_noise"] != policy:
                raise RuntimeError(f"Policy mismatch: {condition}")
            if manifest["dataset"]["max_freq_hz"] != "maxfreq=3kHz":
                raise RuntimeError(f"Frequency mismatch: {condition}")
            if manifest["validation_config"]["run"]["random_seed"] != 42:
                raise RuntimeError(f"Seed mismatch: {condition}")
            fit_id = completed["fit_ids"]
            if len(fit_id) != 1:
                raise RuntimeError(f"Expected one outer fit: {condition}")
            fit_ids.append(fit_id[0])
            suffix = "no_noise" if noise == "clean" else noise
            prediction_rows = read_csv(condition / "fold_pred" / f"pred_f1_{suffix}.csv")
            y = np.asarray([float(row["y_true"]) for row in prediction_rows])
            target_key = (direction, noise)
            if target_key in targets and not np.array_equal(targets[target_key], y):
                raise RuntimeError(f"Target alignment mismatch: {direction}, {noise}")
            targets[target_key] = y
            groups = np.asarray([row["source_wav_id"] for row in prediction_rows])
            if target_key in sample_groups and not np.array_equal(sample_groups[target_key], groups):
                raise RuntimeError(f"WAV alignment mismatch: {direction}, {noise}")
            sample_groups[target_key] = groups
            threshold = float(manifest["dataset"]["threshold"])
            band = float(manifest["validation_config"]["thresholds"]["onb_band_frac"])
            residuals = {}
            for key, model in MODEL_KEYS.items():
                pred = np.asarray([float(row[key]) for row in prediction_rows])
                predictions[(direction, policy, noise, model)] = pred
                residuals[model] = pred - y
                rows.append({
                    "direction": direction, "policy": policy, "snr": noise, "model": model,
                    **metrics(y, pred, threshold, band),
                })
            for left, right in (("RandomForest", "Conformer"), ("RandomForest", "AlexNet"),
                                ("Conformer", "AlexNet")):
                correlations.append({
                    "direction": direction, "policy": policy, "snr": noise,
                    "model_pair": f"{left}__{right}",
                    "residual_correlation": float(np.corrcoef(residuals[left], residuals[right])[0, 1]),
                })
            weight_row = read_csv(condition / f"ensemble_weights_{suffix}.csv")[0]
            internal = json.loads(required[2].read_text(encoding="utf-8"))
            internal_r2 = {
                MODEL_KEYS[key]: 1 - float(value)
                for key, value in internal["individual_errors"].items()
            }
            external_r2 = {
                model: next(row["r2"] for row in reversed(rows)
                            if row["direction"] == direction and row["policy"] == policy
                            and row["snr"] == noise and row["model"] == model)
                for model in SINGLES
            }
            external_rmse = {
                model: next(row["rmse_kw_m2"] for row in reversed(rows)
                            if row["direction"] == direction and row["policy"] == policy
                            and row["snr"] == noise and row["model"] == model)
                for model in SINGLES
            }
            audits.append({
                "direction": direction, "policy": policy, "snr": noise,
                "internal_best": max(internal_r2, key=internal_r2.get),
                "external_best": min(external_rmse, key=external_rmse.get),
                "best_match": int(max(internal_r2, key=internal_r2.get)
                                  == min(external_rmse, key=external_rmse.get)),
                "spearman": rank_correlation(internal_r2, external_r2),
            })
            for key, model in list(MODEL_KEYS.items())[:3]:
                weights.append({
                    "direction": direction, "policy": policy, "snr": noise, "model": model,
                    "internal_oof_r2": internal_r2[model], "weight": float(weight_row[key]),
                    "external_r2": external_r2[model], "external_rmse_kw_m2": external_rmse[model],
                })
        expected_fits = 7 if policy == "matched" else 1
        if len(set(fit_ids)) != expected_fits:
            raise RuntimeError(
                f"Unexpected fit sharing: {direction}, {policy}, "
                f"expected={expected_fits}, actual={len(set(fit_ids))}"
            )

write_csv(OUTPUT / "metrics.csv", rows)
write_csv(OUTPUT / "weights.csv", weights)
write_csv(OUTPUT / "weight_transfer.csv", audits)
write_csv(OUTPUT / "residual_correlations.csv", correlations)

# Descriptive average across the six SNR conditions within each direction.
direction_average = []
for direction in RUNS:
    for policy in ("matched", "clean_only"):
        for model in MODEL_KEYS.values():
            selected = [row for row in rows if row["direction"] == direction
                        and row["policy"] == policy and row["snr"] != "clean"
                        and row["model"] == model]
            direction_average.append({
                "direction": direction, "policy": policy, "model": model, "n_snr": len(selected),
                **{field + "_mean": float(np.mean([row[field] for row in selected]))
                   for field in VALUE_FIELDS},
            })
write_csv(OUTPUT / "noise_direction_average.csv", direction_average)

# Macro average over the two direction-level SNR averages; directions have equal weight.
bidirectional_average = []
for policy in ("matched", "clean_only"):
    for model in MODEL_KEYS.values():
        selected = [row for row in direction_average if row["policy"] == policy and row["model"] == model]
        bidirectional_average.append({
            "policy": policy, "model": model, "n_directions": len(selected),
            **{field + "_mean": float(np.mean([row[field + "_mean"] for row in selected]))
               for field in VALUE_FIELDS},
        })
write_csv(OUTPUT / "noise_bidirectional_average.csv", bidirectional_average)

# Paired matched-minus-clean differences on the same evaluation chunks.
paired = []
for direction in RUNS:
    for noise in NOISES:
        for model in MODEL_KEYS.values():
            matched = next(row for row in rows if row["direction"] == direction
                           and row["policy"] == "matched" and row["snr"] == noise
                           and row["model"] == model)
            clean = next(row for row in rows if row["direction"] == direction
                         and row["policy"] == "clean_only" and row["snr"] == noise
                         and row["model"] == model)
            pred_diff = predictions[(direction, "matched", noise, model)] - predictions[
                (direction, "clean_only", noise, model)
            ]
            paired.append({
                "direction": direction, "snr": noise, "model": model,
                **{f"delta_{field}": matched[field] - clean[field] for field in VALUE_FIELDS},
                "prediction_change_rmse_kw_m2": float(np.sqrt(np.mean(pred_diff ** 2)) / 1000),
                "prediction_change_max_abs_kw_m2": float(np.max(np.abs(pred_diff)) / 1000),
            })
write_csv(OUTPUT / "matched_minus_clean.csv", paired)

# Paired changes per source WAV reveal whether an average change is localized.
wav_paired = []
for direction in RUNS:
    for noise in NOISES:
        y = targets[(direction, noise)]
        groups = sample_groups[(direction, noise)]
        for model in MODEL_KEYS.values():
            matched = predictions[(direction, "matched", noise, model)]
            clean = predictions[(direction, "clean_only", noise, model)]
            for group in np.unique(groups):
                mask = groups == group
                matched_rmse = float(np.sqrt(np.mean((matched[mask] - y[mask]) ** 2)) / 1000)
                clean_rmse = float(np.sqrt(np.mean((clean[mask] - y[mask]) ** 2)) / 1000)
                wav_paired.append({
                    "direction": direction, "snr": noise, "model": model,
                    "source_wav_id": group, "heat_flux_kw_m2": float(np.median(y[mask]) / 1000),
                    "matched_rmse_kw_m2": matched_rmse, "clean_rmse_kw_m2": clean_rmse,
                    "delta_rmse_kw_m2": matched_rmse - clean_rmse,
                    "delta_mse_kw_m2_squared": float(
                        np.mean(((matched[mask] - y[mask]) / 1000) ** 2)
                        - np.mean(((clean[mask] - y[mask]) / 1000) ** 2)
                    ),
                })
write_csv(OUTPUT / "wav_paired_deltas.csv", wav_paired)

# Ensemble comparison with the best post-hoc single and RF.
success = []
for direction_scope in [*RUNS, "both"]:
    directions = list(RUNS) if direction_scope == "both" else [direction_scope]
    for policy in ("matched", "clean_only"):
        for condition_scope, noises in (("all", NOISES), ("noise", NOISES[1:])):
            for ensemble in ("performance_kfold", "simple_equal"):
                cells = []
                for direction in directions:
                    for noise in noises:
                        relevant = [row for row in rows if row["direction"] == direction
                                    and row["policy"] == policy and row["snr"] == noise]
                        ensemble_rmse = next(row["rmse_kw_m2"] for row in relevant
                                             if row["model"] == ensemble)
                        singles = [row["rmse_kw_m2"] for row in relevant if row["model"] in SINGLES]
                        rf = next(row["rmse_kw_m2"] for row in relevant if row["model"] == "RandomForest")
                        best = min(singles)
                        cells.append((ensemble_rmse, best, rf))
                success.append({
                    "direction_scope": direction_scope, "policy": policy,
                    "condition_scope": condition_scope, "ensemble": ensemble, "cells": len(cells),
                    "beat_best_single": sum(e <= best for e, best, _ in cells),
                    "within_2pct_best_single": sum(e <= best * 1.02 for e, best, _ in cells),
                    "beat_rf": sum(e <= rf for e, _, rf in cells),
                    "mean_delta_best_kw_m2": float(np.mean([e - best for e, best, _ in cells])),
                    "mean_delta_rf_kw_m2": float(np.mean([e - rf for e, _, rf in cells])),
                })
write_csv(OUTPUT / "ensemble_success.csv", success)

# Summarize internal-to-external rank transfer.
transfer_summary = []
for direction_scope in [*RUNS, "both"]:
    directions = list(RUNS) if direction_scope == "both" else [direction_scope]
    for policy in ("matched", "clean_only"):
        for condition_scope, noises in (("all", NOISES), ("noise", NOISES[1:])):
            selected = [row for row in audits if row["direction"] in directions
                        and row["policy"] == policy and row["snr"] in noises]
            transfer_summary.append({
                "direction_scope": direction_scope, "policy": policy,
                "condition_scope": condition_scope, "cells": len(selected),
                "best_match": sum(row["best_match"] for row in selected),
                "mean_spearman": float(np.mean([row["spearman"] for row in selected])),
            })
write_csv(OUTPUT / "weight_transfer_summary.csv", transfer_summary)

# Clean should be exactly reproducible between policies under the same seed and parameters.
clean_identity = []
for direction in RUNS:
    for model in MODEL_KEYS.values():
        left = predictions[(direction, "matched", "clean", model)]
        right = predictions[(direction, "clean_only", "clean", model)]
        clean_identity.append({
            "direction": direction, "model": model,
            "max_abs_prediction_difference": float(np.max(np.abs(left - right))),
        })
write_csv(OUTPUT / "clean_identity.csv", clean_identity)
