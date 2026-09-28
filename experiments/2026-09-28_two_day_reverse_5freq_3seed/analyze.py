from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REVERSE_RUNS = {
    42: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
    43: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/28/onb_xd-t0618-v0611_iw3-nc_s0_e150_102124",
    44: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/28/onb_xd-t0618-v0611_iw3-nc_s0_e150_132256",
}
FORWARD_22K_RUNS = {
    42: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_000904",
    43: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_150123",
    44: ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_161102",
}
FREQUENCIES = ["3kHz", "5kHz", "10kHz", "15kHz", "22kHz"]
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
SINGLES = ["RandomForest", "Conformer", "AlexNet"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_dir(run: Path, frequency: str, noise: str) -> Path:
    name = "heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}"
    return run / f"maxfreq={frequency}" / name


def collect(run: Path, seed: int, direction: str, frequencies: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    weights: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    for frequency in frequencies:
        fit_ids = set()
        for noise in NOISES:
            condition = condition_dir(run, frequency, noise)
            completed = json.loads((condition / "completed.json").read_text())
            fit_ids.update(completed["fit_ids"])
            manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest["validation_config"]["run"]["random_seed"] != seed:
                raise RuntimeError(f"Seed mismatch: {condition}")
            threshold = float(manifest["dataset"]["threshold"])
            suffix = "no_noise" if noise == "clean" else noise
            pred_rows = read_csv(condition / "fold_pred" / f"pred_f1_{suffix}.csv")
            summary = {r["model"]: r for r in read_csv(condition / f"metrics_summary_{suffix}.csv")}
            y = np.asarray([float(r["y_true"]) for r in pred_rows])
            before = y < threshold
            after = ~before
            residuals: dict[str, np.ndarray] = {}
            for key, label in MODEL_KEYS.items():
                pred = np.asarray([float(r[key]) for r in pred_rows])
                residual = pred - y
                residuals[label] = residual
                metrics.append({
                    "direction": direction,
                    "seed": seed,
                    "frequency": frequency,
                    "snr": noise,
                    "model": label,
                    "r2": 1 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2),
                    "rmse_kw_m2": np.sqrt(np.mean(residual**2)) / 1000,
                    "bias_kw_m2": np.mean(residual) / 1000,
                    "onb_rmse_kw_m2": float(summary[SUMMARY_LABELS[key]]["rmse_onb_mean"]) / 1000,
                    "pre_onb_fpr": np.mean(pred[before] >= threshold),
                    "post_onb_recall": np.mean(pred[after] >= threshold),
                })
            correlations.append({
                "direction": direction,
                "seed": seed,
                "frequency": frequency,
                "snr": noise,
                "deep_residual_correlation": np.corrcoef(residuals["Conformer"], residuals["AlexNet"])[0, 1],
            })
            if noise == "clean":
                weight_row = read_csv(condition / "ensemble_weights_no_noise.csv")[0]
                internal = json.loads((condition / "internal_validation_fold1.json").read_text(encoding="utf-8"))
                for key in ("randomforest", "conformer", "alexnet"):
                    external = next(r for r in metrics if r["direction"] == direction and r["seed"] == seed and r["frequency"] == frequency and r["snr"] == "clean" and r["model"] == MODEL_KEYS[key])
                    weights.append({
                        "direction": direction,
                        "seed": seed,
                        "frequency": frequency,
                        "model": MODEL_KEYS[key],
                        "internal_oof_r2": 1 - float(internal["individual_errors"][key]),
                        "performance_weight": float(weight_row[key]),
                        "external_r2": external["r2"],
                        "external_rmse_kw_m2": external["rmse_kw_m2"],
                    })
        if len(fit_ids) != 1:
            raise RuntimeError(f"Expected one clean fit for {direction}, seed={seed}, frequency={frequency}; got {fit_ids}")
    return metrics, weights, correlations


metrics: list[dict[str, object]] = []
weights: list[dict[str, object]] = []
correlations: list[dict[str, object]] = []
for seed, run in REVERSE_RUNS.items():
    m, w, c = collect(run, seed, "0618_to_0611", FREQUENCIES)
    metrics.extend(m); weights.extend(w); correlations.extend(c)
for seed, run in FORWARD_22K_RUNS.items():
    m, w, c = collect(run, seed, "0611_to_0618", ["22kHz"])
    metrics.extend(m); weights.extend(w); correlations.extend(c)

write_csv(Path(__file__).with_name("metrics_by_seed.csv"), metrics)
write_csv(Path(__file__).with_name("weight_transfer.csv"), weights)
write_csv(Path(__file__).with_name("deep_residual_correlations.csv"), correlations)


def aggregate(selected: list[dict[str, object]], group_fields: list[str], value_fields: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        groups[tuple(row[f] for f in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, group in groups.items():
        row = dict(zip(group_fields, key))
        row["n"] = len(group)
        for field in value_fields:
            values = np.asarray([float(r[field]) for r in group])
            row[f"{field}_mean"] = np.mean(values)
            row[f"{field}_sd"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        output.append(row)
    return output


values = ["r2", "rmse_kw_m2", "bias_kw_m2", "onb_rmse_kw_m2", "pre_onb_fpr", "post_onb_recall"]
clean_reverse = [r for r in metrics if r["direction"] == "0618_to_0611" and r["snr"] == "clean"]
clean_aggregate = aggregate(clean_reverse, ["frequency", "model"], values)
write_csv(Path(__file__).with_name("clean_3seed_aggregate.csv"), clean_aggregate)

# Average the six noisy conditions within each seed before estimating seed variation.
noise_seed = aggregate(
    [r for r in metrics if r["direction"] == "0618_to_0611" and r["snr"] != "clean"],
    ["seed", "frequency", "model"], values,
)
noise_seed_flat = [
    {**{k: r[k] for k in ("seed", "frequency", "model")}, **{v: r[f"{v}_mean"] for v in values}}
    for r in noise_seed
]
noise_aggregate = aggregate(noise_seed_flat, ["frequency", "model"], values)
write_csv(Path(__file__).with_name("noise_3seed_aggregate.csv"), noise_aggregate)

snr_aggregate = aggregate(
    [r for r in metrics if r["direction"] == "0618_to_0611"],
    ["frequency", "snr", "model"], values,
)
write_csv(Path(__file__).with_name("snr_3seed_aggregate.csv"), snr_aggregate)

bidirectional_22k: list[dict[str, object]] = []
for direction in ("0611_to_0618", "0618_to_0611"):
    direction_rows = [r for r in metrics if r["direction"] == direction and r["frequency"] == "22kHz"]
    clean_rows = [r for r in direction_rows if r["snr"] == "clean"]
    bidirectional_22k.extend(aggregate(clean_rows, ["direction", "model"], values))
    noisy_seed = aggregate([r for r in direction_rows if r["snr"] != "clean"], ["seed", "direction", "model"], values)
    flattened = [{**{k: r[k] for k in ("seed", "direction", "model")}, **{v: r[f"{v}_mean"] for v in values}} for r in noisy_seed]
    output = aggregate(flattened, ["direction", "model"], values)
    for row in output:
        row["direction"] += "_noise_mean"
    bidirectional_22k.extend(output)
write_csv(Path(__file__).with_name("bidirectional_22khz_3seed.csv"), bidirectional_22k)

# Test whether the learned clean ranking transfers to the held-out day.
transfer_summary: list[dict[str, object]] = []
for direction, frequencies in (("0618_to_0611", FREQUENCIES), ("0611_to_0618", ["22kHz"])):
    matched = 0
    correlations_rank: list[float] = []
    total = 0
    for seed in (42, 43, 44):
        for frequency in frequencies:
            group = [r for r in weights if r["direction"] == direction and r["seed"] == seed and r["frequency"] == frequency]
            internal = np.asarray([float(r["internal_oof_r2"]) for r in group])
            external = np.asarray([float(r["external_r2"]) for r in group])
            matched += int(np.argmax(internal) == np.argmax(external))
            correlations_rank.append(float(np.corrcoef(np.argsort(np.argsort(internal)), np.argsort(np.argsort(external)))[0, 1]))
            total += 1
    transfer_summary.append({
        "direction": direction,
        "cells": total,
        "best_model_match_count": matched,
        "best_model_match_rate": matched / total,
        "mean_spearman_rank": np.mean(correlations_rank),
        "sd_spearman_rank": np.std(correlations_rank, ddof=1),
    })
write_csv(Path(__file__).with_name("weight_transfer_summary.csv"), transfer_summary)

# Count ensemble success relative to the best individual for reverse 3 seeds x 7 noise cells.
success: list[dict[str, object]] = []
for frequency in FREQUENCIES:
    frequency_rows = [r for r in metrics if r["direction"] == "0618_to_0611" and r["frequency"] == frequency]
    for ensemble in ("performance_kfold", "simple_equal"):
        beat = within2 = 0
        deltas: list[float] = []
        for seed in (42, 43, 44):
            for noise in NOISES:
                cell = [r for r in frequency_rows if r["seed"] == seed and r["snr"] == noise]
                best = min(float(r["rmse_kw_m2"]) for r in cell if r["model"] in SINGLES)
                candidate = float(next(r["rmse_kw_m2"] for r in cell if r["model"] == ensemble))
                beat += int(candidate < best)
                within2 += int(candidate <= 1.02 * best)
                deltas.append(candidate - best)
        success.append({
            "frequency": frequency,
            "ensemble": ensemble,
            "cells": 21,
            "beat_best_single": beat,
            "within_2pct": within2,
            "mean_rmse_delta_kw_m2": np.mean(deltas),
        })
write_csv(Path(__file__).with_name("ensemble_success.csv"), success)

print("Reverse clean 3-seed RMSE mean +/- SD")
for frequency in FREQUENCIES:
    print(f"\n{frequency}")
    for row in clean_aggregate:
        if row["frequency"] == frequency:
            print(f"{row['model']:20s} {row['rmse_kw_m2_mean']:6.1f} +/- {row['rmse_kw_m2_sd']:4.1f}  R2={row['r2_mean']:.3f} FPR={row['pre_onb_fpr_mean']:.3f}")

print("\nReverse noisy-condition averages, 3 seeds")
for frequency in FREQUENCIES:
    print(f"\n{frequency}")
    for row in noise_aggregate:
        if row["frequency"] == frequency:
            print(f"{row['model']:20s} {row['rmse_kw_m2_mean']:6.1f} +/- {row['rmse_kw_m2_sd']:4.1f}  R2={row['r2_mean']:.3f} FPR={row['pre_onb_fpr_mean']:.3f}")

print("\nWeight transfer", transfer_summary)
print("\nEnsemble success", success)
