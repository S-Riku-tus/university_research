from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "0611_to_0618": {
        "3kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_160352",
        "5kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/28/onb_xd-t0611-v0618_iw3-nc_s0_e150_160352",
        "22kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_000904",
    },
    "0618_to_0611": {
        "3kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
        "5kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
        "22kHz": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_172749",
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
SUMMARY_LABELS = {
    "randomforest": "RandomForest",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "Ensemble individual-performance KFold",
    "ensemble__simple_equal": "Ensemble simple equal",
}
SINGLES = {"RandomForest", "Conformer", "AlexNet"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


rows: list[dict[str, object]] = []
weights: list[dict[str, object]] = []
correlations: list[dict[str, object]] = []
for direction, frequency_runs in RUNS.items():
    for frequency, run in frequency_runs.items():
        fit_ids = set()
        for noise in NOISES:
            noise_dir = "heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}"
            condition = run / f"maxfreq={frequency}" / noise_dir
            fit_ids.update(json.loads((condition / "completed.json").read_text())["fit_ids"])
            manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest["validation_config"]["run"]["random_seed"] != 42:
                raise RuntimeError(f"Unexpected seed: {condition}")
            threshold = float(manifest["dataset"]["threshold"])
            suffix = "no_noise" if noise == "clean" else noise
            pred_rows = read_csv(condition / "fold_pred" / f"pred_f1_{suffix}.csv")
            summary = {r["model"]: r for r in read_csv(condition / f"metrics_summary_{suffix}.csv")}
            y = np.asarray([float(r["y_true"]) for r in pred_rows])
            before, after = y < threshold, y >= threshold
            residuals = {}
            for key, model in MODEL_KEYS.items():
                pred = np.asarray([float(r[key]) for r in pred_rows])
                residual = pred - y
                residuals[model] = residual
                rows.append({
                    "direction": direction, "frequency": frequency, "snr": noise, "model": model,
                    "r2": 1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2),
                    "rmse_kw_m2": np.sqrt(np.mean(residual**2)) / 1000,
                    "bias_kw_m2": np.mean(residual) / 1000,
                    "onb_rmse_kw_m2": float(summary[SUMMARY_LABELS[key]]["rmse_onb_mean"]) / 1000,
                    "pre_onb_fpr": np.mean(pred[before] >= threshold),
                    "post_onb_recall": np.mean(pred[after] >= threshold),
                })
            correlations.append({
                "direction": direction, "frequency": frequency, "snr": noise,
                "deep_residual_correlation": np.corrcoef(residuals["Conformer"], residuals["AlexNet"])[0, 1],
            })
            if noise == "clean":
                weight_row = read_csv(condition / "ensemble_weights_no_noise.csv")[0]
                internal = json.loads((condition / "internal_validation_fold1.json").read_text(encoding="utf-8"))
                for key in ("randomforest", "conformer", "alexnet"):
                    external = next(r for r in rows if r["direction"] == direction and r["frequency"] == frequency and r["snr"] == "clean" and r["model"] == MODEL_KEYS[key])
                    weights.append({
                        "direction": direction, "frequency": frequency, "model": MODEL_KEYS[key],
                        "internal_oof_r2": 1 - float(internal["individual_errors"][key]),
                        "weight": float(weight_row[key]), "external_r2": external["r2"],
                        "external_rmse_kw_m2": external["rmse_kw_m2"],
                    })
        if len(fit_ids) != 1:
            raise RuntimeError(f"Expected one clean fit: {direction}, {frequency}, {fit_ids}")

write_csv(Path(__file__).with_name("metrics.csv"), rows)
write_csv(Path(__file__).with_name("weights.csv"), weights)
write_csv(Path(__file__).with_name("deep_residual_correlations.csv"), correlations)

summary_rows: list[dict[str, object]] = []
for direction in RUNS:
    for frequency in ("3kHz", "5kHz", "22kHz"):
        for model in MODEL_KEYS.values():
            selected = [r for r in rows if r["direction"] == direction and r["frequency"] == frequency and r["model"] == model]
            clean = next(r for r in selected if r["snr"] == "clean")
            noisy = [r for r in selected if r["snr"] != "clean"]
            summary_rows.append({
                "direction": direction, "frequency": frequency, "model": model,
                "clean_rmse_kw_m2": clean["rmse_kw_m2"], "clean_r2": clean["r2"],
                "noise_rmse_kw_m2": np.mean([r["rmse_kw_m2"] for r in noisy]),
                "noise_r2": np.mean([r["r2"] for r in noisy]),
                "noise_pre_onb_fpr": np.mean([r["pre_onb_fpr"] for r in noisy]),
                "noise_post_onb_recall": np.mean([r["post_onb_recall"] for r in noisy]),
            })
write_csv(Path(__file__).with_name("direction_frequency_summary.csv"), summary_rows)

success: list[dict[str, object]] = []
for frequency in ("3kHz", "5kHz"):
    for ensemble in ("performance_kfold", "simple_equal"):
        deltas_best, deltas_rf, within2_flags = [], [], []
        for noise in NOISES:
            cell = [r for r in rows if r["direction"] == "0611_to_0618" and r["frequency"] == frequency and r["snr"] == noise]
            best = min(float(r["rmse_kw_m2"]) for r in cell if r["model"] in SINGLES)
            rf = next(float(r["rmse_kw_m2"]) for r in cell if r["model"] == "RandomForest")
            value = next(float(r["rmse_kw_m2"]) for r in cell if r["model"] == ensemble)
            deltas_best.append(value - best); deltas_rf.append(value - rf)
            within2_flags.append(value <= 1.02 * best)
        success.append({
            "frequency": frequency, "ensemble": ensemble,
            "beat_best_single": sum(d < 0 for d in deltas_best), "within_2pct_best_single": sum(within2_flags),
            "beat_rf": sum(d < 0 for d in deltas_rf), "mean_delta_to_best_kw_m2": np.mean(deltas_best),
            "mean_delta_to_rf_kw_m2": np.mean(deltas_rf),
        })
write_csv(Path(__file__).with_name("forward_ensemble_success.csv"), success)

print("Forward seed42 summary")
for frequency in ("3kHz", "5kHz", "22kHz"):
    print(f"\n{frequency}")
    for row in summary_rows:
        if row["direction"] == "0611_to_0618" and row["frequency"] == frequency:
            print(f"{row['model']:20s} clean={row['clean_rmse_kw_m2']:6.1f} noise={row['noise_rmse_kw_m2']:6.1f} R2={row['noise_r2']:.3f} FPR={row['noise_pre_onb_fpr']:.3f}")
print("\nBidirectional low-frequency summary")
for row in summary_rows:
    if row["frequency"] in ("3kHz", "5kHz") and row["model"] in ("RandomForest", "performance_kfold", "simple_equal"):
        print(row)
print("\nSuccess", success)
