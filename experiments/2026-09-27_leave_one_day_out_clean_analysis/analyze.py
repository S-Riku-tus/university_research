from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "test_0618": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611+0709-v0618_iw3-nc_s0_e150_004249",
    "test_0709": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.07.09_0.3_1/regression_result/npy/ensemble/202609/26/onb_xd-t0611+0618-v0709_iw3-nc_s0_e150_200137",
    "test_0611": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0709+0618-v0611_iw3-nc_s0_e150_110850",
}
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def noise_key(name: str) -> tuple[int, str]:
    if name == "heatflux_no_noise":
        return (0, "clean")
    return (1, name.split("=")[-1])


rows: list[dict[str, object]] = []
clean_rows: list[dict[str, object]] = []
audit_rows: list[dict[str, object]] = []
distribution_rows: list[dict[str, object]] = []
internal_day_rows: list[dict[str, object]] = []

for direction, run in RUNS.items():
    conditions = sorted((run / "maxfreq=22kHz").iterdir(), key=lambda p: noise_key(p.name))
    for condition in conditions:
        manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
        threshold = float(manifest["dataset"]["threshold"])
        suffix = "no_noise" if condition.name == "heatflux_no_noise" else condition.name.removeprefix("heatflux_reference_SNR=")
        pred_path = condition / "fold_pred" / f"pred_f1_{suffix}.csv"
        pred_rows = read_csv(pred_path)
        summary = {
            row["model"]: row
            for row in read_csv(condition / f"metrics_summary_{suffix}.csv")
        }
        y = np.asarray([float(r["y_true"]) for r in pred_rows])
        snr = noise_key(condition.name)[1]
        for key, label in MODEL_KEYS.items():
            pred = np.asarray([float(r[key]) for r in pred_rows])
            residual = pred - y
            before = y < threshold
            after = ~before
            slope, intercept = np.polyfit(y, pred, 1)
            row = {
                "direction": direction,
                "snr": snr,
                "model": label,
                "n": len(y),
                "rmse_kw_m2": np.sqrt(np.mean(residual**2)) / 1000,
                "mae_kw_m2": np.mean(np.abs(residual)) / 1000,
                "onb_rmse_kw_m2": float(summary[SUMMARY_LABELS[key]]["rmse_onb_mean"]) / 1000,
                "bias_kw_m2": np.mean(residual) / 1000,
                "r2": 1 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2),
                "slope": slope,
                "intercept_kw_m2": intercept / 1000,
                "pre_onb_fpr": np.mean(pred[before] >= threshold),
                "post_onb_recall": np.mean(pred[after] >= threshold),
                "prediction_min_kw_m2": np.min(pred) / 1000,
                "prediction_max_kw_m2": np.max(pred) / 1000,
            }
            rows.append(row)
            if snr == "clean":
                clean_rows.append(row)

        if snr == "clean":
            internal = json.loads((condition / "internal_validation_fold1.json").read_text(encoding="utf-8"))
            weights = read_csv(condition / "ensemble_weights_no_noise.csv")[0]
            for model in ("randomforest", "conformer", "alexnet"):
                external = next(r for r in clean_rows if r["direction"] == direction and r["model"] == MODEL_KEYS[model])
                audit_rows.append({
                    "direction": direction,
                    "model": MODEL_KEYS[model],
                    "internal_oof_r2": 1 - float(internal["individual_errors"][model]),
                    "performance_weight": float(weights[model]),
                    "external_r2": external["r2"],
                    "external_rmse_kw_m2": external["rmse_kw_m2"],
                })
            for experiment in sorted({sample["experiment_name"] for sample in internal["samples"]}):
                samples = [sample for sample in internal["samples"] if sample["experiment_name"] == experiment]
                target = np.asarray([float(sample["heat_flux"]) for sample in samples])
                distribution_rows.append({
                    "direction": direction,
                    "role": "train_internal_oof",
                    "experiment": experiment,
                    "n_chunks": len(samples),
                    "n_wavs": len({sample["source_wav_id"] for sample in samples}),
                    "target_min_kw_m2": np.min(target) / 1000,
                    "target_max_kw_m2": np.max(target) / 1000,
                    "target_mean_kw_m2": np.mean(target) / 1000,
                    "threshold_kw_m2": "",
                })
                for model in ("randomforest", "conformer", "alexnet"):
                    pred = np.asarray([float(sample[model]) for sample in samples])
                    internal_day_rows.append({
                        "direction": direction,
                        "training_experiment": experiment,
                        "model": MODEL_KEYS[model],
                        "oof_rmse_kw_m2": np.sqrt(np.mean((pred - target) ** 2)) / 1000,
                        "oof_bias_kw_m2": np.mean(pred - target) / 1000,
                    })
            distribution_rows.append({
                "direction": direction,
                "role": "external_test",
                "experiment": manifest["dataset"]["experiment_name"],
                "n_chunks": len(y),
                "n_wavs": len({r["source_wav_id"] for r in pred_rows}),
                "target_min_kw_m2": np.min(y) / 1000,
                "target_max_kw_m2": np.max(y) / 1000,
                "target_mean_kw_m2": np.mean(y) / 1000,
                "threshold_kw_m2": threshold / 1000,
            })

write_csv(Path(__file__).with_name("all_conditions_metrics.csv"), rows)
write_csv(Path(__file__).with_name("clean_leave_one_day_out_metrics.csv"), clean_rows)
write_csv(Path(__file__).with_name("internal_external_audit.csv"), audit_rows)
write_csv(Path(__file__).with_name("target_distributions.csv"), distribution_rows)
write_csv(Path(__file__).with_name("internal_oof_by_training_day.csv"), internal_day_rows)

for direction in RUNS:
    print(f"\n{direction}")
    for row in clean_rows:
        if row["direction"] == direction:
            print(
                f"{row['model']:20s} R2={row['r2']: .3f} RMSE={row['rmse_kw_m2']:7.1f} "
                f"bias={row['bias_kw_m2']:7.1f} slope={row['slope']:.3f} "
                f"FPR={row['pre_onb_fpr']:.3f} recall={row['post_onb_recall']:.3f}"
            )

print("\nClean macro averages")
for label in MODEL_KEYS.values():
    selected = [r for r in clean_rows if r["model"] == label]
    print(
        f"{label:20s} R2={np.mean([r['r2'] for r in selected]): .3f} "
        f"RMSE={np.mean([r['rmse_kw_m2'] for r in selected]):7.1f} "
        f"FPR={np.mean([r['pre_onb_fpr'] for r in selected]):.3f} "
        f"recall={np.mean([r['post_onb_recall'] for r in selected]):.3f}"
    )

print("\nNoise-condition macro averages (six noisy conditions x three directions)")
for label in MODEL_KEYS.values():
    selected = [r for r in rows if r["model"] == label and r["snr"] != "clean"]
    print(
        f"{label:20s} R2={np.mean([r['r2'] for r in selected]): .3f} "
        f"RMSE={np.mean([r['rmse_kw_m2'] for r in selected]):7.1f} "
        f"FPR={np.mean([r['pre_onb_fpr'] for r in selected]):.3f} "
        f"recall={np.mean([r['post_onb_recall'] for r in selected]):.3f}"
    )

print("\nTarget distributions")
for row in distribution_rows:
    print(row)

print("\nInternal OOF RMSE by training day")
for row in internal_day_rows:
    print(row)
