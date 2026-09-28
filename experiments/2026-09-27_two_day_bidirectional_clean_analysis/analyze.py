from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "0611_to_0618": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3/regression_result/npy/ensemble/202609/26/onb_xd-t0611-v0618_iw3-nc_s0_e150_000904",
    "0618_to_0611": ROOT / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/202609/27/onb_xd-t0618-v0611_iw3-nc_s0_e150_170426",
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


def condition_key(path: Path) -> tuple[int, float]:
    if path.name == "heatflux_no_noise":
        return 0, 1.0
    return 1, -float(path.name.split("=")[-1])


rows: list[dict[str, object]] = []
audit: list[dict[str, object]] = []
diversity: list[dict[str, object]] = []

for direction, run in RUNS.items():
    conditions = sorted((run / "maxfreq=22kHz").iterdir(), key=condition_key)
    fit_ids = set()
    for condition in conditions:
        completed = json.loads((condition / "completed.json").read_text())
        fit_ids.update(completed["fit_ids"])
        manifest = json.loads((condition / "run_manifest.json").read_text(encoding="utf-8"))
        threshold = float(manifest["dataset"]["threshold"])
        snr = "clean" if condition.name == "heatflux_no_noise" else condition.name.split("=")[-1]
        suffix = "no_noise" if snr == "clean" else snr
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
            rows.append({
                "direction": direction,
                "snr": snr,
                "model": label,
                "r2": 1 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2),
                "rmse_kw_m2": np.sqrt(np.mean(residual**2)) / 1000,
                "mae_kw_m2": np.mean(np.abs(residual)) / 1000,
                "bias_kw_m2": np.mean(residual) / 1000,
                "onb_rmse_kw_m2": float(summary[SUMMARY_LABELS[key]]["rmse_onb_mean"]) / 1000,
                "pre_onb_fpr": np.mean(pred[before] >= threshold),
                "post_onb_recall": np.mean(pred[after] >= threshold),
            })
        for left, right in (("RandomForest", "Conformer"), ("RandomForest", "AlexNet"), ("Conformer", "AlexNet")):
            diversity.append({
                "direction": direction,
                "snr": snr,
                "model_pair": f"{left}__{right}",
                "residual_correlation": np.corrcoef(residuals[left], residuals[right])[0, 1],
            })

        if snr == "clean":
            weights = read_csv(condition / "ensemble_weights_no_noise.csv")[0]
            internal = json.loads((condition / "internal_validation_fold1.json").read_text(encoding="utf-8"))
            for key in ("randomforest", "conformer", "alexnet"):
                external = next(r for r in rows if r["direction"] == direction and r["snr"] == "clean" and r["model"] == MODEL_KEYS[key])
                audit.append({
                    "direction": direction,
                    "model": MODEL_KEYS[key],
                    "internal_oof_r2": 1 - float(internal["individual_errors"][key]),
                    "performance_weight": float(weights[key]),
                    "external_r2": external["r2"],
                    "external_rmse_kw_m2": external["rmse_kw_m2"],
                })
    if len(conditions) != 7 or len(fit_ids) != 1:
        raise RuntimeError(f"Unexpected completion/fits for {direction}: conditions={len(conditions)}, fit_ids={fit_ids}")

write_csv(Path(__file__).with_name("metrics.csv"), rows)
write_csv(Path(__file__).with_name("weight_transfer.csv"), audit)
write_csv(Path(__file__).with_name("residual_correlations.csv"), diversity)

for direction in RUNS:
    print(f"\n{direction} clean")
    for row in rows:
        if row["direction"] == direction and row["snr"] == "clean":
            print(
                f"{row['model']:20s} R2={row['r2']: .3f} RMSE={row['rmse_kw_m2']:6.1f} "
                f"bias={row['bias_kw_m2']:6.1f} ONB={row['onb_rmse_kw_m2']:6.1f} "
                f"FPR={row['pre_onb_fpr']:.3f} recall={row['post_onb_recall']:.3f}"
            )

print("\nNoise-only averages by direction")
for direction in RUNS:
    for label in MODEL_KEYS.values():
        selected = [r for r in rows if r["direction"] == direction and r["snr"] != "clean" and r["model"] == label]
        print(
            f"{direction:12s} {label:20s} RMSE={np.mean([r['rmse_kw_m2'] for r in selected]):6.1f} "
            f"R2={np.mean([r['r2'] for r in selected]): .3f} "
            f"FPR={np.mean([r['pre_onb_fpr'] for r in selected]):.3f} "
            f"recall={np.mean([r['post_onb_recall'] for r in selected]):.3f}"
        )
