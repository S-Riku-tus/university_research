"""Audit and summarize the tuned 22 kHz outer holdout run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUN = (
    REPO / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
    / "2025.06.11_0.3_2_6.18_0.3_3" / "regression_result" / "npy"
    / "ensemble" / "202609" / "30"
    / "onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_190945"
)
THRESHOLDS = {
    "2025.06.11_0.3_2": 221505.1102,
    "2025.06.18_0.3_3": 271677.6816,
}
SOURCE_PREFIX = {
    "20250611": "2025.06.11_0.3_2",
    "20250618": "2025.06.18_0.3_3",
}
NOISES = [
    ("heatflux_no_noise", "no_noise", "clean", 1),
    ("heatflux_reference_SNR=0", "0", "0", 0),
    ("heatflux_reference_SNR=-4", "-4", "-4", -4),
    ("heatflux_reference_SNR=-8", "-8", "-8", -8),
    ("heatflux_reference_SNR=-12", "-12", "-12", -12),
    ("heatflux_reference_SNR=-16", "-16", "-16", -16),
    ("heatflux_reference_SNR=-20", "-20", "-20", -20),
]
MODELS = {
    "randomforest": "RF",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "performance",
    "ensemble__simple_equal": "equal",
}


def long_path(path: Path) -> Path:
    resolved = path.resolve()
    text = str(resolved)
    return Path(text if text.startswith("\\\\?\\") else "\\\\?\\" + text)


def read_rows(path: Path):
    with long_path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def read_json(path: Path):
    return json.loads(long_path(path).read_text(encoding="utf-8-sig"))


def write_rows(name: str, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_path(noise_dir: str) -> Path:
    return RUN / "maxfreq=22kHz" / noise_dir


def prediction_path(noise_dir: str, suffix: str) -> Path:
    return condition_path(noise_dir) / "fold_pred" / f"pred_f1_{suffix}.csv"


def source_day(row) -> str:
    return SOURCE_PREFIX[row["source_wav_id"].split("::", 1)[0]]


def identity(row):
    return source_day(row), row["source_wav_id"], int(float(row["chunk_index"]))


def arrays(rows, model: str):
    return (
        np.asarray([float(row["y_true"]) for row in rows], dtype=float),
        np.asarray([float(row[model]) for row in rows], dtype=float),
    )


def rmse(y, prediction, mask=None) -> float:
    if mask is not None:
        y, prediction = y[mask], prediction[mask]
    return float(np.sqrt(np.mean((prediction - y) ** 2))) if len(y) else float("nan")


def metrics(y, prediction, sample_thresholds):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    threshold = np.broadcast_to(np.asarray(sample_thresholds, dtype=float), y.shape)
    actual = y >= threshold
    estimated = prediction >= threshold
    pre = ~actual
    post = actual
    near = np.abs(y - threshold) <= 0.10 * np.abs(threshold)
    tp = int(np.sum(estimated & post))
    fp = int(np.sum(estimated & pre))
    tn = int(np.sum(~estimated & pre))
    fn = int(np.sum(~estimated & post))

    def mean_error(mask):
        return float(np.mean((prediction - y)[mask])) if np.any(mask) else float("nan")

    return {
        "n": len(y),
        "r2": float(r2_score(y, prediction)),
        "rmse": rmse(y, prediction),
        "mae": float(np.mean(np.abs(prediction - y))),
        "bias": float(np.mean(prediction - y)),
        "rmse_pre": rmse(y, prediction, pre),
        "bias_pre": mean_error(pre),
        "rmse_post": rmse(y, prediction, post),
        "bias_post": mean_error(post),
        "rmse_onb": rmse(y, prediction, near),
        "n_pre": int(pre.sum()),
        "n_post": int(post.sum()),
        "n_onb": int(near.sum()),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "roc_auc": float(roc_auc_score(actual, prediction)),
        "pr_auc": float(average_precision_score(actual, prediction)),
        "fp": fp,
        "fn": fn,
    }


def q100(y, prediction, threshold):
    """First measured heat flux from which every later sample is predicted boiling."""
    for level in sorted(set(float(value) for value in y)):
        if np.all(prediction[y >= level] >= threshold):
            return level, level - threshold
    return float("nan"), float("nan")


def bootstrap_rmse_delta(rows, model_a: str, model_b: str, rng, repeats=5000):
    """Return RMSE(model_a)-RMSE(model_b), resampling source WAV groups."""
    y, prediction_a = arrays(rows, model_a)
    _, prediction_b = arrays(rows, model_b)
    clusters = sorted({row["source_wav_id"] for row in rows})
    indices = {
        cluster: np.asarray(
            [index for index, row in enumerate(rows) if row["source_wav_id"] == cluster],
            dtype=int,
        )
        for cluster in clusters
    }
    deltas = []
    for _ in range(repeats):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        deltas.append(
            rmse(y[selected], prediction_a[selected])
            - rmse(y[selected], prediction_b[selected])
        )
    deltas = np.asarray(deltas)
    point = rmse(y, prediction_a) - rmse(y, prediction_b)
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975)), float(np.mean(deltas < 0))


def main():
    loaded = {}
    completion_rows = []
    reference_keys = None
    reference_targets = None
    reference_fit_id = None

    for noise_dir, suffix, noise, noise_order in NOISES:
        condition = condition_path(noise_dir)
        completed = read_json(condition / "completed.json")
        manifest = read_json(condition / "run_manifest.json")
        split = read_json(condition / "split_manifest.json")["folds"][0]
        rows = read_rows(prediction_path(noise_dir, suffix))
        loaded[noise] = rows
        keys = [identity(row) for row in rows]
        targets = np.asarray([float(row["y_true"]) for row in rows])
        if reference_keys is None:
            reference_keys = keys
            reference_targets = targets
            reference_fit_id = completed["fit_ids"][0]
        completion_rows.append({
            "noise": noise,
            "noise_order": noise_order,
            "completed": 1,
            "n_predictions": len(rows),
            "same_ordered_samples_as_clean": int(keys == reference_keys),
            "max_target_abs_difference_from_clean": float(np.max(np.abs(targets - reference_targets))),
            "n_0611": sum(source_day(row) == "2025.06.11_0.3_2" for row in rows),
            "n_0618": sum(source_day(row) == "2025.06.18_0.3_3" for row in rows),
            "n_training_chunks": split["n_training_chunks"],
            "n_evaluation_chunks": split["n_evaluation_chunks"],
            "n_training_wavs": len(split["training_wav_groups"]),
            "n_evaluation_wavs": len(split["evaluation_wav_groups"]),
            "fit_id": completed["fit_ids"][0],
            "same_fit_as_clean": int(completed["fit_ids"][0] == reference_fit_id),
            "selection_enabled": int(manifest["validation_config"]["acoustic_selection"]["enabled"]),
            "training_noise": manifest["learning_context"]["training_noise"],
            "evaluation_scheme": manifest["learning_context"]["evaluation_scheme"],
        })
    write_rows("completion_audit.csv", completion_rows)

    metric_rows = []
    q100_rows = []
    wav_rows = []
    for _, _, noise, noise_order in NOISES:
        rows = loaded[noise]
        for day, threshold in THRESHOLDS.items():
            day_rows = [row for row in rows if source_day(row) == day]
            for model, label in MODELS.items():
                y, prediction = arrays(day_rows, model)
                metric_rows.append({
                    "scope": day,
                    "noise": noise,
                    "noise_order": noise_order,
                    "model": label,
                    **metrics(y, prediction, threshold),
                })
                q, gap = q100(y, prediction, threshold)
                q100_rows.append({
                    "experiment": day,
                    "noise": noise,
                    "noise_order": noise_order,
                    "model": label,
                    "onb_threshold": threshold,
                    "q100": q,
                    "g100": gap,
                })
        thresholds = np.asarray([THRESHOLDS[source_day(row)] for row in rows])
        for model, label in MODELS.items():
            y, prediction = arrays(rows, model)
            metric_rows.append({
                "scope": "two_day_source_threshold",
                "noise": noise,
                "noise_order": noise_order,
                "model": label,
                **metrics(y, prediction, thresholds),
            })
        for wav in sorted({row["source_wav_id"] for row in rows}):
            wav_subset = [row for row in rows if row["source_wav_id"] == wav]
            day = source_day(wav_subset[0])
            for model, label in MODELS.items():
                y, prediction = arrays(wav_subset, model)
                wav_rows.append({
                    "noise": noise,
                    "noise_order": noise_order,
                    "experiment": day,
                    "source_wav_id": wav,
                    "heatflux": float(y[0]),
                    "n": len(y),
                    "model": label,
                    "rmse": rmse(y, prediction),
                    "bias": float(np.mean(prediction - y)),
                })
    write_rows("metrics_source_day_thresholds.csv", metric_rows)
    write_rows("q100_source_day_thresholds.csv", q100_rows)
    write_rows("wav_metrics.csv", wav_rows)

    aggregate_rows = []
    for label in MODELS.values():
        rows = [row for row in metric_rows if row["scope"] == "two_day_source_threshold" and row["model"] == label]
        by_noise = {row["noise"]: row for row in rows}
        strong = [by_noise[noise] for noise in ("-12", "-16", "-20")]
        aggregate_rows.append({
            "model": label,
            "clean_rmse": by_noise["clean"]["rmse"],
            "clean_rmse_onb": by_noise["clean"]["rmse_onb"],
            "clean_fpr": by_noise["clean"]["fpr"],
            "clean_recall": by_noise["clean"]["recall"],
            "noise_mean_rmse": float(np.mean([by_noise[str(value)]["rmse"] for value in (0, -4, -8, -12, -16, -20)])),
            "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong])),
            "strong_mean_rmse_onb": float(np.mean([row["rmse_onb"] for row in strong])),
            "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong])),
            "strong_mean_recall": float(np.mean([row["recall"] for row in strong])),
            "minus20_rmse": by_noise["-20"]["rmse"],
            "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
            "minus20_fpr": by_noise["-20"]["fpr"],
            "minus20_recall": by_noise["-20"]["recall"],
            "minus20_bias_pre": by_noise["-20"]["bias_pre"],
        })
    write_rows("aggregate_two_day.csv", aggregate_rows)

    comparison_rows = []
    for _, _, noise, noise_order in NOISES:
        rows = [row for row in metric_rows if row["scope"] == "two_day_source_threshold" and row["noise"] == noise]
        by_model = {row["model"]: row for row in rows}
        best_single_name = min(("RF", "Conformer", "AlexNet"), key=lambda name: by_model[name]["rmse"])
        comparison_rows.append({
            "noise": noise,
            "noise_order": noise_order,
            "performance_rmse": by_model["performance"]["rmse"],
            "equal_rmse": by_model["equal"]["rmse"],
            "best_single_posthoc": best_single_name,
            "best_single_rmse_posthoc": by_model[best_single_name]["rmse"],
            "performance_minus_equal_rmse": by_model["performance"]["rmse"] - by_model["equal"]["rmse"],
            "performance_minus_best_single_rmse_posthoc": by_model["performance"]["rmse"] - by_model[best_single_name]["rmse"],
            "performance_rmse_onb": by_model["performance"]["rmse_onb"],
            "performance_fpr": by_model["performance"]["fpr"],
            "performance_recall": by_model["performance"]["recall"],
        })
    write_rows("ensemble_comparison.csv", comparison_rows)

    weight_rows = []
    for noise_dir, suffix, noise, noise_order in NOISES:
        for row in read_rows(condition_path(noise_dir) / f"ensemble_weights_{suffix}.csv"):
            weight_rows.append({"noise": noise, "noise_order": noise_order, **row})
    write_rows("weights.csv", weight_rows)

    rng = np.random.default_rng(20261001)
    bootstrap_rows = []
    comparisons = {
        "performance_minus_equal": ("ensemble__performance_kfold", "ensemble__simple_equal"),
        "performance_minus_rf": ("ensemble__performance_kfold", "randomforest"),
        "performance_minus_conformer": ("ensemble__performance_kfold", "conformer"),
        "performance_minus_alexnet": ("ensemble__performance_kfold", "alexnet"),
    }
    for _, _, noise, noise_order in NOISES:
        for comparison, (model_a, model_b) in comparisons.items():
            point, low, high, probability = bootstrap_rmse_delta(
                loaded[noise], model_a, model_b, rng
            )
            bootstrap_rows.append({
                "noise": noise,
                "noise_order": noise_order,
                "comparison": comparison,
                "n_source_wavs": len({row["source_wav_id"] for row in loaded[noise]}),
                "n_bootstrap": 5000,
                "point_delta_rmse": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_probability_performance_better": probability,
            })
    write_rows("cluster_bootstrap_ensemble_comparison.csv", bootstrap_rows)

    print("analysis complete")
    print(f"run: {RUN}")
    print(f"conditions: {len(completion_rows)}")
    print(f"metric rows: {len(metric_rows)}")


if __name__ == "__main__":
    main()
