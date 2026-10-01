"""Compare the tuned 3 kHz and 22 kHz outer-holdout runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RESULT_ROOT = (
    REPO / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
    / "2025.06.11_0.3_2_6.18_0.3_3" / "regression_result" / "npy"
    / "ensemble"
)
RUNS = {
    "3kHz": RESULT_ROOT / "202610" / "01"
    / "onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_170655",
    "22kHz": RESULT_ROOT / "202609" / "30"
    / "onb_wc-t0611-v0611_iw3-nc_c1s_s0_e150_190945",
}
PRIOR_3K_RUN = (
    RESULT_ROOT / "202609" / "29"
    / "onb_wc-t0611-v0611_iw3-nc_s0_e150_161952"
)
FREQUENCY_DIR = {"3kHz": "maxfreq=3kHz", "22kHz": "maxfreq=22kHz"}
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
    text = str(path.resolve())
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


def condition_path(frequency: str, noise_dir: str) -> Path:
    return RUNS[frequency] / FREQUENCY_DIR[frequency] / noise_dir


def prediction_path(frequency: str, noise_dir: str, suffix: str) -> Path:
    return condition_path(frequency, noise_dir) / "fold_pred" / f"pred_f1_{suffix}.csv"


def prior_3k_prediction_path(noise_dir: str, suffix: str) -> Path:
    return (
        PRIOR_3K_RUN / FREQUENCY_DIR["3kHz"] / noise_dir
        / "fold_pred" / f"pred_f1_{suffix}.csv"
    )


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


def metrics(y, prediction, thresholds):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), y.shape)
    actual = y >= thresholds
    estimated = prediction >= thresholds
    pre = ~actual
    post = actual
    near = np.abs(y - thresholds) <= 0.10 * np.abs(thresholds)
    tp = int(np.sum(estimated & post))
    fp = int(np.sum(estimated & pre))
    tn = int(np.sum(~estimated & pre))
    fn = int(np.sum(~estimated & post))

    def bias(mask):
        return float(np.mean((prediction - y)[mask])) if np.any(mask) else float("nan")

    return {
        "n": len(y),
        "r2": float(r2_score(y, prediction)),
        "rmse": rmse(y, prediction),
        "mae": float(np.mean(np.abs(prediction - y))),
        "bias": float(np.mean(prediction - y)),
        "rmse_pre": rmse(y, prediction, pre),
        "bias_pre": bias(pre),
        "rmse_post": rmse(y, prediction, post),
        "bias_post": bias(post),
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
    for level in sorted(set(float(value) for value in y)):
        if np.all(prediction[y >= level] >= threshold):
            return level, level - threshold
    return float("nan"), float("nan")


def source_thresholds(rows):
    return np.asarray([THRESHOLDS[source_day(row)] for row in rows], dtype=float)


def cluster_indices(rows):
    clusters = sorted({row["source_wav_id"] for row in rows})
    return clusters, {
        cluster: np.asarray(
            [index for index, row in enumerate(rows) if row["source_wav_id"] == cluster],
            dtype=int,
        )
        for cluster in clusters
    }


def bootstrap_frequency_delta(rows_3k, rows_22k, model, rng, repeats=5000):
    """Return paired 3 kHz minus 22 kHz deltas by source-WAV bootstrap."""
    if [identity(row) for row in rows_3k] != [identity(row) for row in rows_22k]:
        raise ValueError("Frequency comparison samples are not aligned")
    y, prediction_3k = arrays(rows_3k, model)
    y_22k, prediction_22k = arrays(rows_22k, model)
    if not np.array_equal(y, y_22k):
        raise ValueError("Frequency comparison targets differ")
    thresholds = source_thresholds(rows_3k)
    clusters, indices = cluster_indices(rows_3k)

    def values(selected):
        selected_y = y[selected]
        selected_thresholds = thresholds[selected]
        near = np.abs(selected_y - selected_thresholds) <= 0.10 * np.abs(selected_thresholds)
        pre = selected_y < selected_thresholds
        post = ~pre
        p3 = prediction_3k[selected]
        p22 = prediction_22k[selected]
        return {
            "rmse": rmse(selected_y, p3) - rmse(selected_y, p22),
            "rmse_onb": rmse(selected_y, p3, near) - rmse(selected_y, p22, near),
            "fpr": float(np.mean(p3[pre] >= selected_thresholds[pre])
                         - np.mean(p22[pre] >= selected_thresholds[pre])),
            "recall": float(np.mean(p3[post] >= selected_thresholds[post])
                            - np.mean(p22[post] >= selected_thresholds[post])),
        }

    point = values(np.arange(len(y)))
    draws = {name: [] for name in point}
    for _ in range(repeats):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        for name, value in values(selected).items():
            draws[name].append(value)
    result = {}
    for name, values_ in draws.items():
        values_ = np.asarray(values_)
        finite_values = values_[np.isfinite(values_)]
        result[f"delta_{name}"] = point[name]
        result[f"n_valid_bootstrap_{name}"] = int(len(finite_values))
        result[f"ci95_low_{name}"] = float(np.quantile(finite_values, 0.025))
        result[f"ci95_high_{name}"] = float(np.quantile(finite_values, 0.975))
        if name == "recall":
            result[f"probability_3k_better_{name}"] = float(np.mean(finite_values > 0))
        else:
            result[f"probability_3k_better_{name}"] = float(np.mean(finite_values < 0))
    return result


def main():
    loaded = {}
    completion_rows = []
    reference_keys = None
    reference_targets = None

    for frequency in RUNS:
        reference_fit_id = None
        for noise_dir, suffix, noise, noise_order in NOISES:
            condition = condition_path(frequency, noise_dir)
            completed = read_json(condition / "completed.json")
            manifest = read_json(condition / "run_manifest.json")
            split = read_json(condition / "split_manifest.json")["folds"][0]
            rows = read_rows(prediction_path(frequency, noise_dir, suffix))
            loaded[(frequency, noise)] = rows
            keys = [identity(row) for row in rows]
            targets = np.asarray([float(row["y_true"]) for row in rows])
            if reference_keys is None:
                reference_keys, reference_targets = keys, targets
            if reference_fit_id is None:
                reference_fit_id = completed["fit_ids"][0]
            completion_rows.append({
                "frequency": frequency,
                "noise": noise,
                "noise_order": noise_order,
                "completed": 1,
                "n_predictions": len(rows),
                "same_ordered_samples_as_reference": int(keys == reference_keys),
                "max_target_abs_difference_from_reference": float(np.max(np.abs(targets - reference_targets))),
                "n_0611": sum(source_day(row) == "2025.06.11_0.3_2" for row in rows),
                "n_0618": sum(source_day(row) == "2025.06.18_0.3_3" for row in rows),
                "n_training_chunks": split["n_training_chunks"],
                "n_evaluation_chunks": split["n_evaluation_chunks"],
                "n_training_wavs": len(split["training_wav_groups"]),
                "n_evaluation_wavs": len(split["evaluation_wav_groups"]),
                "fit_id": completed["fit_ids"][0],
                "same_fit_within_frequency": int(completed["fit_ids"][0] == reference_fit_id),
                "selection_enabled": int(manifest["validation_config"]["acoustic_selection"]["enabled"]),
                "training_noise": manifest["learning_context"]["training_noise"],
                "parameter_set_name": manifest["parameter_set_name"],
            })
    write_rows("completion_audit.csv", completion_rows)

    metric_rows = []
    q100_rows = []
    for frequency in RUNS:
        for _, _, noise, noise_order in NOISES:
            rows = loaded[(frequency, noise)]
            for day, threshold in THRESHOLDS.items():
                day_rows = [row for row in rows if source_day(row) == day]
                for model, label in MODELS.items():
                    y, prediction = arrays(day_rows, model)
                    metric_rows.append({
                        "scope": day,
                        "frequency": frequency,
                        "noise": noise,
                        "noise_order": noise_order,
                        "model": label,
                        **metrics(y, prediction, threshold),
                    })
                    q, gap = q100(y, prediction, threshold)
                    q100_rows.append({
                        "experiment": day,
                        "frequency": frequency,
                        "noise": noise,
                        "noise_order": noise_order,
                        "model": label,
                        "onb_threshold": threshold,
                        "q100": q,
                        "g100": gap,
                    })
            thresholds = source_thresholds(rows)
            for model, label in MODELS.items():
                y, prediction = arrays(rows, model)
                metric_rows.append({
                    "scope": "two_day_source_threshold",
                    "frequency": frequency,
                    "noise": noise,
                    "noise_order": noise_order,
                    "model": label,
                    **metrics(y, prediction, thresholds),
                })
    write_rows("metrics_source_day_thresholds.csv", metric_rows)
    write_rows("q100_source_day_thresholds.csv", q100_rows)

    heatflux_rows = []
    for frequency in RUNS:
        for _, _, noise, noise_order in NOISES:
            rows = loaded[(frequency, noise)]
            for day, threshold in THRESHOLDS.items():
                day_rows = [row for row in rows if source_day(row) == day]
                heatflux_levels = sorted({float(row["y_true"]) for row in day_rows})
                for heatflux in heatflux_levels:
                    level_rows = [row for row in day_rows if float(row["y_true"]) == heatflux]
                    y, prediction = arrays(level_rows, "ensemble__performance_kfold")
                    heatflux_rows.append({
                        "frequency": frequency,
                        "noise": noise,
                        "noise_order": noise_order,
                        "experiment": day,
                        "heatflux": heatflux,
                        "onb_threshold": threshold,
                        "relative_to_onb": "pre" if heatflux < threshold else "post",
                        "n": len(y),
                        "prediction_mean": float(np.mean(prediction)),
                        "bias": float(np.mean(prediction - y)),
                        "rmse": rmse(y, prediction),
                        "predicted_boiling_rate": float(np.mean(prediction >= threshold)),
                    })
    write_rows("heatflux_metrics_performance.csv", heatflux_rows)

    aggregate_rows = []
    for frequency in RUNS:
        for label in MODELS.values():
            rows = [
                row for row in metric_rows
                if row["scope"] == "two_day_source_threshold"
                and row["frequency"] == frequency and row["model"] == label
            ]
            by_noise = {row["noise"]: row for row in rows}
            noise_rows = [by_noise[str(value)] for value in (0, -4, -8, -12, -16, -20)]
            strong = [by_noise[str(value)] for value in (-12, -16, -20)]
            aggregate_rows.append({
                "frequency": frequency,
                "model": label,
                "clean_rmse": by_noise["clean"]["rmse"],
                "clean_rmse_onb": by_noise["clean"]["rmse_onb"],
                "clean_fpr": by_noise["clean"]["fpr"],
                "clean_recall": by_noise["clean"]["recall"],
                "noise_mean_rmse": float(np.mean([row["rmse"] for row in noise_rows])),
                "noise_mean_rmse_onb": float(np.mean([row["rmse_onb"] for row in noise_rows])),
                "noise_mean_fpr": float(np.mean([row["fpr"] for row in noise_rows])),
                "noise_mean_recall": float(np.mean([row["recall"] for row in noise_rows])),
                "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong])),
                "strong_mean_rmse_onb": float(np.mean([row["rmse_onb"] for row in strong])),
                "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong])),
                "strong_mean_recall": float(np.mean([row["recall"] for row in strong])),
                "minus20_rmse": by_noise["-20"]["rmse"],
                "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
                "minus20_bias_pre": by_noise["-20"]["bias_pre"],
                "minus20_fpr": by_noise["-20"]["fpr"],
                "minus20_recall": by_noise["-20"]["recall"],
            })
    write_rows("aggregate_two_day.csv", aggregate_rows)

    frequency_rows = []
    for _, _, noise, noise_order in NOISES:
        for label in MODELS.values():
            row_3k = next(row for row in metric_rows if row["scope"] == "two_day_source_threshold"
                          and row["frequency"] == "3kHz" and row["noise"] == noise
                          and row["model"] == label)
            row_22k = next(row for row in metric_rows if row["scope"] == "two_day_source_threshold"
                           and row["frequency"] == "22kHz" and row["noise"] == noise
                           and row["model"] == label)
            frequency_rows.append({
                "noise": noise,
                "noise_order": noise_order,
                "model": label,
                **{f"3k_{name}": row_3k[name] for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
                **{f"22k_{name}": row_22k[name] for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
                **{f"delta_3k_minus_22k_{name}": row_3k[name] - row_22k[name]
                   for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
            })
    write_rows("frequency_comparison.csv", frequency_rows)

    ensemble_rows = []
    for frequency in RUNS:
        for _, _, noise, noise_order in NOISES:
            rows = [row for row in metric_rows if row["scope"] == "two_day_source_threshold"
                    and row["frequency"] == frequency and row["noise"] == noise]
            by_model = {row["model"]: row for row in rows}
            best_single = min(("RF", "Conformer", "AlexNet"), key=lambda name: by_model[name]["rmse"])
            ensemble_rows.append({
                "frequency": frequency,
                "noise": noise,
                "noise_order": noise_order,
                "performance_rmse": by_model["performance"]["rmse"],
                "equal_rmse": by_model["equal"]["rmse"],
                "best_single_posthoc": best_single,
                "best_single_rmse_posthoc": by_model[best_single]["rmse"],
                "performance_minus_equal_rmse": by_model["performance"]["rmse"] - by_model["equal"]["rmse"],
                "performance_minus_best_single_rmse_posthoc": by_model["performance"]["rmse"] - by_model[best_single]["rmse"],
            })
    write_rows("ensemble_comparison.csv", ensemble_rows)

    weight_rows = []
    for frequency in RUNS:
        for noise_dir, suffix, noise, noise_order in NOISES:
            for row in read_rows(condition_path(frequency, noise_dir) / f"ensemble_weights_{suffix}.csv"):
                weight_rows.append({"frequency": frequency, "noise": noise, "noise_order": noise_order, **row})
    write_rows("weights.csv", weight_rows)

    rng = np.random.default_rng(20261001)
    bootstrap_rows = []
    for _, _, noise, noise_order in NOISES:
        for model, label in MODELS.items():
            bootstrap_rows.append({
                "noise": noise,
                "noise_order": noise_order,
                "model": label,
                "n_source_wavs": len({row["source_wav_id"] for row in loaded[("3kHz", noise)]}),
                "n_bootstrap": 5000,
                **bootstrap_frequency_delta(
                    loaded[("3kHz", noise)], loaded[("22kHz", noise)], model, rng
                ),
            })
    write_rows("cluster_bootstrap_frequency_delta.csv", bootstrap_rows)

    prior_3k = {}
    tuning_rows = []
    tuning_q100_rows = []
    tuning_bootstrap_rows = []
    tuning_rng = np.random.default_rng(20261002)
    for noise_dir, suffix, noise, noise_order in NOISES:
        current_rows = loaded[("3kHz", noise)]
        prior_rows = read_rows(prior_3k_prediction_path(noise_dir, suffix))
        prior_3k[noise] = prior_rows
        if [identity(row) for row in current_rows] != [identity(row) for row in prior_rows]:
            raise ValueError(f"Prior and tuned 3 kHz samples are not aligned for noise={noise}")
        y_current, current_prediction = arrays(current_rows, "ensemble__performance_kfold")
        y_prior, prior_prediction = arrays(prior_rows, "ensemble__performance_kfold")
        if not np.array_equal(y_current, y_prior):
            raise ValueError(f"Prior and tuned 3 kHz targets differ for noise={noise}")
        thresholds = source_thresholds(current_rows)
        current_metrics = metrics(y_current, current_prediction, thresholds)
        prior_metrics = metrics(y_prior, prior_prediction, thresholds)
        tuning_rows.append({
            "noise": noise,
            "noise_order": noise_order,
            **{f"tuned_{name}": current_metrics[name]
               for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
            **{f"prior_{name}": prior_metrics[name]
               for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
            **{f"delta_tuned_minus_prior_{name}": current_metrics[name] - prior_metrics[name]
               for name in ("rmse", "rmse_onb", "bias_pre", "fpr", "recall")},
        })
        bootstrap = bootstrap_frequency_delta(
            current_rows, prior_rows, "ensemble__performance_kfold", tuning_rng
        )
        tuning_bootstrap_rows.append({
            "noise": noise,
            "noise_order": noise_order,
            "n_source_wavs": len({row["source_wav_id"] for row in current_rows}),
            "n_bootstrap": 5000,
            **{
                key.replace("probability_3k_better", "probability_tuned_better"): value
                for key, value in bootstrap.items()
            },
        })
        for day, threshold in THRESHOLDS.items():
            current_day = [row for row in current_rows if source_day(row) == day]
            prior_day = [row for row in prior_rows if source_day(row) == day]
            current_y, current_p = arrays(current_day, "ensemble__performance_kfold")
            prior_y, prior_p = arrays(prior_day, "ensemble__performance_kfold")
            current_q100, current_g100 = q100(current_y, current_p, threshold)
            prior_q100, prior_g100 = q100(prior_y, prior_p, threshold)
            tuning_q100_rows.append({
                "experiment": day,
                "noise": noise,
                "noise_order": noise_order,
                "tuned_q100": current_q100,
                "prior_q100": prior_q100,
                "tuned_g100": current_g100,
                "prior_g100": prior_g100,
                "same_q100": int(current_q100 == prior_q100),
            })
    write_rows("tuning_comparison.csv", tuning_rows)
    write_rows("cluster_bootstrap_tuning_delta.csv", tuning_bootstrap_rows)
    write_rows("tuning_q100_comparison.csv", tuning_q100_rows)

    print("analysis complete")
    print(f"completion rows: {len(completion_rows)}")
    print(f"metric rows: {len(metric_rows)}")
    print(f"bootstrap rows: {len(bootstrap_rows)}")


if __name__ == "__main__":
    main()
