"""ONB保護付き2.1--2.5 kHzピーク選別を、選別なしrunと対応比較する。"""

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
    / "ensemble" / "202609" / "29"
)
RUNS = {
    "baseline": RESULT_ROOT / "onb_wc-t0611-v0611_iw3-nc_s0_e150_161952",
    "selected": RESULT_ROOT / "onb_wc-t0611-v0611_iw3-nc_c1s_s1e-9b10_e150_194751",
}
THRESHOLDS = {
    "2025.06.11_0.3_2": 221505.1102,
    "2025.06.18_0.3_3": 271677.6816,
}
SOURCE_PREFIX = {
    "20250611": "2025.06.11_0.3_2",
    "20250618": "2025.06.18_0.3_3",
}
NOISES = [
    ("heatflux_no_noise", "no_noise", "clean"),
    ("heatflux_reference_SNR=0", "0", "0"),
    ("heatflux_reference_SNR=-4", "-4", "-4"),
    ("heatflux_reference_SNR=-8", "-8", "-8"),
    ("heatflux_reference_SNR=-12", "-12", "-12"),
    ("heatflux_reference_SNR=-16", "-16", "-16"),
    ("heatflux_reference_SNR=-20", "-20", "-20"),
]
MODELS = {
    "randomforest": "RF",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__performance_kfold": "performance",
    "ensemble__simple_equal": "equal",
}
SINGLE_MODELS = ("randomforest", "conformer", "alexnet")


def long_path(path: Path) -> Path:
    """Allow reading result paths longer than the legacy Windows limit."""
    resolved = path.resolve()
    return Path("\\\\?\\" + str(resolved)) if str(resolved)[:4] != "\\\\?\\" else resolved


def read_rows(path: Path):
    with long_path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def read_json(path: Path):
    return json.loads(long_path(path).read_text(encoding="utf-8"))


def write_rows(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(run, noise_dir, suffix):
    return (run / "maxfreq=3kHz" / noise_dir / "fold_pred"
            / f"pred_f1_{suffix}.csv")


def condition_path(run, noise_dir):
    return run / "maxfreq=3kHz" / noise_dir


def source_day(row):
    return SOURCE_PREFIX[row["source_wav_id"].split("::", 1)[0]]


def identity(row):
    return source_day(row), row["source_wav_id"], int(float(row["chunk_index"]))


def arrays(rows, model):
    return (
        np.asarray([float(row["y_true"]) for row in rows]),
        np.asarray([float(row[model]) for row in rows]),
    )


def rmse(y, prediction, mask=None):
    if mask is not None:
        y, prediction = y[mask], prediction[mask]
    return float(np.sqrt(np.mean((prediction - y) ** 2))) if len(y) else float("nan")


def metrics(y, prediction, sample_thresholds):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    sample_thresholds = np.broadcast_to(np.asarray(sample_thresholds, dtype=float), y.shape)
    actual = y >= sample_thresholds
    estimated = prediction >= sample_thresholds
    pre = ~actual
    post = actual
    near = np.abs(y - sample_thresholds) <= 0.10 * np.abs(sample_thresholds)
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
    for level in sorted(set(float(value) for value in y)):
        if np.all(prediction[y >= level] >= threshold):
            return level, level - threshold
    return float("nan"), float("nan")


def main():
    loaded = {}
    completion_rows = []
    split_rows = []
    for run_name, run in RUNS.items():
        for noise_dir, suffix, noise in NOISES:
            prediction_file = prediction_path(run, noise_dir, suffix)
            rows = read_rows(prediction_file)
            if len(rows) != 540:
                raise ValueError(f"{run_name}/{noise}: prediction count={len(rows)}")
            loaded[(run_name, noise)] = rows
            condition = condition_path(run, noise_dir)
            completed = read_json(condition / "completed.json")
            manifest = read_json(condition / "run_manifest.json")
            split = read_json(condition / "split_manifest.json")["folds"][0]
            completion_rows.append({
                "run": run_name,
                "noise": noise,
                "completed": 1,
                "run_hash": completed["run_hash"],
                "fit_id": completed["fit_ids"][0],
                "n_fit_ids": len(completed["fit_ids"]),
                "selection_enabled": int(manifest["validation_config"]["acoustic_selection"]["enabled"]),
                "random_seed": manifest["validation_config"]["run"]["random_seed"],
                "epochs": manifest["validation_config"]["run"]["epochs"],
                "n_training_chunks": split["n_training_chunks"],
                "n_evaluation_chunks": split["n_evaluation_chunks"],
                "n_training_wavs": len(split["training_wav_groups"]),
                "n_evaluation_wavs": len(split["evaluation_wav_groups"]),
            })
    write_rows("completion_audit.csv", completion_rows)

    reference_keys = None
    reference_targets = None
    for run_name in RUNS:
        for _, _, noise in NOISES:
            rows = loaded[(run_name, noise)]
            keys = [identity(row) for row in rows]
            targets = np.asarray([float(row["y_true"]) for row in rows])
            if reference_keys is None:
                reference_keys, reference_targets = keys, targets
            split_rows.append({
                "run": run_name,
                "noise": noise,
                "n": len(rows),
                "same_ordered_samples_as_reference": int(keys == reference_keys),
                "max_target_abs_difference": float(np.max(np.abs(targets - reference_targets))),
                "n_0611": sum(source_day(row) == "2025.06.11_0.3_2" for row in rows),
                "n_0618": sum(source_day(row) == "2025.06.18_0.3_3" for row in rows),
            })
    write_rows("split_audit.csv", split_rows)

    selection_rows = []
    removed_group_rows = []
    for run_name, run in RUNS.items():
        audit = read_json(condition_path(run, "heatflux_no_noise")
                          / "training_selection_fold1.json")
        selection_rows.append({
            "run": run_name,
            "scope": "outer_fit",
            "source_day": "all",
            "enabled": int(audit["enabled"]),
            "n_before": audit["n_before"],
            "n_after": audit["n_after"],
            "n_removed": audit["n_before"] - audit["n_after"],
            "n_protected": sum(item.get("protected_chunks", 0)
                               for item in audit.get("by_experiment", {}).values()),
        })
        for day, values in audit.get("by_experiment", {}).items():
            selection_rows.append({
                "run": run_name,
                "scope": "outer_fit",
                "source_day": day,
                "enabled": int(audit["enabled"]),
                "n_before": "",
                "n_after": "",
                "n_removed": values["excluded_chunks"],
                "n_protected": values["protected_chunks"],
            })
        if run_name == "selected":
            groups = {}
            for decision in audit.get("decisions", []):
                key = (
                    decision["source_experiment_name"],
                    decision["original_source_wav_id"],
                    float(decision["heat_flux"]),
                )
                group = groups.setdefault(key, {
                    "n_training_chunks": 0,
                    "n_removed": 0,
                    "n_protected": 0,
                })
                group["n_training_chunks"] += 1
                group["n_removed"] += int(not decision["keep"])
                group["n_protected"] += int(decision.get("protected_onb_band", False))
            for (day, wav, heat_flux), values in sorted(groups.items()):
                if values["n_removed"]:
                    removed_group_rows.append({
                        "source_day": day,
                        "source_wav_id": wav,
                        "heatflux": heat_flux,
                        **values,
                        "removed_fraction": (
                            values["n_removed"] / values["n_training_chunks"]),
                    })
        internal = read_json(condition_path(run, "heatflux_no_noise")
                             / "internal_validation_fold1.json")
        for fold in internal["folds"]:
            selected = fold["selection"]
            selection_rows.append({
                "run": run_name,
                "scope": f"inner_fit_{fold['fold']}",
                "source_day": "all",
                "enabled": int(selected["enabled"]),
                "n_before": selected["n_before"],
                "n_after": selected["n_after"],
                "n_removed": selected["n_before"] - selected["n_after"],
                "n_protected": sum(item.get("protected_chunks", 0)
                                   for item in selected.get("by_experiment", {}).values()),
            })
    write_rows("selection_audit.csv", selection_rows)
    write_rows("removed_training_groups.csv", removed_group_rows)

    metric_rows = []
    q100_rows = []
    wav_delta_rows = []
    for run_name in RUNS:
        for _, _, noise in NOISES:
            rows = loaded[(run_name, noise)]
            for day, threshold in THRESHOLDS.items():
                day_rows = [row for row in rows if source_day(row) == day]
                for model, label in MODELS.items():
                    y, prediction = arrays(day_rows, model)
                    result = metrics(y, prediction, threshold)
                    metric_rows.append({
                        "scope": day,
                        "run": run_name,
                        "noise": noise,
                        "model": label,
                        **result,
                    })
                    q, gap = q100(y, prediction, threshold)
                    q100_rows.append({
                        "experiment": day,
                        "run": run_name,
                        "noise": noise,
                        "model": label,
                        "q100": q,
                        "g100": gap,
                    })
            thresholds = np.asarray([THRESHOLDS[source_day(row)] for row in rows])
            for model, label in MODELS.items():
                y, prediction = arrays(rows, model)
                metric_rows.append({
                    "scope": "two_day",
                    "run": run_name,
                    "noise": noise,
                    "model": label,
                    **metrics(y, prediction, thresholds),
                })
    write_rows("metrics.csv", metric_rows)
    write_rows("q100.csv", q100_rows)

    delta_rows = []
    metric_names = [
        "rmse", "mae", "bias", "rmse_pre", "bias_pre", "rmse_post",
        "bias_post", "rmse_onb", "fpr", "recall", "precision", "f1",
        "roc_auc", "pr_auc", "fp", "fn",
    ]
    for selected in [row for row in metric_rows if row["run"] == "selected"]:
        baseline = next(
            row for row in metric_rows
            if row["run"] == "baseline" and row["scope"] == selected["scope"]
            and row["noise"] == selected["noise"] and row["model"] == selected["model"]
        )
        delta_rows.append({
            "scope": selected["scope"],
            "noise": selected["noise"],
            "model": selected["model"],
            **{f"baseline_{name}": baseline[name] for name in metric_names},
            **{f"selected_{name}": selected[name] for name in metric_names},
            **{f"delta_{name}": selected[name] - baseline[name] for name in metric_names},
        })
    write_rows("selected_minus_baseline.csv", delta_rows)

    aggregate_rows = []
    for run_name in RUNS:
        for model in MODELS.values():
            rows = [row for row in metric_rows if row["scope"] == "two_day"
                    and row["run"] == run_name and row["model"] == model]
            by_noise = {row["noise"]: row for row in rows}
            noise_rows = [row for row in rows if row["noise"] != "clean"]
            strong_rows = [row for row in rows if row["noise"] in {"-12", "-16", "-20"}]
            aggregate_rows.append({
                "run": run_name,
                "model": model,
                "clean_rmse": by_noise["clean"]["rmse"],
                "noise_mean_rmse": float(np.mean([row["rmse"] for row in noise_rows])),
                "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong_rows])),
                "minus20_rmse": by_noise["-20"]["rmse"],
                "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong_rows])),
                "strong_mean_recall": float(np.mean([row["recall"] for row in strong_rows])),
                "strong_mean_rmse_onb": float(np.mean([row["rmse_onb"] for row in strong_rows])),
                "strong_mean_rmse_post": float(np.mean([row["rmse_post"] for row in strong_rows])),
                "minus20_fpr": by_noise["-20"]["fpr"],
                "minus20_recall": by_noise["-20"]["recall"],
                "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
                "minus20_bias_pre": by_noise["-20"]["bias_pre"],
            })
    write_rows("aggregate_two_day.csv", aggregate_rows)

    comparison_rows = []
    for run_name in RUNS:
        for _, _, noise in NOISES:
            rows = [row for row in metric_rows if row["scope"] == "two_day"
                    and row["run"] == run_name and row["noise"] == noise]
            by_model = {row["model"]: row for row in rows}
            best_single = min(by_model[name]["rmse"] for name in ("RF", "Conformer", "AlexNet"))
            comparison_rows.append({
                "run": run_name,
                "noise": noise,
                "performance_rmse": by_model["performance"]["rmse"],
                "equal_rmse": by_model["equal"]["rmse"],
                "best_single_rmse_posthoc": best_single,
                "performance_minus_equal": (
                    by_model["performance"]["rmse"] - by_model["equal"]["rmse"]),
                "performance_minus_best_single_posthoc": (
                    by_model["performance"]["rmse"] - best_single),
                "performance_fpr": by_model["performance"]["fpr"],
                "performance_recall": by_model["performance"]["recall"],
                "performance_rmse_onb": by_model["performance"]["rmse_onb"],
            })
    write_rows("ensemble_comparison.csv", comparison_rows)

    weight_rows = []
    for run_name, run in RUNS.items():
        rows = read_rows(condition_path(run, "heatflux_no_noise")
                         / "ensemble_weights_no_noise.csv")
        for row in rows:
            if row["strategy_name"] == "performance_kfold":
                weight_rows.append({"run": run_name, **row})
    write_rows("weights.csv", weight_rows)

    for _, _, noise in NOISES:
        for model, label in MODELS.items():
            base_rows = loaded[("baseline", noise)]
            selected_rows = loaded[("selected", noise)]
            for day in THRESHOLDS:
                base_by_wav = {}
                selected_by_wav = {}
                for row in base_rows:
                    if source_day(row) == day:
                        base_by_wav.setdefault(row["source_wav_id"], []).append(row)
                for row in selected_rows:
                    if source_day(row) == day:
                        selected_by_wav.setdefault(row["source_wav_id"], []).append(row)
                for wav in sorted(base_by_wav):
                    by, bp = arrays(base_by_wav[wav], model)
                    sy, sp = arrays(selected_by_wav[wav], model)
                    if not np.array_equal(by, sy):
                        raise ValueError(f"Target mismatch: {noise}/{model}/{wav}")
                    wav_delta_rows.append({
                        "experiment": day,
                        "noise": noise,
                        "model": label,
                        "source_wav_id": wav,
                        "heatflux": float(by[0]),
                        "n": len(by),
                        "baseline_rmse": rmse(by, bp),
                        "selected_rmse": rmse(sy, sp),
                        "delta_rmse": rmse(sy, sp) - rmse(by, bp),
                        "delta_sse": float(np.sum((sp - sy) ** 2) - np.sum((bp - by) ** 2)),
                    })
    write_rows("wav_deltas.csv", wav_delta_rows)

    bootstrap_rows = []
    rng = np.random.default_rng(20260929)
    for _, _, noise in NOISES:
        base_rows = loaded[("baseline", noise)]
        selected_rows = loaded[("selected", noise)]
        keys = [identity(row) for row in base_rows]
        if keys != [identity(row) for row in selected_rows]:
            raise ValueError(f"Bootstrap sample mismatch: {noise}")
        cluster_names = sorted({row["source_wav_id"] for row in base_rows})
        indices_by_cluster = {
            wav: np.asarray([index for index, row in enumerate(base_rows)
                             if row["source_wav_id"] == wav], dtype=int)
            for wav in cluster_names
        }
        for model, label in MODELS.items():
            y, baseline_prediction = arrays(base_rows, model)
            selected_y, selected_prediction = arrays(selected_rows, model)
            if not np.array_equal(y, selected_y):
                raise ValueError(f"Bootstrap target mismatch: {noise}/{model}")
            deltas = []
            for _ in range(2000):
                sampled_clusters = rng.choice(cluster_names, len(cluster_names), replace=True)
                indices = np.concatenate([indices_by_cluster[wav] for wav in sampled_clusters])
                deltas.append(
                    rmse(y[indices], selected_prediction[indices])
                    - rmse(y[indices], baseline_prediction[indices])
                )
            bootstrap_rows.append({
                "noise": noise,
                "model": label,
                "n_source_wavs": len(cluster_names),
                "n_bootstrap": len(deltas),
                "point_delta_rmse": rmse(y, selected_prediction) - rmse(y, baseline_prediction),
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "bootstrap_probability_improved": float(np.mean(np.asarray(deltas) < 0)),
            })
    write_rows("cluster_bootstrap_rmse.csv", bootstrap_rows)

    print("analysis complete")
    print(f"completion rows: {len(completion_rows)}")
    print(f"metric rows: {len(metric_rows)}")
    print(f"wav delta rows: {len(wav_delta_rows)}")


if __name__ == "__main__":
    main()
