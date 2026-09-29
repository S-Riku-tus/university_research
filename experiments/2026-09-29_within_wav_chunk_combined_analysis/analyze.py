"""6/11＋6/18統合within_wav_chunk runを単日runと対応比較する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASE = REPO / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"

COMBINED_NAME = "2025.06.11_0.3_2_6.18_0.3_3"
COMBINED_RUN = (
    BASE / COMBINED_NAME / "regression_result" / "npy" / "ensemble"
    / "202609" / "29" / "onb_wc-t0611-v0611_iw3-nc_s0_e150_161952"
)
SINGLE_RUNS = {
    "2025.06.11_0.3_2": (
        BASE / "2025.06.11_0.3_2" / "regression_result" / "npy" / "ensemble"
        / "202609" / "29" / "onb_wc-t0611-v0611_iw3-nc_s0_e150_152011"
    ),
    "2025.06.18_0.3_3": (
        BASE / "2025.06.18_0.3_3" / "regression_result" / "npy" / "ensemble"
        / "202609" / "29" / "onb_wc-t0618-v0618_iw3-nc_s0_e150_152911"
    ),
}
THRESHOLDS = {
    "2025.06.11_0.3_2": 221505.1102,
    "2025.06.18_0.3_3": 271677.6816,
}
AVERAGE_THRESHOLD = 246591.3959
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


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_rows(name: str, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(run: Path, noise_dir: str, suffix: str):
    return run / "maxfreq=3kHz" / noise_dir / "fold_pred" / f"pred_f1_{suffix}.csv"


def arrays(rows, model):
    return (
        np.asarray([float(row["y_true"]) for row in rows]),
        np.asarray([float(row[model]) for row in rows]),
    )


def rmse(y, pred, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else float("nan")


def metrics(y, pred, threshold):
    actual = y >= threshold
    estimated = pred >= threshold
    negative = ~actual
    positive = actual
    tp = int(np.sum(estimated & positive))
    fp = int(np.sum(estimated & negative))
    tn = int(np.sum(~estimated & negative))
    fn = int(np.sum(~estimated & positive))
    near = np.abs(y - threshold) <= 0.10 * threshold
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)),
        "rmse": rmse(y, pred),
        "mae": float(np.mean(np.abs(pred - y))),
        "bias": float(np.mean(pred - y)),
        "rmse_pre": rmse(y, pred, negative),
        "bias_pre": float(np.mean((pred - y)[negative])),
        "rmse_high": rmse(y, pred, positive),
        "bias_high": float(np.mean((pred - y)[positive])),
        "rmse_onb": rmse(y, pred, near),
        "n_pre": int(np.sum(negative)),
        "n_high": int(np.sum(positive)),
        "n_onb": int(np.sum(near)),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "accuracy": (tp + tn) / len(y),
        "roc_auc": float(roc_auc_score(actual, pred)),
        "pr_auc": float(average_precision_score(actual, pred)),
        "fp": fp,
        "fn": fn,
    }


def metrics_with_day_thresholds(y, pred, thresholds):
    """日ごとに異なるONBを使い、2日をまとめた指標を返す。"""
    actual = y >= thresholds
    estimated = pred >= thresholds
    negative = ~actual
    positive = actual
    tp = int(np.sum(estimated & positive))
    fp = int(np.sum(estimated & negative))
    tn = int(np.sum(~estimated & negative))
    fn = int(np.sum(~estimated & positive))
    near = np.abs(y - thresholds) <= 0.10 * thresholds
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)),
        "rmse": rmse(y, pred),
        "mae": float(np.mean(np.abs(pred - y))),
        "bias": float(np.mean(pred - y)),
        "rmse_pre": rmse(y, pred, negative),
        "rmse_high": rmse(y, pred, positive),
        "rmse_onb": rmse(y, pred, near),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "fp": fp,
        "fn": fn,
    }


def combined_identity(row):
    prefix, original_wav = row["source_wav_id"].split("::", 1)
    return SOURCE_PREFIX[prefix], original_wav, int(float(row["chunk_index"]))


def single_identity(day, row):
    return day, row["source_wav_id"], int(float(row["chunk_index"]))


def q100(y, pred, threshold):
    for level in sorted(set(float(value) for value in y)):
        if np.all(pred[y >= level] >= threshold):
            return level, level - threshold
    return float("nan"), float("nan")


def main():
    loaded = {}
    completion = []
    for noise_dir, suffix, noise in NOISES:
        combined_path = prediction_path(COMBINED_RUN, noise_dir, suffix)
        if not combined_path.is_file():
            raise FileNotFoundError(combined_path)
        combined_rows = read_rows(combined_path)
        if len(combined_rows) != 540:
            raise ValueError(f"combined prediction count: {noise}={len(combined_rows)}")
        completed_path = combined_path.parents[1] / "completed.json"
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        split = json.loads((combined_path.parents[1] / "split_manifest.json").read_text(encoding="utf-8"))
        completion.append({
            "noise": noise,
            "completed": int(completed_path.is_file()),
            "run_hash": completed["run_hash"],
            "fit_id": completed["fit_ids"][0],
            "n_fit_ids": len(completed["fit_ids"]),
            "n_training_chunks": split["folds"][0]["n_training_chunks"],
            "n_evaluation_chunks": split["folds"][0]["n_evaluation_chunks"],
            "n_training_wavs": len(split["folds"][0]["training_wav_groups"]),
            "n_evaluation_wavs": len(split["folds"][0]["evaluation_wav_groups"]),
        })
        for day in SINGLE_RUNS:
            selected = [row for row in combined_rows if combined_identity(row)[0] == day]
            if len(selected) != 270:
                raise ValueError(f"combined day count: {day}, {noise}={len(selected)}")
            loaded[(day, "pooled", noise)] = selected
            single_rows = read_rows(prediction_path(SINGLE_RUNS[day], noise_dir, suffix))
            loaded[(day, "single", noise)] = single_rows
    write_rows("completion_audit.csv", completion)

    split_rows = []
    for day in SINGLE_RUNS:
        for _, _, noise in NOISES:
            pooled = loaded[(day, "pooled", noise)]
            single = loaded[(day, "single", noise)]
            pooled_keys = {combined_identity(row) for row in pooled}
            single_keys = {single_identity(day, row) for row in single}
            pooled_targets = {combined_identity(row): float(row["y_true"]) for row in pooled}
            single_targets = {single_identity(day, row): float(row["y_true"]) for row in single}
            common = pooled_keys & single_keys
            target_max_difference = max(
                abs(pooled_targets[key] - single_targets[key]) for key in common
            )
            split_rows.append({
                "experiment": day,
                "noise": noise,
                "pooled_chunks": len(pooled_keys),
                "single_chunks": len(single_keys),
                "common_chunks": len(common),
                "sets_equal": int(pooled_keys == single_keys),
                "target_max_abs_difference": target_max_difference,
            })
    write_rows("split_audit.csv", split_rows)

    metric_rows = []
    q100_rows = []
    wav_rows = []
    for day, threshold in THRESHOLDS.items():
        for scope in ("single", "pooled"):
            for _, _, noise in NOISES:
                rows = loaded[(day, scope, noise)]
                key_function = (
                    combined_identity if scope == "pooled"
                    else lambda row, current_day=day: single_identity(current_day, row)
                )
                for model, label in MODELS.items():
                    y, pred = arrays(rows, model)
                    result = metrics(y, pred, threshold)
                    metric_rows.append({
                        "experiment": day,
                        "training_scope": scope,
                        "noise": noise,
                        "model": label,
                        "threshold": threshold,
                        **result,
                    })
                    q, gap = q100(y, pred, threshold)
                    q100_rows.append({
                        "experiment": day,
                        "training_scope": scope,
                        "noise": noise,
                        "model": label,
                        "threshold": threshold,
                        "q100": q,
                        "g100": gap,
                    })
                    by_wav = {}
                    for row in rows:
                        identity = key_function(row)
                        by_wav.setdefault(identity[1], []).append(row)
                    for wav, wav_data in by_wav.items():
                        wy, wp = arrays(wav_data, model)
                        wav_rows.append({
                            "experiment": day,
                            "training_scope": scope,
                            "noise": noise,
                            "model": label,
                            "source_wav_id": wav,
                            "heatflux": float(wy[0]),
                            "n": len(wy),
                            "rmse": rmse(wy, wp),
                            "bias": float(np.mean(wp - wy)),
                        })
    write_rows("metrics_by_day.csv", metric_rows)
    write_rows("q100.csv", q100_rows)
    write_rows("wav_metrics.csv", wav_rows)

    aggregate = []
    for day in SINGLE_RUNS:
        for scope in ("single", "pooled"):
            for model in MODELS.values():
                rows = [row for row in metric_rows if row["experiment"] == day
                        and row["training_scope"] == scope and row["model"] == model]
                by_noise = {row["noise"]: row for row in rows}
                noise_rows = [row for row in rows if row["noise"] != "clean"]
                strong = [row for row in rows if row["noise"] in {"-12", "-16", "-20"}]
                aggregate.append({
                    "experiment": day,
                    "training_scope": scope,
                    "model": model,
                    "clean_rmse": by_noise["clean"]["rmse"],
                    "noise_mean_rmse": float(np.mean([row["rmse"] for row in noise_rows])),
                    "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong])),
                    "minus20_rmse": by_noise["-20"]["rmse"],
                    "noise_mean_bias": float(np.mean([row["bias"] for row in noise_rows])),
                    "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong])),
                    "minus20_fpr": by_noise["-20"]["fpr"],
                    "strong_mean_recall": float(np.mean([row["recall"] for row in strong])),
                    "minus20_recall": by_noise["-20"]["recall"],
                    "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
                    "minus20_rmse_pre": by_noise["-20"]["rmse_pre"],
                    "minus20_bias_pre": by_noise["-20"]["bias_pre"],
                })
    write_rows("aggregate_by_day.csv", aggregate)

    comparison_rows = []
    for day in SINGLE_RUNS:
        for _, _, noise in NOISES:
            for model in MODELS.values():
                pooled = next(row for row in metric_rows if row["experiment"] == day
                              and row["training_scope"] == "pooled"
                              and row["noise"] == noise and row["model"] == model)
                single = next(row for row in metric_rows if row["experiment"] == day
                              and row["training_scope"] == "single"
                              and row["noise"] == noise and row["model"] == model)
                comparison_rows.append({
                    "experiment": day,
                    "noise": noise,
                    "model": model,
                    "pooled_rmse": pooled["rmse"],
                    "single_rmse": single["rmse"],
                    "rmse_change_pooled_minus_single": pooled["rmse"] - single["rmse"],
                    "pooled_bias": pooled["bias"],
                    "single_bias": single["bias"],
                    "bias_change_pooled_minus_single": pooled["bias"] - single["bias"],
                    "pooled_fpr": pooled["fpr"],
                    "single_fpr": single["fpr"],
                    "fpr_change_pooled_minus_single": pooled["fpr"] - single["fpr"],
                    "pooled_recall": pooled["recall"],
                    "single_recall": single["recall"],
                    "recall_change_pooled_minus_single": pooled["recall"] - single["recall"],
                    "pooled_rmse_onb": pooled["rmse_onb"],
                    "single_rmse_onb": single["rmse_onb"],
                    "pooled_rmse_pre": pooled["rmse_pre"],
                    "single_rmse_pre": single["rmse_pre"],
                })
    write_rows("pooled_minus_single.csv", comparison_rows)

    two_day_rows = []
    for scope in ("single", "pooled"):
        for _, _, noise in NOISES:
            for model, label in MODELS.items():
                y_parts = []
                prediction_parts = []
                threshold_parts = []
                for day, threshold in THRESHOLDS.items():
                    y, prediction = arrays(loaded[(day, scope, noise)], model)
                    y_parts.append(y)
                    prediction_parts.append(prediction)
                    threshold_parts.append(np.full(len(y), threshold, dtype=float))
                result = metrics_with_day_thresholds(
                    np.concatenate(y_parts),
                    np.concatenate(prediction_parts),
                    np.concatenate(threshold_parts),
                )
                two_day_rows.append({
                    "training_scope": scope,
                    "noise": noise,
                    "model": label,
                    **result,
                })
    write_rows("two_day_day_specific_onb_metrics.csv", two_day_rows)

    overall_rows = []
    for _, _, noise in NOISES:
        rows = read_rows(prediction_path(
            COMBINED_RUN,
            next(item[0] for item in NOISES if item[2] == noise),
            next(item[1] for item in NOISES if item[2] == noise),
        ))
        for model, label in MODELS.items():
            y, pred = arrays(rows, model)
            overall_rows.append({
                "noise": noise,
                "model": label,
                "threshold": AVERAGE_THRESHOLD,
                **metrics(y, pred, AVERAGE_THRESHOLD),
            })
    write_rows("overall_average_onb_metrics.csv", overall_rows)

    weight_rows = []
    for scope, day, run in [
        ("pooled", COMBINED_NAME, COMBINED_RUN),
        *[("single", day, run) for day, run in SINGLE_RUNS.items()],
    ]:
        path = run / "maxfreq=3kHz" / "heatflux_no_noise" / "ensemble_weights_no_noise.csv"
        row = read_rows(path)[0]
        weight_rows.append({
            "training_scope": scope,
            "experiment": day,
            "rf_weight": float(row["randomforest"]),
            "conformer_weight": float(row["conformer"]),
            "alexnet_weight": float(row["alexnet"]),
        })
    write_rows("weights.csv", weight_rows)

    ensemble_rows = []
    correlation_rows = []
    for day in SINGLE_RUNS:
        for scope in ("single", "pooled"):
            for _, _, noise in NOISES:
                rows = loaded[(day, scope, noise)]
                y, performance = arrays(rows, "ensemble__performance_kfold")
                _, equal = arrays(rows, "ensemble__simple_equal")
                single_rmse = {}
                residuals = {}
                for model in SINGLE_MODELS:
                    _, pred = arrays(rows, model)
                    single_rmse[model] = rmse(y, pred)
                    residuals[model] = pred - y
                best_key = min(single_rmse, key=single_rmse.get)
                ensemble_rows.append({
                    "experiment": day,
                    "training_scope": scope,
                    "noise": noise,
                    "performance_rmse": rmse(y, performance),
                    "equal_rmse": rmse(y, equal),
                    "performance_minus_equal": rmse(y, performance) - rmse(y, equal),
                    "best_single": MODELS[best_key],
                    "best_single_rmse": single_rmse[best_key],
                    "performance_minus_best_single": rmse(y, performance) - single_rmse[best_key],
                    "performance_beats_equal": int(rmse(y, performance) < rmse(y, equal)),
                    "performance_beats_best_single": int(rmse(y, performance) < single_rmse[best_key]),
                })
                correlation_rows.append({
                    "experiment": day,
                    "training_scope": scope,
                    "noise": noise,
                    "rf_conformer": float(np.corrcoef(residuals["randomforest"], residuals["conformer"])[0, 1]),
                    "rf_alexnet": float(np.corrcoef(residuals["randomforest"], residuals["alexnet"])[0, 1]),
                    "conformer_alexnet": float(np.corrcoef(residuals["conformer"], residuals["alexnet"])[0, 1]),
                })
    write_rows("ensemble_comparison.csv", ensemble_rows)
    write_rows("residual_correlation.csv", correlation_rows)

    wav_lookup = {
        (row["experiment"], row["training_scope"], row["noise"], row["model"], row["source_wav_id"]): row
        for row in wav_rows
    }
    wav_change_rows = []
    for key, pooled in wav_lookup.items():
        day, scope, noise, model, wav = key
        if scope != "pooled":
            continue
        single = wav_lookup[(day, "single", noise, model, wav)]
        wav_change_rows.append({
            "experiment": day,
            "noise": noise,
            "model": model,
            "source_wav_id": wav,
            "heatflux": pooled["heatflux"],
            "pooled_rmse": pooled["rmse"],
            "single_rmse": single["rmse"],
            "rmse_change_pooled_minus_single": pooled["rmse"] - single["rmse"],
            "pooled_bias": pooled["bias"],
            "single_bias": single["bias"],
        })
    write_rows("wav_changes.csv", wav_change_rows)

    summary = {
        "combined_run": str(COMBINED_RUN.relative_to(REPO)),
        "single_runs": {day: str(run.relative_to(REPO)) for day, run in SINGLE_RUNS.items()},
        "conditions": {
            "chunk_seconds": 1,
            "max_frequency_khz": 3,
            "training_noise": "clean_only",
            "training_seed": 42,
            "outer_split_seed": 42,
            "epochs": 150,
            "inner_wav_group_folds": 3,
            "acoustic_selection": None,
            "explainability": False,
        },
        "completion": {
            "conditions": len(completion),
            "unique_fit_ids": sorted({row["fit_id"] for row in completion}),
            "training_chunks": sorted({row["n_training_chunks"] for row in completion}),
            "evaluation_chunks": sorted({row["n_evaluation_chunks"] for row in completion}),
        },
        "paired_test_chunks": {
            day: sorted({row["common_chunks"] for row in split_rows if row["experiment"] == day})
            for day in SINGLE_RUNS
        },
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
