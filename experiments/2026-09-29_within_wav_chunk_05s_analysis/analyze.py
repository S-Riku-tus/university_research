"""Compare the completed 0.5 s and 1.0 s combined within-WAV runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASE = (
    REPO
    / "Pool_boiling"
    / "Subcooling_20_degrees"
    / "0.3"
    / "2025.06.11_0.3_2_6.18_0.3_3"
    / "regression_result"
    / "npy"
    / "ensemble"
    / "202609"
    / "29"
)
RUNS = {
    "0.5s": BASE / "onb_wc-t0611-v0611_iw3-nc_c0p5s_s0_e150_173043",
    "1.0s": BASE / "onb_wc-t0611-v0611_iw3-nc_s0_e150_161952",
}
DAY_NAMES = {
    "20250611": "2025.06.11_0.3_2",
    "20250618": "2025.06.18_0.3_3",
}
THRESHOLDS = {
    "2025.06.11_0.3_2": 221505.1102,
    "2025.06.18_0.3_3": 271677.6816,
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
    """Use the Windows extended-length form for deeply nested result paths."""
    absolute = path.resolve()
    if str(absolute).startswith("\\\\?\\"):
        return absolute
    return Path("\\\\?\\" + str(absolute))


def read_rows(path: Path) -> list[dict[str, str]]:
    with long_path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def read_json(path: Path):
    return json.loads(long_path(path).read_text(encoding="utf-8"))


def write_rows(name: str, rows) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows produced for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(run: Path, noise_dir: str, suffix: str) -> Path:
    return run / "maxfreq=3kHz" / noise_dir / "fold_pred" / f"pred_f1_{suffix}.csv"


def identify(row: dict[str, str]) -> tuple[str, str, float]:
    prefix, wav = row["source_wav_id"].split("::", 1)
    return DAY_NAMES[prefix], wav, round(float(row["chunk_start_seconds"]), 6)


def day_of(row: dict[str, str]) -> str:
    return identify(row)[0]


def arrays(rows, model: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["y_true"]) for row in rows], dtype=float),
        np.asarray([float(row[model]) for row in rows], dtype=float),
    )


def thresholds_for(rows) -> np.ndarray:
    return np.asarray([THRESHOLDS[day_of(row)] for row in rows], dtype=float)


def safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def rmse(y: np.ndarray, pred: np.ndarray, mask=None) -> float:
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else float("nan")


def safe_auc(actual: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(actual)) < 2:
        return float("nan")
    function = roc_auc_score if kind == "roc" else average_precision_score
    return float(function(actual, score))


def metrics(y: np.ndarray, pred: np.ndarray, thresholds: np.ndarray) -> dict[str, float | int]:
    actual = y >= thresholds
    estimated = pred >= thresholds
    negative = ~actual
    positive = actual
    near = np.abs(y - thresholds) <= 0.10 * thresholds
    tp = int(np.sum(estimated & positive))
    fp = int(np.sum(estimated & negative))
    tn = int(np.sum(~estimated & negative))
    fn = int(np.sum(~estimated & positive))
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else float("nan"),
        "rmse": rmse(y, pred),
        "mae": safe_mean(np.abs(pred - y)),
        "bias": safe_mean(pred - y),
        "rmse_pre": rmse(y, pred, negative),
        "bias_pre": safe_mean((pred - y)[negative]),
        "rmse_post": rmse(y, pred, positive),
        "bias_post": safe_mean((pred - y)[positive]),
        "rmse_onb": rmse(y, pred, near),
        "n_pre": int(np.sum(negative)),
        "n_post": int(np.sum(positive)),
        "n_onb": int(np.sum(near)),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "roc_auc": safe_auc(actual, pred, "roc"),
        "pr_auc": safe_auc(actual, pred, "pr"),
        "fp": fp,
        "fn": fn,
    }


def q100(y: np.ndarray, pred: np.ndarray, threshold: float) -> tuple[float, float]:
    for level in sorted(set(float(value) for value in y)):
        if np.all(pred[y >= level] >= threshold):
            return level, level - threshold
    return float("nan"), float("nan")


def select_scope(rows, scope: str):
    if scope == "both_day_specific":
        return rows
    return [row for row in rows if day_of(row) == scope]


def pair_half_second_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Average selected adjacent 0.5 s halves into one 1 s prediction."""
    grouped: dict[tuple[str, str, float], dict[float, dict[str, str]]] = {}
    for row in rows:
        day, wav, start = identify(row)
        base = float(math.floor(start + 1e-9))
        offset = round(start - base, 6)
        if offset not in (0.0, 0.5):
            raise ValueError(f"Unexpected 0.5 s start: {start}")
        grouped.setdefault((day, wav, base), {})[offset] = row

    paired = []
    for (day, wav, base), halves in grouped.items():
        if set(halves) != {0.0, 0.5}:
            continue
        first, second = halves[0.0], halves[0.5]
        if not np.isclose(float(first["y_true"]), float(second["y_true"])):
            raise ValueError(f"Target differs within pair: {(day, wav, base)}")
        result = {
            "source_wav_id": f"{next(key for key, value in DAY_NAMES.items() if value == day)}::{wav}",
            "chunk_start_seconds": str(base),
            "chunk_duration_seconds": "1.0",
            "y_true": str(float(first["y_true"])),
        }
        for model in MODELS:
            result[model] = str((float(first[model]) + float(second[model])) / 2.0)
        paired.append(result)
    return paired


def main() -> None:
    loaded: dict[tuple[str, str], list[dict[str, str]]] = {}
    completion = []
    identities_by_duration: dict[str, set[tuple[str, str, float]]] = {}

    for duration, run in RUNS.items():
        expected_count = 1080 if duration == "0.5s" else 540
        for noise_dir, suffix, noise in NOISES:
            path = prediction_path(run, noise_dir, suffix)
            if not long_path(path).is_file():
                raise FileNotFoundError(path)
            rows = read_rows(path)
            if len(rows) != expected_count:
                raise ValueError(f"{duration}/{noise}: {len(rows)} rows, expected {expected_count}")
            loaded[(duration, noise)] = rows

            ids = {identify(row) for row in rows}
            if noise == "clean":
                identities_by_duration[duration] = ids
            elif ids != identities_by_duration[duration]:
                raise ValueError(f"Evaluation rows differ across noises: {duration}/{noise}")

            condition_dir = path.parents[1]
            completed = read_json(condition_dir / "completed.json")
            split = read_json(condition_dir / "split_manifest.json")
            fold = split["folds"][0]
            completion.append({
                "duration": duration,
                "noise": noise,
                "completed": 1,
                "run_hash": completed["run_hash"],
                "fit_id": completed["fit_ids"][0],
                "n_fit_ids": len(completed["fit_ids"]),
                "n_training_chunks": fold["n_training_chunks"],
                "n_evaluation_chunks": fold["n_evaluation_chunks"],
                "n_training_wavs": len(fold["training_wav_groups"]),
                "n_evaluation_wavs": len(fold["evaluation_wav_groups"]),
                "inner_error_rf": fold["inner_holdout_errors"]["randomforest"],
                "inner_error_conformer": fold["inner_holdout_errors"]["conformer"],
                "inner_error_alexnet": fold["inner_holdout_errors"]["alexnet"],
            })
    write_rows("completion_audit.csv", completion)

    alignment_rows = []
    clean_half = loaded[("0.5s", "clean")]
    clean_one = loaded[("1.0s", "clean")]
    for day in [*THRESHOLDS, "both_days"]:
        half_rows = clean_half if day == "both_days" else select_scope(clean_half, day)
        one_rows = clean_one if day == "both_days" else select_scope(clean_one, day)
        half_ids = {identify(row) for row in half_rows}
        one_ids = {identify(row) for row in one_rows}
        coverage_counts = []
        for current_day, wav, start in one_ids:
            coverage_counts.append(sum(
                (current_day, wav, round(start + offset, 6)) in half_ids
                for offset in (0.0, 0.5)
            ))
        half_pairs = pair_half_second_rows(half_rows)
        pair_ids = {identify(row) for row in half_pairs}
        common = pair_ids & one_ids
        alignment_rows.append({
            "scope": day,
            "half_second_test_chunks": len(half_ids),
            "one_second_test_intervals": len(one_ids),
            "one_second_intervals_with_both_halves": coverage_counts.count(2),
            "one_second_intervals_with_one_half": coverage_counts.count(1),
            "one_second_intervals_with_no_halves": coverage_counts.count(0),
            "half_chunks_inside_selected_one_second_intervals": sum(coverage_counts),
            "complete_pairs_in_half_second_test": len(pair_ids),
            "complete_pairs_also_in_one_second_test": len(common),
            "common_fraction_of_one_second_test": len(common) / len(one_ids),
        })
    write_rows("time_alignment_audit.csv", alignment_rows)

    scopes = [*THRESHOLDS, "both_day_specific"]
    raw_metrics = []
    raw_q100 = []
    wav_metrics = []
    for duration in RUNS:
        for _, _, noise in NOISES:
            all_rows = loaded[(duration, noise)]
            for scope in scopes:
                rows = select_scope(all_rows, scope)
                for model, label in MODELS.items():
                    y, pred = arrays(rows, model)
                    result = metrics(y, pred, thresholds_for(rows))
                    raw_metrics.append({
                        "duration": duration,
                        "scope": scope,
                        "noise": noise,
                        "model": label,
                        **result,
                    })
                    if scope != "both_day_specific":
                        q, gap = q100(y, pred, THRESHOLDS[scope])
                        raw_q100.append({
                            "duration": duration,
                            "experiment": scope,
                            "noise": noise,
                            "model": label,
                            "threshold": THRESHOLDS[scope],
                            "q100": q,
                            "g100": gap,
                        })

            by_wav: dict[tuple[str, str], list[dict[str, str]]] = {}
            for row in all_rows:
                day, wav, _ = identify(row)
                by_wav.setdefault((day, wav), []).append(row)
            for (day, wav), rows in by_wav.items():
                for model, label in MODELS.items():
                    y, pred = arrays(rows, model)
                    wav_metrics.append({
                        "duration": duration,
                        "experiment": day,
                        "noise": noise,
                        "model": label,
                        "source_wav_id": wav,
                        "heatflux": float(y[0]),
                        "n": len(y),
                        "rmse": rmse(y, pred),
                        "bias": safe_mean(pred - y),
                    })
    write_rows("raw_metrics.csv", raw_metrics)
    write_rows("q100.csv", raw_q100)
    write_rows("wav_metrics.csv", wav_metrics)

    lookup = {
        (row["duration"], row["scope"], row["noise"], row["model"]): row
        for row in raw_metrics
    }
    raw_comparison = []
    for scope in scopes:
        for _, _, noise in NOISES:
            for label in MODELS.values():
                half = lookup[("0.5s", scope, noise, label)]
                one = lookup[("1.0s", scope, noise, label)]
                raw_comparison.append({
                    "scope": scope,
                    "noise": noise,
                    "model": label,
                    "half_rmse": half["rmse"],
                    "one_rmse": one["rmse"],
                    "rmse_half_minus_one": half["rmse"] - one["rmse"],
                    "half_bias": half["bias"],
                    "one_bias": one["bias"],
                    "half_rmse_pre": half["rmse_pre"],
                    "one_rmse_pre": one["rmse_pre"],
                    "half_rmse_onb": half["rmse_onb"],
                    "one_rmse_onb": one["rmse_onb"],
                    "half_fpr": half["fpr"],
                    "one_fpr": one["fpr"],
                    "half_recall": half["recall"],
                    "one_recall": one["recall"],
                })
    write_rows("raw_comparison.csv", raw_comparison)

    aggregate = []
    for duration in RUNS:
        for scope in scopes:
            for label in MODELS.values():
                rows = [lookup[(duration, scope, noise, label)] for _, _, noise in NOISES]
                by_noise = {row["noise"]: row for row in rows}
                noisy = [row for row in rows if row["noise"] != "clean"]
                strong = [row for row in rows if row["noise"] in {"-12", "-16", "-20"}]
                aggregate.append({
                    "duration": duration,
                    "scope": scope,
                    "model": label,
                    "clean_rmse": by_noise["clean"]["rmse"],
                    "noise_mean_rmse": safe_mean(np.asarray([row["rmse"] for row in noisy])),
                    "strong_mean_rmse": safe_mean(np.asarray([row["rmse"] for row in strong])),
                    "minus20_rmse": by_noise["-20"]["rmse"],
                    "noise_mean_bias": safe_mean(np.asarray([row["bias"] for row in noisy])),
                    "strong_mean_fpr": safe_mean(np.asarray([row["fpr"] for row in strong])),
                    "minus20_fpr": by_noise["-20"]["fpr"],
                    "strong_mean_recall": safe_mean(np.asarray([row["recall"] for row in strong])),
                    "minus20_recall": by_noise["-20"]["recall"],
                    "minus20_rmse_pre": by_noise["-20"]["rmse_pre"],
                    "minus20_bias_pre": by_noise["-20"]["bias_pre"],
                    "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
                })
    write_rows("aggregate.csv", aggregate)

    wav_lookup = {
        (row["duration"], row["experiment"], row["noise"], row["model"], row["source_wav_id"]): row
        for row in wav_metrics
    }
    wav_comparison = []
    for key, half in wav_lookup.items():
        duration, day, noise, model, wav = key
        if duration != "0.5s":
            continue
        one = wav_lookup[("1.0s", day, noise, model, wav)]
        wav_comparison.append({
            "experiment": day,
            "noise": noise,
            "model": model,
            "source_wav_id": wav,
            "heatflux": half["heatflux"],
            "half_n": half["n"],
            "one_n": one["n"],
            "half_rmse": half["rmse"],
            "one_rmse": one["rmse"],
            "rmse_half_minus_one": half["rmse"] - one["rmse"],
            "half_better": int(half["rmse"] < one["rmse"]),
            "half_bias": half["bias"],
            "one_bias": one["bias"],
        })
    write_rows("wav_comparison.csv", wav_comparison)

    wav_summary = []
    for day in THRESHOLDS:
        for _, _, noise in NOISES:
            for label in MODELS.values():
                rows = [row for row in wav_comparison if row["experiment"] == day
                        and row["noise"] == noise and row["model"] == label]
                deltas = np.asarray([row["rmse_half_minus_one"] for row in rows], dtype=float)
                wav_summary.append({
                    "experiment": day,
                    "noise": noise,
                    "model": label,
                    "n_wavs": len(rows),
                    "half_better_wavs": int(sum(row["half_better"] for row in rows)),
                    "mean_wav_rmse_delta": safe_mean(deltas),
                    "median_wav_rmse_delta": float(np.median(deltas)),
                })
    write_rows("wav_comparison_summary.csv", wav_summary)

    # Paired cluster bootstrap over the 36 source WAV/heat-flux strata.  This
    # measures evaluation-set uncertainty only; it does not cover training-seed
    # uncertainty because each duration currently has one fitted model.
    rng = np.random.default_rng(42)
    bootstrap_rows = []
    bootstrap_groups = {
        "clean": ["clean"],
        "noise_mean": ["0", "-4", "-8", "-12", "-16", "-20"],
        "strong_mean": ["-12", "-16", "-20"],
        "minus20": ["-20"],
    }
    for scope in scopes:
        reference_rows = select_scope(loaded[("1.0s", "clean")], scope)
        clusters = sorted({identify(row)[:2] for row in reference_rows})
        draws = rng.integers(0, len(clusters), size=(5000, len(clusters)))
        duration_noise_rmse = {duration: {} for duration in RUNS}
        duration_noise_observed = {duration: {} for duration in RUNS}
        for duration in RUNS:
            for _, _, noise in NOISES:
                rows = select_scope(loaded[(duration, noise)], scope)
                by_cluster = {cluster: [] for cluster in clusters}
                for row in rows:
                    by_cluster[identify(row)[:2]].append(row)
                sse = []
                counts = []
                for cluster in clusters:
                    y, pred = arrays(by_cluster[cluster], "ensemble__performance_kfold")
                    sse.append(float(np.sum((pred - y) ** 2)))
                    counts.append(len(y))
                sse_array = np.asarray(sse)
                count_array = np.asarray(counts)
                duration_noise_rmse[duration][noise] = np.sqrt(
                    np.sum(sse_array[draws], axis=1) / np.sum(count_array[draws], axis=1)
                )
                duration_noise_observed[duration][noise] = math.sqrt(
                    float(np.sum(sse_array)) / int(np.sum(count_array))
                )
        for group, noises in bootstrap_groups.items():
            half_distribution = np.mean(
                np.stack([duration_noise_rmse["0.5s"][noise] for noise in noises]), axis=0
            )
            one_distribution = np.mean(
                np.stack([duration_noise_rmse["1.0s"][noise] for noise in noises]), axis=0
            )
            delta = half_distribution - one_distribution
            observed_half = float(np.mean([
                duration_noise_observed["0.5s"][noise] for noise in noises
            ]))
            observed_one = float(np.mean([
                duration_noise_observed["1.0s"][noise] for noise in noises
            ]))
            bootstrap_rows.append({
                "scope": scope,
                "metric_group": group,
                "n_wav_clusters": len(clusters),
                "bootstrap_replicates": len(delta),
                "observed_half_rmse": observed_half,
                "observed_one_rmse": observed_one,
                "observed_delta_half_minus_one": observed_half - observed_one,
                "delta_ci_low": float(np.percentile(delta, 2.5)),
                "delta_ci_high": float(np.percentile(delta, 97.5)),
                "fraction_delta_above_zero": float(np.mean(delta > 0)),
            })
    write_rows("bootstrap_duration_comparison.csv", bootstrap_rows)

    pair_metrics = []
    exact_comparison = []
    for _, _, noise in NOISES:
        half_pairs = pair_half_second_rows(loaded[("0.5s", noise)])
        one_rows = loaded[("1.0s", noise)]
        half_map = {identify(row): row for row in half_pairs}
        one_map = {identify(row): row for row in one_rows}
        common_keys = set(half_map) & set(one_map)
        for scope in scopes:
            def in_scope(key):
                return scope == "both_day_specific" or key[0] == scope

            scoped_all_pairs = [row for key, row in half_map.items() if in_scope(key)]
            scoped_common = sorted(key for key in common_keys if in_scope(key))
            datasets = {
                "half_pair_all": scoped_all_pairs,
                "half_pair_common": [half_map[key] for key in scoped_common],
                "one_common": [one_map[key] for key in scoped_common],
            }
            for model, label in MODELS.items():
                for dataset, rows in datasets.items():
                    y, pred = arrays(rows, model)
                    pair_metrics.append({
                        "scope": scope,
                        "noise": noise,
                        "model": label,
                        "dataset": dataset,
                        **metrics(y, pred, thresholds_for(rows)),
                    })

                half_y, half_pred = arrays(datasets["half_pair_common"], model)
                one_y, one_pred = arrays(datasets["one_common"], model)
                if not np.allclose(half_y, one_y):
                    raise ValueError(f"Target mismatch on common intervals: {scope}/{noise}/{label}")
                exact_comparison.append({
                    "scope": scope,
                    "noise": noise,
                    "model": label,
                    "n": len(half_y),
                    "half_pair_rmse": rmse(half_y, half_pred),
                    "one_rmse": rmse(one_y, one_pred),
                    "rmse_half_minus_one": rmse(half_y, half_pred) - rmse(one_y, one_pred),
                    "half_pair_bias": safe_mean(half_pred - half_y),
                    "one_bias": safe_mean(one_pred - one_y),
                    "prediction_mae_between_durations": safe_mean(np.abs(half_pred - one_pred)),
                    "prediction_correlation": float(np.corrcoef(half_pred, one_pred)[0, 1]),
                })
    write_rows("pair_metrics.csv", pair_metrics)
    write_rows("exact_common_comparison.csv", exact_comparison)

    weight_rows = []
    for duration, run in RUNS.items():
        path = run / "maxfreq=3kHz" / "heatflux_no_noise" / "ensemble_weights_no_noise.csv"
        row = read_rows(path)[0]
        weight_rows.append({
            "duration": duration,
            "rf_weight": float(row["randomforest"]),
            "conformer_weight": float(row["conformer"]),
            "alexnet_weight": float(row["alexnet"]),
        })
    write_rows("weights.csv", weight_rows)

    ensemble_rows = []
    correlation_rows = []
    for duration in RUNS:
        for scope in scopes:
            for _, _, noise in NOISES:
                rows = select_scope(loaded[(duration, noise)], scope)
                y, performance = arrays(rows, "ensemble__performance_kfold")
                _, equal = arrays(rows, "ensemble__simple_equal")
                residuals = {}
                single_rmses = {}
                for model in SINGLE_MODELS:
                    _, pred = arrays(rows, model)
                    residuals[model] = pred - y
                    single_rmses[model] = rmse(y, pred)
                best = min(single_rmses, key=single_rmses.get)
                performance_rmse = rmse(y, performance)
                equal_rmse = rmse(y, equal)
                ensemble_rows.append({
                    "duration": duration,
                    "scope": scope,
                    "noise": noise,
                    "performance_rmse": performance_rmse,
                    "equal_rmse": equal_rmse,
                    "performance_minus_equal": performance_rmse - equal_rmse,
                    "best_single": MODELS[best],
                    "best_single_rmse": single_rmses[best],
                    "performance_minus_best_single": performance_rmse - single_rmses[best],
                    "performance_beats_equal": int(performance_rmse < equal_rmse),
                    "performance_beats_best_single": int(performance_rmse < single_rmses[best]),
                })
                correlation_rows.append({
                    "duration": duration,
                    "scope": scope,
                    "noise": noise,
                    "rf_conformer": float(np.corrcoef(residuals["randomforest"], residuals["conformer"])[0, 1]),
                    "rf_alexnet": float(np.corrcoef(residuals["randomforest"], residuals["alexnet"])[0, 1]),
                    "conformer_alexnet": float(np.corrcoef(residuals["conformer"], residuals["alexnet"])[0, 1]),
                })
    write_rows("ensemble_comparison.csv", ensemble_rows)
    write_rows("residual_correlation.csv", correlation_rows)

    manifest = {
        "runs": {duration: str(run.relative_to(REPO)) for duration, run in RUNS.items()},
        "common_conditions": {
            "evaluation_mode": "within_wav_chunk",
            "experiments": ["2025.06.11_0.3_2", "2025.06.18_0.3_3"],
            "max_frequency_khz": 3,
            "training_noise": "clean_only",
            "test_fraction": 0.25,
            "seed": 42,
            "epochs": 150,
            "inner_wav_group_folds": 3,
            "acoustic_selection": None,
            "explainability": False,
        },
        "difference": {"chunk_seconds": [0.5, 1.0]},
        "completion": {
            "conditions_per_duration": 7,
            "fit_ids": {
                duration: sorted({row["fit_id"] for row in completion if row["duration"] == duration})
                for duration in RUNS
            },
        },
        "comparison_notes": [
            "The raw test sets have the same WAV/heat-flux strata but were randomly sampled independently at each chunk size.",
            "half_pair_common versus one_common is time-aligned, but it is a much smaller selected subset.",
        ],
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
