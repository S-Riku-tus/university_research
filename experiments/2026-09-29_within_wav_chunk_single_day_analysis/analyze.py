"""6/11・6/18のwithin_wav_chunkを監査し、従来WAV holdoutと比較する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASE = REPO / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
sys.path.insert(0, str(REPO / "code"))

from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.experiment.learning_policy import checked_metadata, outer_splits

RUNS = {
    "2025.06.11_0.3_2": {
        "threshold": 221505.1102,
        "within_wav_chunk": BASE / "2025.06.11_0.3_2" / "regression_result" / "npy" / "ensemble" / "202609" / "29" / "onb_wc-t0611-v0611_iw3-nc_s0_e150_152011",
        "within_day_wav_holdout": BASE / "2025.06.11_0.3_2" / "regression_result" / "npy" / "ensemble" / "202609" / "28" / "onb_wh-t0611-v0611_iw3-nc_s0_e150_221833",
    },
    "2025.06.18_0.3_3": {
        "threshold": 271677.6816,
        "within_wav_chunk": BASE / "2025.06.18_0.3_3" / "regression_result" / "npy" / "ensemble" / "202609" / "29" / "onb_wc-t0618-v0618_iw3-nc_s0_e150_152911",
        "within_day_wav_holdout": BASE / "2025.06.18_0.3_3" / "regression_result" / "npy" / "ensemble" / "202609" / "29" / "onb_wh-t0618-v0618_iw3-nc_s0_e150_094511",
    },
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
SINGLE_MODELS = ["randomforest", "conformer", "alexnet"]


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


def sample_key(row):
    return row["source_wav_id"], int(float(row["chunk_index"]))


def arrays(rows, model):
    return (
        np.asarray([float(row["y_true"]) for row in rows]),
        np.asarray([float(row[model]) for row in rows]),
    )


def safe_rmse(y, pred, mask=None):
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return float(np.sqrt(np.mean((pred - y) ** 2))) if len(y) else float("nan")


def metrics(y, pred, threshold):
    actual = y >= threshold
    estimated = pred >= threshold
    negatives = ~actual
    positives = actual
    tp = int(np.sum(estimated & positives))
    fp = int(np.sum(estimated & negatives))
    tn = int(np.sum(~estimated & negatives))
    fn = int(np.sum(~estimated & positives))
    onb = np.abs(y - threshold) <= 0.10 * threshold
    high = y >= threshold
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)),
        "rmse": safe_rmse(y, pred),
        "mae": float(np.mean(np.abs(pred - y))),
        "bias": float(np.mean(pred - y)),
        "rmse_high": safe_rmse(y, pred, high),
        "rmse_onb": safe_rmse(y, pred, onb),
        "n_high": int(np.sum(high)),
        "n_onb": int(np.sum(onb)),
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


def q100(y, pred, threshold):
    levels = sorted(set(float(value) for value in y))
    answer = float("nan")
    for level in levels:
        if np.all(pred[y >= level] >= threshold):
            answer = level
            break
    if not np.isfinite(answer):
        status = "not_reached"
    elif answer < threshold:
        status = "premature"
    elif answer == threshold:
        status = "at_threshold"
    else:
        status = "delayed"
    return answer, answer - threshold if np.isfinite(answer) else float("nan"), status


def ranks(values, higher=False):
    ordered = sorted(values, key=values.get, reverse=higher)
    return {key: rank + 1 for rank, key in enumerate(ordered)}


def spearman_three(left, right):
    keys = SINGLE_MODELS
    x = np.asarray([left[key] for key in keys], dtype=float)
    y = np.asarray([right[key] for key in keys], dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    all_metrics = []
    q100_rows = []
    wav_rows = []
    loaded = {}

    for day, config in RUNS.items():
        threshold = config["threshold"]
        for scheme in ("within_wav_chunk", "within_day_wav_holdout"):
            run = config[scheme]
            for noise_dir, suffix, noise in NOISES:
                path = prediction_path(run, noise_dir, suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows = read_rows(path)
                loaded[(day, scheme, noise)] = rows
                for model, label in MODELS.items():
                    y, pred = arrays(rows, model)
                    result = metrics(y, pred, threshold)
                    all_metrics.append({
                        "experiment": day,
                        "scheme": scheme,
                        "noise": noise,
                        "model": label,
                        **result,
                    })
                    if scheme == "within_wav_chunk":
                        q, gap, status = q100(y, pred, threshold)
                        q100_rows.append({
                            "experiment": day,
                            "noise": noise,
                            "model": label,
                            "threshold": threshold,
                            "q100": q,
                            "g100": gap,
                            "status": status,
                            "chunks_per_heatflux": 15,
                        })
                        by_wav = {}
                        for row in rows:
                            by_wav.setdefault(row["source_wav_id"], []).append(row)
                        for wav, wav_data in by_wav.items():
                            wy, wp = arrays(wav_data, model)
                            wav_rows.append({
                                "experiment": day,
                                "noise": noise,
                                "model": label,
                                "source_wav_id": wav,
                                "heatflux": float(wy[0]),
                                "n": len(wy),
                                "rmse": safe_rmse(wy, wp),
                                "bias": float(np.mean(wp - wy)),
                            })

    write_rows("metrics_all.csv", all_metrics)
    write_rows("q100.csv", q100_rows)
    write_rows("wav_metrics.csv", wav_rows)

    # 全評価集合のnoise平均。方式間で評価WAV集合が異なる点はREADMEで明示する。
    aggregate_rows = []
    for day in RUNS:
        for scheme in ("within_wav_chunk", "within_day_wav_holdout"):
            for model in MODELS.values():
                rows = [row for row in all_metrics if row["experiment"] == day
                        and row["scheme"] == scheme and row["model"] == model]
                clean = next(row for row in rows if row["noise"] == "clean")
                noise = [row for row in rows if row["noise"] != "clean"]
                strong = [row for row in rows if row["noise"] in {"-12", "-16", "-20"}]
                aggregate_rows.append({
                    "experiment": day,
                    "scheme": scheme,
                    "model": model,
                    "clean_rmse": clean["rmse"],
                    "noise_mean_rmse": float(np.mean([row["rmse"] for row in noise])),
                    "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong])),
                    "minus20_rmse": next(row["rmse"] for row in rows if row["noise"] == "-20"),
                    "noise_mean_fpr": float(np.mean([row["fpr"] for row in noise])),
                    "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong])),
                    "minus20_fpr": next(row["fpr"] for row in rows if row["noise"] == "-20"),
                    "noise_mean_recall": float(np.mean([row["recall"] for row in noise])),
                    "noise_mean_bias": float(np.mean([row["bias"] for row in noise])),
                })
    write_rows("aggregate_by_scheme.csv", aggregate_rows)

    # 新方式の75共通chunk（従来holdoutの5 WAV×15 chunk）だけへ評価集合を揃える。
    common_rows = []
    for day, config in RUNS.items():
        threshold = config["threshold"]
        for _, _, noise in NOISES:
            current = loaded[(day, "within_wav_chunk", noise)]
            previous = loaded[(day, "within_day_wav_holdout", noise)]
            previous_map = {sample_key(row): row for row in previous}
            common_current = [row for row in current if sample_key(row) in previous_map]
            common_previous = [previous_map[sample_key(row)] for row in common_current]
            if len(common_current) != 75:
                raise ValueError(f"common sample count is not 75: {day}, {noise}: {len(common_current)}")
            for model, label in MODELS.items():
                y_current, pred_current = arrays(common_current, model)
                y_previous, pred_previous = arrays(common_previous, model)
                if not np.allclose(y_current, y_previous, rtol=0, atol=1e-6):
                    raise ValueError(f"target mismatch: {day}, {noise}, {model}")
                current_metrics = metrics(y_current, pred_current, threshold)
                previous_metrics = metrics(y_previous, pred_previous, threshold)
                common_rows.append({
                    "experiment": day,
                    "noise": noise,
                    "model": label,
                    "n_common": len(y_current),
                    "chunk_rmse": current_metrics["rmse"],
                    "wav_holdout_rmse": previous_metrics["rmse"],
                    "rmse_change_chunk_minus_wav": current_metrics["rmse"] - previous_metrics["rmse"],
                    "chunk_bias": current_metrics["bias"],
                    "wav_holdout_bias": previous_metrics["bias"],
                    "chunk_fpr": current_metrics["fpr"],
                    "wav_holdout_fpr": previous_metrics["fpr"],
                    "chunk_recall": current_metrics["recall"],
                    "wav_holdout_recall": previous_metrics["recall"],
                })
    write_rows("common_chunk_comparison.csv", common_rows)

    # 内部重み順位と外側単体順位。clean_onlyなので内部重みはnoise間で共通。
    transfer_rows = []
    for day, config in RUNS.items():
        run = config["within_wav_chunk"]
        clean_dir = run / "maxfreq=3kHz" / "heatflux_no_noise"
        weights = read_rows(clean_dir / "ensemble_weights_no_noise.csv")[0]
        weight_values = {key: float(weights[key]) for key in SINGLE_MODELS}
        internal_ranks = ranks(weight_values, higher=True)
        internal = json.loads((clean_dir / "internal_validation_fold1.json").read_text(encoding="utf-8"))
        if internal.get("test_used") is not False:
            raise ValueError(f"internal validation used test data: {day}")
        for _, _, noise in NOISES:
            rows = loaded[(day, "within_wav_chunk", noise)]
            outer_rmse = {}
            for model in SINGLE_MODELS:
                y, pred = arrays(rows, model)
                outer_rmse[model] = safe_rmse(y, pred)
            outer_ranks = ranks(outer_rmse, higher=False)
            transfer_rows.append({
                "experiment": day,
                "noise": noise,
                "max_weight_model": MODELS[max(weight_values, key=weight_values.get)],
                "outer_best_single": MODELS[min(outer_rmse, key=outer_rmse.get)],
                "max_weight_matches_outer_best": int(
                    max(weight_values, key=weight_values.get) == min(outer_rmse, key=outer_rmse.get)
                ),
                "rank_spearman": spearman_three(internal_ranks, outer_ranks),
                "rf_weight": weight_values["randomforest"],
                "conformer_weight": weight_values["conformer"],
                "alexnet_weight": weight_values["alexnet"],
                "rf_rmse": outer_rmse["randomforest"],
                "conformer_rmse": outer_rmse["conformer"],
                "alexnet_rmse": outer_rmse["alexnet"],
            })
    write_rows("weight_transfer.csv", transfer_rows)

    # 現実装のまま2日poolすると、単日runのテストchunkを再利用できるかを事前監査する。
    combined_name = "2025.06.11_0.3_2_6.18_0.3_3"
    def policy(experiment):
        return {
            "evaluation_mode": "within_wav_chunk",
            "within_day_experiment": experiment,
            "test_fraction": 0.25,
            "test_split_seed": 42,
            "test_stratify": "none",
            "train_experiments": [experiment],
            "test_experiments": [experiment],
            "training_noise": "clean_only",
        }

    single_test_keys = {}
    for day in RUNS:
        path = BASE / day / "data" / "npy" / "waterflow_20260817_1s" / "maxfreq=3kHz" / "heatflux_no_noise"
        metadata = checked_metadata(load_sample_metadata_without_arrays(path), day)
        _, test = outer_splits(metadata, metadata, 3, policy(day), RUNS[day]["threshold"])[0]
        single_test_keys[day] = {
            (metadata[index]["source_wav_id"], int(metadata[index]["chunk_index"]))
            for index in test
        }
    path = BASE / combined_name / "data" / "npy" / "waterflow_20260817_1s" / "maxfreq=3kHz" / "heatflux_no_noise"
    combined_metadata = checked_metadata(load_sample_metadata_without_arrays(path), combined_name)
    _, combined_test = outer_splits(
        combined_metadata, combined_metadata, 3, policy(combined_name), 246591.3959
    )[0]
    pooled_keys = {
        day: {
            (combined_metadata[index]["original_source_wav_id"],
             int(combined_metadata[index]["chunk_index"]))
            for index in combined_test
            if combined_metadata[index]["source_experiment_name"] == day
        }
        for day in RUNS
    }
    pooled_split_rows = []
    for day in RUNS:
        same = len(single_test_keys[day] & pooled_keys[day])
        pooled_split_rows.append({
            "experiment": day,
            "single_day_test_chunks": len(single_test_keys[day]),
            "current_pooled_test_chunks": len(pooled_keys[day]),
            "same_test_chunks": same,
            "same_fraction": same / len(single_test_keys[day]),
        })
    write_rows("pooled_split_preview.csv", pooled_split_rows)

    summary = {
        "runs": {
            day: {key: str(value.relative_to(REPO)) if isinstance(value, Path) else value
                  for key, value in config.items()}
            for day, config in RUNS.items()
        },
        "current_scheme": {
            "completed_conditions_each_day": 7,
            "fit_chunks_each_day": 810,
            "test_chunks_each_day": 270,
            "source_wavs_each_side": 18,
            "fit_chunks_per_wav": 45,
            "test_chunks_per_wav": 15,
            "exact_chunk_overlap": 0,
        },
        "common_comparison_chunks_each_day_noise": 75,
        "pooled_split_preview": pooled_split_rows,
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
