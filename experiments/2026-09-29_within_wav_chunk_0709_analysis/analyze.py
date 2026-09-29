"""7/9単日within_wav_chunk clean_only runを監査・再集計する。"""

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
    / "2025.07.09_0.3_1" / "regression_result" / "npy" / "ensemble"
    / "202609" / "29" / "onb_wc-t0709-v0709_iw3-nc_s0_e150_165729"
)
THRESHOLD = 571694.252491167
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


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_rows(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows for {name}")
    with (OUTPUT / name).open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(noise_dir, suffix):
    return RUN / "maxfreq=3kHz" / noise_dir / "fold_pred" / f"pred_f1_{suffix}.csv"


def arrays(rows, model):
    return (
        np.asarray([float(row["y_true"]) for row in rows]),
        np.asarray([float(row[model]) for row in rows]),
    )


def rmse(y, prediction, mask=None):
    if mask is not None:
        y, prediction = y[mask], prediction[mask]
    return float(np.sqrt(np.mean((prediction - y) ** 2))) if len(y) else float("nan")


def metrics(y, prediction):
    actual = y >= THRESHOLD
    estimated = prediction >= THRESHOLD
    negative = ~actual
    positive = actual
    near = np.abs(y - THRESHOLD) <= 0.10 * THRESHOLD
    tp = int(np.sum(estimated & positive))
    fp = int(np.sum(estimated & negative))
    tn = int(np.sum(~estimated & negative))
    fn = int(np.sum(~estimated & positive))
    return {
        "n": len(y),
        "r2": float(r2_score(y, prediction)),
        "rmse": rmse(y, prediction),
        "mae": float(np.mean(np.abs(prediction - y))),
        "bias": float(np.mean(prediction - y)),
        "rmse_pre": rmse(y, prediction, negative),
        "bias_pre": float(np.mean((prediction - y)[negative])),
        "rmse_high": rmse(y, prediction, positive),
        "bias_high": float(np.mean((prediction - y)[positive])),
        "rmse_onb": rmse(y, prediction, near),
        "n_pre": int(np.sum(negative)),
        "n_high": int(np.sum(positive)),
        "n_onb": int(np.sum(near)),
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "accuracy": (tp + tn) / len(y),
        "roc_auc": float(roc_auc_score(actual, prediction)),
        "pr_auc": float(average_precision_score(actual, prediction)),
        "fp": fp,
        "fn": fn,
    }


def q100(y, prediction):
    for level in sorted(set(float(value) for value in y)):
        if np.all(prediction[y >= level] >= THRESHOLD):
            return level, level - THRESHOLD
    return float("nan"), float("nan")


def main():
    completion = []
    metric_rows = []
    q100_rows = []
    wav_rows = []
    ensemble_rows = []
    correlation_rows = []

    for noise_dir, suffix, noise in NOISES:
        prediction_file = prediction_path(noise_dir, suffix)
        if not prediction_file.is_file():
            raise FileNotFoundError(prediction_file)
        rows = read_rows(prediction_file)
        if len(rows) != 195:
            raise ValueError(f"prediction count for {noise}: {len(rows)}")
        condition_dir = prediction_file.parents[1]
        completed = json.loads((condition_dir / "completed.json").read_text(encoding="utf-8"))
        split = json.loads((condition_dir / "split_manifest.json").read_text(encoding="utf-8"))
        fold = split["folds"][0]
        completion.append({
            "noise": noise,
            "run_hash": completed["run_hash"],
            "fit_id": completed["fit_ids"][0],
            "n_fit_ids": len(completed["fit_ids"]),
            "n_training_chunks": fold["n_training_chunks"],
            "n_evaluation_chunks": fold["n_evaluation_chunks"],
            "n_training_wavs": len(fold["training_wav_groups"]),
            "n_evaluation_wavs": len(fold["evaluation_wav_groups"]),
        })
        y = np.asarray([float(row["y_true"]) for row in rows])
        residuals = {}
        single_rmse = {}
        for model, label in MODELS.items():
            _, prediction = arrays(rows, model)
            result = metrics(y, prediction)
            metric_rows.append({"noise": noise, "model": label, **result})
            q, gap = q100(y, prediction)
            q100_rows.append({
                "noise": noise,
                "model": label,
                "threshold": THRESHOLD,
                "q100": q,
                "g100": gap,
            })
            by_wav = {}
            for row in rows:
                by_wav.setdefault(row["source_wav_id"], []).append(row)
            for wav, wav_data in by_wav.items():
                wy, wp = arrays(wav_data, model)
                wav_rows.append({
                    "noise": noise,
                    "model": label,
                    "source_wav_id": wav,
                    "heatflux": float(wy[0]),
                    "n": len(wy),
                    "rmse": rmse(wy, wp),
                    "bias": float(np.mean(wp - wy)),
                })
            if model in SINGLE_MODELS:
                residuals[model] = prediction - y
                single_rmse[model] = result["rmse"]
        _, performance = arrays(rows, "ensemble__performance_kfold")
        _, equal = arrays(rows, "ensemble__simple_equal")
        best = min(single_rmse, key=single_rmse.get)
        ensemble_rows.append({
            "noise": noise,
            "performance_rmse": rmse(y, performance),
            "equal_rmse": rmse(y, equal),
            "performance_minus_equal": rmse(y, performance) - rmse(y, equal),
            "best_single": MODELS[best],
            "best_single_rmse": single_rmse[best],
            "performance_minus_best_single": rmse(y, performance) - single_rmse[best],
            "performance_beats_equal": int(rmse(y, performance) < rmse(y, equal)),
            "performance_beats_best_single": int(rmse(y, performance) < single_rmse[best]),
        })
        correlation_rows.append({
            "noise": noise,
            "rf_conformer": float(np.corrcoef(residuals["randomforest"], residuals["conformer"])[0, 1]),
            "rf_alexnet": float(np.corrcoef(residuals["randomforest"], residuals["alexnet"])[0, 1]),
            "conformer_alexnet": float(np.corrcoef(residuals["conformer"], residuals["alexnet"])[0, 1]),
        })

    write_rows("completion_audit.csv", completion)
    write_rows("metrics.csv", metric_rows)
    write_rows("q100.csv", q100_rows)
    write_rows("wav_metrics.csv", wav_rows)
    write_rows("ensemble_comparison.csv", ensemble_rows)
    write_rows("residual_correlation.csv", correlation_rows)

    aggregate = []
    for model in MODELS.values():
        rows = [row for row in metric_rows if row["model"] == model]
        by_noise = {row["noise"]: row for row in rows}
        noise_rows = [row for row in rows if row["noise"] != "clean"]
        strong = [row for row in rows if row["noise"] in {"-12", "-16", "-20"}]
        aggregate.append({
            "model": model,
            "clean_rmse": by_noise["clean"]["rmse"],
            "noise_mean_rmse": float(np.mean([row["rmse"] for row in noise_rows])),
            "strong_mean_rmse": float(np.mean([row["rmse"] for row in strong])),
            "minus20_rmse": by_noise["-20"]["rmse"],
            "clean_r2": by_noise["clean"]["r2"],
            "minus20_r2": by_noise["-20"]["r2"],
            "strong_mean_fpr": float(np.mean([row["fpr"] for row in strong])),
            "minus20_fpr": by_noise["-20"]["fpr"],
            "strong_mean_recall": float(np.mean([row["recall"] for row in strong])),
            "minus20_recall": by_noise["-20"]["recall"],
            "minus20_rmse_onb": by_noise["-20"]["rmse_onb"],
            "minus20_rmse_pre": by_noise["-20"]["rmse_pre"],
            "minus20_bias_pre": by_noise["-20"]["bias_pre"],
        })
    write_rows("aggregate.csv", aggregate)

    weights = read_rows(
        RUN / "maxfreq=3kHz" / "heatflux_no_noise" / "ensemble_weights_no_noise.csv"
    )[0]
    write_rows("weights.csv", [{
        "rf_weight": float(weights["randomforest"]),
        "conformer_weight": float(weights["conformer"]),
        "alexnet_weight": float(weights["alexnet"]),
    }])

    manifest = {
        "run": str(RUN.relative_to(REPO)),
        "threshold": THRESHOLD,
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
            "source_wavs": sorted({row["n_training_wavs"] for row in completion}),
        },
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
