"""Evaluate the full SNR curve of the existing 20260722 datasets.

This reuses a small, deterministic subset of the arrays used by the current
training runs.  It compares raw linear-power input with the bachelor's
magnitude + per-chunk min-max representation on identical samples/splits.
"""

import csv
from pathlib import Path

import numpy as np

from utils.diagnostics.noise_shortcut import (
    evaluate_noise_shortcut_dataset,
    summarize_fold_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPO_ROOT / "Pool_boiling" / "Subcooling_20_degrees" / "0.3"
)
EXPERIMENTS = [
    "2025.06.18_0.3_3",
    "2025.07.09_0.3_1",
    "2025.06.11_0.3_2",
]
SOURCE_DIR = "waterflow_20260722_1s_y_power"
MAX_FREQ_DIR = "maxfreq=22kHz"
NOISE_DIRS = [
    "heatflux_no_noise",
    "heatflux_SNR=0",
    "heatflux_SNR=-4",
    "heatflux_SNR=-8",
    "heatflux_SNR=-12",
    "heatflux_SNR=-16",
    "heatflux_SNR=-20",
]
RESULT_DIR = (
    REPO_ROOT / "experiments" / "2026-07-24_noise_shortcut_diagnostic"
)
CHUNKS_PER_SOURCE = 5


def _selected_files(data_path):
    by_label = {}
    for path in sorted(data_path.glob("*.npy"), key=lambda item: item.name):
        label = path.name.split("_", 1)[0]
        by_label.setdefault(label, []).append(path)

    selected = []
    for label, paths in sorted(by_label.items(), key=lambda item: float(item[0])):
        indices = np.linspace(
            0, len(paths) - 1, num=min(CHUNKS_PER_SOURCE, len(paths)), dtype=int
        )
        selected.extend((label, paths[index]) for index in sorted(set(indices)))
    return selected


def _load_subset(data_path):
    x = []
    y = []
    metadata = []
    for label, path in _selected_files(data_path):
        array = np.load(path).astype(np.float32, copy=False)
        x.append(array)
        y.append(float(label))
        metadata.append(
            {
                "sample_filename": path.name,
                # One source WAV exists for each heat-flux label in this data.
                "source_wav_id": f"heat_flux={label}",
                "model_input_power": float(array.mean(dtype=np.float64)),
            }
        )
    if not x:
        raise FileNotFoundError(f"No npy files found: {data_path}")
    return np.asarray(x, dtype=np.float32), np.asarray(y), metadata


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    fold_rows = []
    summary_rows = []
    input_rows = []
    for experiment in EXPERIMENTS:
        for noise_dir in NOISE_DIRS:
            data_path = (
                EXPERIMENT_ROOT
                / experiment
                / "data"
                / "npy"
                / SOURCE_DIR
                / MAX_FREQ_DIR
                / noise_dir
            )
            print(f"{experiment} | {noise_dir}")
            x, y, metadata = _load_subset(data_path)
            result = evaluate_noise_shortcut_dataset(
                x,
                y,
                metadata,
                split_strategies=("kfold", "group_kfold"),
                feature_sets=(
                    "total_power_only",
                    "full_spectrogram_pca",
                    "old_style_magnitude_minmax_pca",
                ),
                folds=3,
                random_seed=42,
                pca_components=100,
                rf_params={
                    "n_estimators": 300,
                    "max_depth": 8,
                    "subsample": 0.8,
                    "colsample_bynode": 0.6,
                },
            )
            context = {
                "experiment_name": experiment,
                "noise_dir": noise_dir,
                "source_dir": SOURCE_DIR,
                "max_freq_dir": MAX_FREQ_DIR,
                "chunks_per_source": CHUNKS_PER_SOURCE,
                "data_path": str(data_path),
            }
            fold_rows.extend({**context, **row} for row in result["fold_metrics"])
            summary_rows.extend(
                {**context, **row}
                for row in summarize_fold_metrics(result["fold_metrics"])
            )
            input_rows.append({**context, **result["input_summary"]})

    _write_csv(RESULT_DIR / "existing_curve_fold_metrics.csv", fold_rows)
    _write_csv(RESULT_DIR / "existing_curve_metrics_summary.csv", summary_rows)
    _write_csv(RESULT_DIR / "existing_curve_input_summary.csv", input_rows)
    print(f"results={RESULT_DIR}")


if __name__ == "__main__":
    main()
