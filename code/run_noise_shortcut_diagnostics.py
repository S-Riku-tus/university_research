"""Run the P0 tests for the counter-intuitive noise/accuracy result.

Run order:
1. Generate the three diagnostic datasets with
   ``2.run_npy_waterflow_2つhighpass.py``.
2. Run this file to compare:
   - total-power-only RF versus full-spectrogram RF,
   - ordinary KFold versus source-WAV GroupKFold,
   - source-relative mixture, source-relative noise-only, and fixed-RMS mixture.

The script is intentionally independent from the long CNN/Transformer run.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from utils.diagnostics.noise_shortcut import (
    _make_rf,
    evaluate_noise_shortcut_dataset,
    load_npy_dataset_with_metadata,
    summarize_fold_metrics,
    summarize_realized_snr,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTIC_CONFIG = {
    "purpose": "P0 noise-amplitude shortcut and split-leakage diagnosis",
    "experiment_root_parts": [
        "Pool_boiling",
        "Subcooling_20_degrees",
        "0.3",
    ],
    "experiment_names": [
        "2025.06.18_0.3_3",
        "2025.07.09_0.3_1",
        "2025.06.11_0.3_2",
    ],
    "max_freq_dir": "maxfreq=22kHz",
    "dataset_variants": [
        {
            "variant": "relative_mixture",
            "source_dir": "waterflow_20260724_1s_p0_relative_mixture",
            "noise_dirs": ["heatflux_no_noise", "heatflux_SNR=-20"],
        },
        {
            "variant": "relative_noise_only",
            "source_dir": "waterflow_20260724_1s_p0_relative_noise_only",
            "noise_dirs": ["heatflux_SNR=-20"],
        },
        {
            "variant": "fixed_mixture",
            "source_dir": "waterflow_20260724_1s_p0_fixed_mixture",
            "noise_dirs": ["heatflux_no_noise", "heatflux_SNR=-20"],
        },
    ],
    "split_strategies": ["kfold", "group_kfold"],
    "feature_sets": [
        "total_power_only",
        "full_spectrogram_pca",
        "old_style_magnitude_minmax_pca",
    ],
    "folds": 3,
    "random_seed": 42,
    "pca_components": 100,
    "rf_params": {
        "n_estimators": 300,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bynode": 0.6,
    },
    "result_dir": [
        "experiments",
        "2026-07-24_noise_shortcut_diagnostic",
    ],
}


def _write_csv(path, rows):
    if not rows:
        return
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


def _job_context(experiment_name, variant, noise_dir, data_path):
    return {
        "experiment_name": experiment_name,
        "variant": variant,
        "noise_dir": noise_dir,
        "data_path": str(data_path),
    }


def _leave_one_experiment_out_scalar(datasets):
    """Test whether the amplitude cue transfers across experiment days."""
    rows = []
    for (variant, noise_dir), by_experiment in sorted(datasets.items()):
        experiments = sorted(by_experiment)
        if len(experiments) < 3:
            continue
        for held_out in experiments:
            train_experiments = [name for name in experiments if name != held_out]
            train_y = np.concatenate(
                [by_experiment[name][0] for name in train_experiments]
            )
            test_y = by_experiment[held_out][0]
            train_power = np.concatenate(
                [by_experiment[name][1] for name in train_experiments]
            ).reshape(-1, 1)
            test_power = by_experiment[held_out][1].reshape(-1, 1)

            scaler = MinMaxScaler()
            train_y_scaled = scaler.fit_transform(
                train_y.reshape(-1, 1)
            ).ravel()
            model = _make_rf(
                42,
                {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "subsample": 0.8,
                    "colsample_bynode": 0.6,
                },
            )
            model.fit(train_power, train_y_scaled)
            prediction = scaler.inverse_transform(
                model.predict(test_power).reshape(-1, 1)
            ).ravel()
            rows.append(
                {
                    "variant": variant,
                    "noise_dir": noise_dir,
                    "held_out_experiment": held_out,
                    "train_experiments": ";".join(train_experiments),
                    "train_samples": len(train_y),
                    "test_samples": len(test_y),
                    "r2": float(r2_score(test_y, prediction)),
                    "rmse": float(
                        np.sqrt(mean_squared_error(test_y, prediction))
                    ),
                    "mae": float(mean_absolute_error(test_y, prediction)),
                    "prediction_pearson": float(
                        np.corrcoef(test_y, prediction)[0, 1]
                    ),
                }
            )
    return rows


def build_jobs():
    experiment_root = REPO_ROOT.joinpath(
        *DIAGNOSTIC_CONFIG["experiment_root_parts"]
    )
    jobs = []
    for experiment_name in DIAGNOSTIC_CONFIG["experiment_names"]:
        for variant_config in DIAGNOSTIC_CONFIG["dataset_variants"]:
            for noise_dir in variant_config["noise_dirs"]:
                data_path = (
                    experiment_root
                    / experiment_name
                    / "data"
                    / "npy"
                    / variant_config["source_dir"]
                    / DIAGNOSTIC_CONFIG["max_freq_dir"]
                    / noise_dir
                )
                jobs.append(
                    {
                        **_job_context(
                            experiment_name,
                            variant_config["variant"],
                            noise_dir,
                            data_path,
                        ),
                        "source_dir": variant_config["source_dir"],
                    }
                )
    return jobs


def main(skip_missing=False, scalar_only=False):
    result_root = REPO_ROOT.joinpath(*DIAGNOSTIC_CONFIG["result_dir"])
    result_root.mkdir(parents=True, exist_ok=True)

    feature_sets = (
        ["total_power_only"]
        if scalar_only
        else DIAGNOSTIC_CONFIG["feature_sets"]
    )
    all_fold_metrics = []
    all_summaries = []
    all_predictions = []
    all_split_assignments = []
    all_input_summaries = []
    all_snr_summaries = []
    completed_jobs = []
    missing_jobs = []
    cross_experiment_datasets = {}

    jobs = build_jobs()
    for job_index, job in enumerate(jobs, start=1):
        data_path = Path(job["data_path"])
        manifest_path = data_path / "chunk_manifest.csv"
        if not data_path.is_dir() or not manifest_path.is_file():
            missing_jobs.append(job)
            message = (
                f"[missing {job_index}/{len(jobs)}] "
                f"{job['experiment_name']} | {job['variant']} | "
                f"{job['noise_dir']} | {data_path}"
            )
            if skip_missing:
                print(message)
                continue
            raise FileNotFoundError(
                message
                + "\nRun code/2.run_npy_waterflow_2つhighpass.py first."
            )

        print(
            f"[diagnostic {job_index}/{len(jobs)}] "
            f"{job['experiment_name']} | {job['variant']} | "
            f"{job['noise_dir']}"
        )
        x, y, metadata = load_npy_dataset_with_metadata(data_path)
        model_input_power = np.asarray(
            [float(row["model_input_power"]) for row in metadata],
            dtype=float,
        )
        cross_experiment_datasets.setdefault(
            (job["variant"], job["noise_dir"]), {}
        )[job["experiment_name"]] = (y, model_input_power)
        evaluation = evaluate_noise_shortcut_dataset(
            x,
            y,
            metadata,
            split_strategies=DIAGNOSTIC_CONFIG["split_strategies"],
            feature_sets=feature_sets,
            folds=DIAGNOSTIC_CONFIG["folds"],
            random_seed=DIAGNOSTIC_CONFIG["random_seed"],
            pca_components=DIAGNOSTIC_CONFIG["pca_components"],
            rf_params=DIAGNOSTIC_CONFIG["rf_params"],
        )
        context = _job_context(
            job["experiment_name"],
            job["variant"],
            job["noise_dir"],
            data_path,
        )

        fold_rows = [
            {**context, **row} for row in evaluation["fold_metrics"]
        ]
        all_fold_metrics.extend(fold_rows)
        all_summaries.extend(
            {
                **context,
                **row,
            }
            for row in summarize_fold_metrics(evaluation["fold_metrics"])
        )
        all_predictions.extend(
            {**context, **row} for row in evaluation["predictions"]
        )
        all_split_assignments.extend(
            {**context, **row} for row in evaluation["split_assignments"]
        )
        all_input_summaries.append(
            {**context, **evaluation["input_summary"]}
        )
        all_snr_summaries.append(
            {**context, **summarize_realized_snr(metadata)}
        )
        completed_jobs.append(job)

    if not completed_jobs:
        raise RuntimeError("No diagnostic dataset was evaluated.")

    _write_csv(result_root / "fold_metrics.csv", all_fold_metrics)
    _write_csv(result_root / "metrics_summary.csv", all_summaries)
    _write_csv(result_root / "fold_predictions.csv", all_predictions)
    _write_csv(
        result_root / "split_assignments.csv", all_split_assignments
    )
    _write_csv(result_root / "input_power_summary.csv", all_input_summaries)
    _write_csv(result_root / "realized_snr_summary.csv", all_snr_summaries)
    cross_rows = _leave_one_experiment_out_scalar(
        cross_experiment_datasets
    )
    _write_csv(
        result_root / "leave_one_experiment_out_scalar.csv", cross_rows
    )

    run_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "status": "diagnostic_not_final_claim",
        "config": {
            **DIAGNOSTIC_CONFIG,
            "feature_sets_used": feature_sets,
        },
        "completed_jobs": completed_jobs,
        "missing_jobs": missing_jobs,
        "outputs": [
            "fold_metrics.csv",
            "metrics_summary.csv",
            "fold_predictions.csv",
            "split_assignments.csv",
            "input_power_summary.csv",
            "realized_snr_summary.csv",
            "leave_one_experiment_out_scalar.csv",
        ],
        "interpretation": {
            "P0-2": (
                "If total_power_only approaches full_spectrogram_pca, "
                "absolute input power is a dominant shortcut."
            ),
            "P0-3": (
                "High performance for relative_noise_only means scaled "
                "water-flow amplitude predicts heat flux without boiling sound."
            ),
            "P0-4": (
                "If the -20 dB gain disappears for fixed_mixture, per-source "
                "relative scaling was a primary cause."
            ),
            "P0-5": (
                "A large KFold-to-GroupKFold drop indicates dependence on "
                "chunks from the same source WAV."
            ),
            "P0-6": (
                "realized_snr_summary.csv reports chunk-level SNR rather than "
                "only the requested whole-WAV SNR."
            ),
        },
    }
    with (result_root / "diagnostic_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(run_manifest, f, ensure_ascii=False, indent=2)

    print(f"completed_jobs={len(completed_jobs)}")
    print(f"results={result_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Evaluate available jobs and record missing jobs in the manifest.",
    )
    parser.add_argument(
        "--scalar-only",
        action="store_true",
        help="Run only the fast one-variable total-power RF baseline.",
    )
    args = parser.parse_args()
    main(skip_missing=args.skip_missing, scalar_only=args.scalar_only)
