"""Export a small, Git-friendly snapshot from ignored ensemble result folders.

The raw ``regression_result`` tree contains weights, figures, predictions, and
other generated artifacts that should stay outside Git.  This script extracts
the tabular evidence needed to review an important run on GitHub.

Example
-------
python code/export_ensemble_result_snapshot.py ^
  --result-root Pool_boiling/.../2025.06.18_0.3_3/.../20260902_selected_log_architecture ^
  --result-root Pool_boiling/.../2025.07.09_0.3_1/.../20260902_selected_log_architecture ^
  --output-dir experiments/2026-09-02_selected_log_architecture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


SINGLE_MODEL_KEYS = ("rf", "cnntf_v2_gap", "alexnet")
SAFE_ENSEMBLE_KEYS = (
    "ensemble__simple_equal",
    "ensemble__prediction_max",
    "ensemble__inner_holdout",
)
LEGACY_ENSEMBLE_KEY = "ensemble__val_fold_legacy"

IDENTITY_COLUMNS = [
    "created_at",
    "experiment_name",
    "max_freq_hz",
    "noise_dir_name",
    "snr_value",
    "threshold",
    "model_key",
    "model_label",
    "epochs_completed",
    "stopped_by_memory_error",
]
METRIC_COLUMNS = [
    "r2_mean",
    "r2_se",
    "rmse_all_mean",
    "rmse_all_se",
    "mae_all_mean",
    "mae_all_se",
    "rmse_onb_mean",
    "rmse_onb_se",
    "mae_onb_mean",
    "mae_onb_se",
    "roc_auc_cont_mean",
    "roc_auc_cont_se",
    "pr_auc_cont_mean",
    "pr_auc_cont_se",
    "accuracy_mean",
    "accuracy_se",
    "precision_mean",
    "precision_se",
    "recall_mean",
    "recall_se",
    "f1_mean",
    "f1_se",
]
AGGREGATE_METRICS = [
    "r2_mean",
    "rmse_all_mean",
    "mae_all_mean",
    "rmse_onb_mean",
    "mae_onb_mean",
    "roc_auc_cont_mean",
    "pr_auc_cont_mean",
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        action="append",
        required=True,
        type=Path,
        help="Result-date directory containing tuning_summary.csv (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Tracked experiment snapshot directory.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_safe(model_key: str) -> bool:
    return model_key != LEGACY_ENSEMBLE_KEY


def load_summaries(result_roots: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    sources = []
    for root in result_roots:
        summary_path = root / "tuning_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing tuning summary: {summary_path}")
        frame = pd.read_csv(summary_path)
        frames.append(frame)
        sources.append(
            {
                "result_root": root.as_posix(),
                "tuning_summary_sha256": sha256(summary_path),
                "source_rows": int(len(frame)),
            }
        )

    data = pd.concat(frames, ignore_index=True)
    condition_keys = ["experiment_name", "max_freq_hz", "snr_value", "model_key"]
    data = data.drop_duplicates(condition_keys, keep="last").copy()
    data["claim_safe"] = data["model_key"].map(claim_safe)
    return data, sources


def write_condition_metrics(data: pd.DataFrame, output_dir: Path) -> None:
    columns = [column for column in IDENTITY_COLUMNS + ["claim_safe"] + METRIC_COLUMNS
               if column in data.columns]
    ordered = data.sort_values(
        ["experiment_name", "max_freq_hz", "snr_value", "model_key"]
    )
    ordered[columns].to_csv(
        output_dir / "metrics_by_condition.csv", index=False, float_format="%.10g"
    )


def write_snr_metrics(data: pd.DataFrame, output_dir: Path) -> None:
    available = [column for column in AGGREGATE_METRICS if column in data.columns]
    grouped = data.groupby(
        ["experiment_name", "snr_value", "model_key", "model_label", "claim_safe"],
        dropna=False,
    )
    summary = grouped[available].mean().reset_index()
    count = grouped["max_freq_hz"].nunique().rename("frequency_condition_count")
    summary = summary.merge(count.reset_index())
    summary = summary.rename(
        columns={column: f"{column}_across_frequencies" for column in available}
    )
    summary.sort_values(
        ["experiment_name", "snr_value", "model_key"]
    ).to_csv(output_dir / "metrics_by_snr.csv", index=False, float_format="%.10g")


def write_ensemble_comparison(data: pd.DataFrame, output_dir: Path) -> None:
    condition_keys = ["experiment_name", "max_freq_hz", "snr_value"]
    single = data[data["model_key"].isin(SINGLE_MODEL_KEYS)].copy()
    best_index = single.groupby(condition_keys)["r2_mean"].idxmax()
    best = single.loc[
        best_index,
        condition_keys + ["model_key", "r2_mean", "recall_mean", "f1_mean"],
    ].rename(
        columns={
            "model_key": "best_single_model_by_r2",
            "r2_mean": "best_single_r2",
            "recall_mean": "best_single_recall",
            "f1_mean": "best_single_f1",
        }
    )

    ensemble_keys = set(SAFE_ENSEMBLE_KEYS) | {LEGACY_ENSEMBLE_KEY}
    comparison = data[data["model_key"].isin(ensemble_keys)].merge(
        best, on=condition_keys, how="left"
    )
    comparison["r2_delta_vs_best_single"] = (
        comparison["r2_mean"] - comparison["best_single_r2"]
    )
    comparison["recall_delta_vs_best_r2_single"] = (
        comparison["recall_mean"] - comparison["best_single_recall"]
    )
    comparison["f1_delta_vs_best_r2_single"] = (
        comparison["f1_mean"] - comparison["best_single_f1"]
    )
    columns = condition_keys + [
        "model_key",
        "model_label",
        "claim_safe",
        "best_single_model_by_r2",
        "best_single_r2",
        "r2_mean",
        "r2_delta_vs_best_single",
        "best_single_recall",
        "recall_mean",
        "recall_delta_vs_best_r2_single",
        "best_single_f1",
        "f1_mean",
        "f1_delta_vs_best_r2_single",
    ]
    comparison.sort_values(condition_keys + ["model_key"])[columns].to_csv(
        output_dir / "ensemble_comparison.csv", index=False, float_format="%.10g"
    )


def write_completion(data: pd.DataFrame, output_dir: Path) -> None:
    conditions = data[
        ["experiment_name", "max_freq_hz", "snr_value"]
    ].drop_duplicates()
    rows = []
    for experiment_name, group in conditions.groupby("experiment_name"):
        rows.append(
            {
                "experiment_name": experiment_name,
                "completed_condition_count": int(len(group)),
                "completed_frequency_count": int(group["max_freq_hz"].nunique()),
                "completed_frequencies": ";".join(sorted(group["max_freq_hz"].unique())),
                "completed_snr_count": int(group["snr_value"].nunique()),
                "completed_snrs": ";".join(sorted(map(str, group["snr_value"].unique()))),
            }
        )
    pd.DataFrame(rows).sort_values("experiment_name").to_csv(
        output_dir / "completion.csv", index=False
    )


def _windows_extended_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def write_xai_frequency_summary(
    result_roots: list[Path], data: pd.DataFrame, output_dir: Path
) -> int:
    """Export the top frequency mask group for every completed XAI condition."""
    condition_lookup = {
        (row.experiment_name, row.max_freq_hz, row.noise_dir_name): str(row.snr_value)
        for row in data[
            ["experiment_name", "max_freq_hz", "noise_dir_name", "snr_value"]
        ].drop_duplicates().itertuples(index=False)
    }
    rows = []
    source_file_count = 0
    for root in result_roots:
        source_summary = pd.read_csv(root / "tuning_summary.csv")
        experiments = source_summary["experiment_name"].dropna().unique()
        if len(experiments) != 1:
            raise ValueError(f"Expected one experiment in {root}, found {experiments}")
        experiment_name = str(experiments[0])

        for directory, _, filenames in os.walk(_windows_extended_path(root)):
            if "model_group_mask_comparison.csv" not in filenames:
                continue
            source_file_count += 1
            normalized_parts = directory.replace("\\", "/").split("/")
            max_freq_hz = next(
                (part for part in normalized_parts if part.startswith("maxfreq=")), None
            )
            noise_dir_name = next(
                (part for part in normalized_parts if part.startswith("heatflux_")), None
            )
            if max_freq_hz is None or noise_dir_name is None:
                raise ValueError(f"Cannot parse condition from {directory}")

            source_path = os.path.join(directory, "model_group_mask_comparison.csv")
            comparison = pd.read_csv(source_path)
            comparison = comparison[comparison["axis"] == "frequency"].copy()
            comparison["r2_drop"] = pd.to_numeric(
                comparison["r2_drop"], errors="coerce"
            )
            for model_key, model_rows in comparison.groupby("model_key"):
                grouped = model_rows.groupby("group", as_index=False).agg(
                    mean_r2_drop_after_mask=("r2_drop", "mean"),
                    contributing_fold_rows=("r2_drop", "count"),
                )
                grouped = grouped.dropna(subset=["mean_r2_drop_after_mask"])
                if grouped.empty:
                    continue
                top = grouped.loc[grouped["mean_r2_drop_after_mask"].idxmax()]
                rows.append(
                    {
                        "experiment_name": experiment_name,
                        "max_freq_hz": max_freq_hz,
                        "noise_dir_name": noise_dir_name,
                        "snr_value": condition_lookup.get(
                            (experiment_name, max_freq_hz, noise_dir_name), ""
                        ),
                        "model_key": model_key,
                        "top_frequency_group": top["group"],
                        "mean_r2_drop_after_mask": top["mean_r2_drop_after_mask"],
                        "contributing_fold_rows": int(top["contributing_fold_rows"]),
                    }
                )

    if not rows:
        return source_file_count

    detail = pd.DataFrame(rows).sort_values(
        ["experiment_name", "max_freq_hz", "snr_value", "model_key"]
    )
    detail.to_csv(
        output_dir / "xai_top_frequency_group_by_condition.csv",
        index=False,
        float_format="%.10g",
    )
    counts = (
        detail.groupby(
            ["experiment_name", "model_key", "top_frequency_group"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "condition_count"})
        .sort_values(["experiment_name", "model_key", "condition_count"], ascending=[True, True, False])
    )
    counts.to_csv(output_dir / "xai_top_frequency_group_counts.csv", index=False)
    return source_file_count


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data, sources = load_summaries(args.result_root)

    write_condition_metrics(data, output_dir)
    write_snr_metrics(data, output_dir)
    write_ensemble_comparison(data, output_dir)
    write_completion(data, output_dir)
    xai_source_file_count = write_xai_frequency_summary(
        args.result_root, data, output_dir
    )

    condition_count = int(
        len(data[["experiment_name", "max_freq_hz", "snr_value"]].drop_duplicates())
    )
    manifest = {
        "schema_version": 1,
        "source_artifacts_are_git_ignored": True,
        "source_summaries": sources,
        "deduplicated_metric_rows": int(len(data)),
        "completed_condition_count": condition_count,
        "single_model_keys": list(SINGLE_MODEL_KEYS),
        "safe_ensemble_keys": list(SAFE_ENSEMBLE_KEYS),
        "legacy_not_claim_safe": LEGACY_ENSEMBLE_KEY,
        "xai_source_file_count": xai_source_file_count,
        "generated_files": [
            "completion.csv",
            "metrics_by_condition.csv",
            "metrics_by_snr.csv",
            "ensemble_comparison.csv",
            "xai_top_frequency_group_by_condition.csv",
            "xai_top_frequency_group_counts.csv",
        ],
    }
    (output_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
