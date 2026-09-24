from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    REPO_ROOT
    / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3"
    / "regression_result/npy/ensemble/20260924"
)
OUTPUT_DIR = Path(__file__).resolve().parent
CLEAN_ANALYSIS_DIR = REPO_ROOT / "experiments/2026-09-24_clean_train_noise_inference"
RUN_GLOB = "matched_nois_xd-t0611-v0618_iw3-nm_s0_e200_*"
THRESHOLD = 271677.6816
SNR_PLAN = [
    ("clean", "heatflux_no_noise", "no_noise"),
    ("0", "heatflux_reference_SNR=0", "0"),
    ("-4", "heatflux_reference_SNR=-4", "-4"),
    ("-8", "heatflux_reference_SNR=-8", "-8"),
    ("-12", "heatflux_reference_SNR=-12", "-12"),
    ("-16", "heatflux_reference_SNR=-16", "-16"),
    ("-20", "heatflux_reference_SNR=-20", "-20"),
]
SINGLE_KEYS = ["randomforest", "conformer", "alexnet"]
ENSEMBLE_KEYS = ["ensemble__simple_equal", "ensemble__inner_holdout"]
MODEL_LABELS = {
    "randomforest": "RandomForest",
    "conformer": "Conformer",
    "alexnet": "AlexNet",
    "ensemble__simple_equal": "Equal ensemble",
    "ensemble__inner_holdout": "Inner-holdout ensemble",
}
MODEL_KEYS_BY_OUTPUT_LABEL = {
    "RandomForest": "randomforest",
    "Conformer": "conformer",
    "AlexNet": "alexnet",
    "Ensemble simple equal": "ensemble__simple_equal",
    "Ensemble": "ensemble__inner_holdout",
}


def extended(path: Path) -> Path:
    """Allow reading the long result paths on Windows."""
    resolved = path.resolve()
    return Path("\\\\?\\" + str(resolved))


def read_json(path: Path):
    return json.loads(extended(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(extended(path))


def suffix_for(snr_value: str) -> str:
    return "no_noise" if snr_value == "no_noise" else snr_value


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(actual)))))


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(actual))))


def r2(actual, predicted) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    denominator = np.sum(np.square(actual - np.mean(actual)))
    if denominator == 0:
        return float("nan")
    return float(1 - np.sum(np.square(predicted - actual)) / denominator)


def main():
    condition_audit_rows = []
    metric_rows = []
    region_rows = []
    correlation_rows = []
    weight_rows = []
    prediction_cache = {}

    run_dirs = sorted(RESULT_ROOT.glob(RUN_GLOB))
    if len(run_dirs) != 3:
        raise ValueError(f"Expected 3 run directories, found {len(run_dirs)}: {run_dirs}")

    for run_dir in run_dirs:
        for snr_label, noise_dir, snr_value in SNR_PLAN:
            condition_dir = run_dir / "maxfreq=22kHz" / noise_dir
            suffix = suffix_for(snr_value)
            required_paths = [
                condition_dir / "completed.json",
                condition_dir / "run_manifest.json",
                condition_dir / "split_manifest.json",
                condition_dir / "training_validation_f1.json",
                condition_dir / f"metrics_summary_{suffix}.csv",
                condition_dir / f"ensemble_weights_{suffix}.csv",
                condition_dir / "fold_pred" / f"pred_f1_{suffix}.csv",
            ]
            missing = [str(path) for path in required_paths if not extended(path).exists()]
            if missing:
                raise FileNotFoundError(f"Missing outputs for {condition_dir}: {missing}")

            manifest = read_json(condition_dir / "run_manifest.json")
            completed = read_json(condition_dir / "completed.json")
            split = read_json(condition_dir / "split_manifest.json")
            training_validation = read_json(condition_dir / "training_validation_f1.json")
            context = manifest["learning_context"]
            seed = int(manifest["validation_config"]["run"]["random_seed"])
            fit_id = completed["fit_ids"][0]
            split_fold = split["folds"][0]
            if fit_id != split_fold["fit_id"]:
                raise ValueError(f"fit_id mismatch: {condition_dir}")
            if context["training_noise_dir"] != noise_dir or context["evaluation_noise_dir"] != noise_dir:
                raise ValueError(f"Noise scope mismatch: {condition_dir}")
            if context["ensemble_weight_scope"] != "per_training_noise":
                raise ValueError(f"Unexpected weight scope: {condition_dir}")
            if training_validation["scope"] != "training_days_only" or training_validation["test_used"]:
                raise ValueError(f"Evaluation data used for epoch selection: {condition_dir}")

            weights_frame = read_csv(condition_dir / f"ensemble_weights_{suffix}.csv")
            inner_weights = weights_frame.loc[weights_frame["strategy_name"] == "inner_holdout"].iloc[0]
            selected_epochs = training_validation["selected_epochs"]
            inner_errors = split_fold["inner_holdout_errors"]
            weight_values = np.asarray([inner_weights[key] for key in SINGLE_KEYS], dtype=float)
            condition_audit_rows.append(
                {
                    "seed": seed,
                    "snr": snr_label,
                    "run_instance_id": manifest["run_instance_id"],
                    "run_hash": manifest["run_hash"],
                    "fit_id": fit_id,
                    "training_noise_dir": context["training_noise_dir"],
                    "evaluation_noise_dir": context["evaluation_noise_dir"],
                    "ensemble_weight_scope": context["ensemble_weight_scope"],
                    "training_experiments": ";".join(context["training_experiments"]),
                    "evaluation_experiments": ";".join(context["evaluation_experiments"]),
                    "training_validation_scope": training_validation["scope"],
                    "training_validation_test_used": training_validation["test_used"],
                    "conformer_epoch": selected_epochs["conformer"],
                    "alexnet_epoch": selected_epochs["alexnet"],
                    "weight_randomforest": inner_weights["randomforest"],
                    "weight_conformer": inner_weights["conformer"],
                    "weight_alexnet": inner_weights["alexnet"],
                    "effective_model_count": float(1.0 / np.sum(np.square(weight_values))),
                }
            )
            for model_key in SINGLE_KEYS:
                weight_rows.append(
                    {
                        "seed": seed,
                        "snr": snr_label,
                        "model_key": model_key,
                        "model": MODEL_LABELS[model_key],
                        "inner_holdout_error_1_minus_r2": float(inner_errors[model_key]),
                        "inner_holdout_weight": float(inner_weights[model_key]),
                        "selected_epoch": (
                            int(selected_epochs[model_key]) if model_key in selected_epochs else np.nan
                        ),
                    }
                )

            metrics = read_csv(condition_dir / f"metrics_summary_{suffix}.csv")
            metrics["model_key"] = metrics["model"].map(MODEL_KEYS_BY_OUTPUT_LABEL)
            if metrics["model_key"].isna().any():
                raise ValueError(f"Unknown model label in {condition_dir}")
            for row in metrics.to_dict("records"):
                metric_rows.append(
                    {
                        "seed": seed,
                        "snr": snr_label,
                        "model_key": row["model_key"],
                        "model": MODEL_LABELS[row["model_key"]],
                        **{
                            name: row[name]
                            for name in [
                                "r2_mean",
                                "rmse_all_mean",
                                "mae_all_mean",
                                "r2_high_mean",
                                "rmse_high_mean",
                                "mae_high_mean",
                                "rmse_onb_mean",
                                "mae_onb_mean",
                                "roc_auc_cont_mean",
                                "pr_auc_cont_mean",
                                "accuracy_mean",
                                "precision_mean",
                                "recall_mean",
                                "f1_mean",
                            ]
                        },
                    }
                )

            prediction = read_csv(condition_dir / "fold_pred" / f"pred_f1_{suffix}.csv")
            if len(prediction) != 1080 or prediction["sample_index"].nunique() != 1080:
                raise ValueError(f"Unexpected prediction keys: {condition_dir}")
            prediction_cache[(seed, snr_label)] = prediction
            actual = prediction["y_true"].to_numpy()
            regions = {
                "below": actual < THRESHOLD * 0.9,
                "near": (actual >= THRESHOLD * 0.9) & (actual <= THRESHOLD * 1.1),
                "above": actual > THRESHOLD * 1.1,
                "all": np.ones(len(actual), dtype=bool),
            }
            residuals = {}
            for model_key, model_label in MODEL_LABELS.items():
                predicted = prediction[model_key].to_numpy()
                residuals[model_key] = predicted - actual
                for region, mask in regions.items():
                    region_rows.append(
                        {
                            "seed": seed,
                            "snr": snr_label,
                            "model_key": model_key,
                            "model": model_label,
                            "region": region,
                            "chunks": int(mask.sum()),
                            "r2": r2(actual[mask], predicted[mask]),
                            "rmse": rmse(actual[mask], predicted[mask]),
                            "mae": mae(actual[mask], predicted[mask]),
                            "mean_residual": float(np.mean(predicted[mask] - actual[mask])),
                            "misses": int(
                                np.sum((actual[mask] >= THRESHOLD) & (predicted[mask] < THRESHOLD))
                            ),
                            "false_alarms": int(
                                np.sum((actual[mask] < THRESHOLD) & (predicted[mask] >= THRESHOLD))
                            ),
                        }
                    )
            for left, right in [
                ("randomforest", "conformer"),
                ("randomforest", "alexnet"),
                ("conformer", "alexnet"),
            ]:
                correlation_rows.append(
                    {
                        "seed": seed,
                        "snr": snr_label,
                        "model_left": MODEL_LABELS[left],
                        "model_right": MODEL_LABELS[right],
                        "residual_correlation": float(np.corrcoef(residuals[left], residuals[right])[0, 1]),
                    }
                )

    condition_audit = pd.DataFrame(condition_audit_rows).sort_values(["seed", "snr"])
    metrics = pd.DataFrame(metric_rows)
    regions = pd.DataFrame(region_rows)
    correlations = pd.DataFrame(correlation_rows)
    weights = pd.DataFrame(weight_rows)
    if sorted(condition_audit["seed"].unique().tolist()) != [42, 43, 44]:
        raise ValueError(f"Unexpected seeds: {condition_audit['seed'].unique()}")
    if not (condition_audit.groupby("seed")["fit_id"].nunique() == 7).all():
        raise ValueError("Each matched seed must have seven distinct fit_ids")

    run_audit = (
        condition_audit.groupby(["seed", "run_instance_id", "run_hash"], as_index=False)
        .agg(
            completed_conditions=("snr", "count"),
            unique_fit_ids_across_snr=("fit_id", "nunique"),
            unique_training_noise_dirs=("training_noise_dir", "nunique"),
            unique_weight_scopes=("ensemble_weight_scope", "nunique"),
        )
        .sort_values("seed")
    )

    aggregate_columns = [
        "r2_mean",
        "rmse_all_mean",
        "mae_all_mean",
        "rmse_onb_mean",
        "roc_auc_cont_mean",
        "recall_mean",
        "f1_mean",
    ]
    aggregate = (
        metrics.groupby(["snr", "model_key", "model"], sort=False)[aggregate_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]

    metric_index = metrics.set_index(["seed", "snr", "model_key"])
    region_index = regions.set_index(["seed", "snr", "model_key", "region"])
    weight_index = weights.set_index(["seed", "snr", "model_key"])
    delta_rows = []
    oracle_rows = []
    grid_weights = []
    for rf_percent in range(101):
        for conformer_percent in range(101 - rf_percent):
            alexnet_percent = 100 - rf_percent - conformer_percent
            grid_weights.append(
                [rf_percent / 100, conformer_percent / 100, alexnet_percent / 100]
            )
    grid_weights = np.asarray(grid_weights)

    for seed in sorted(metrics["seed"].unique()):
        for snr_label, _, _ in SNR_PLAN:
            singles = metrics[
                (metrics["seed"] == seed)
                & (metrics["snr"] == snr_label)
                & metrics["model_key"].isin(SINGLE_KEYS)
            ].sort_values("rmse_all_mean")
            best_single_key = singles.iloc[0]["model_key"]
            prediction = prediction_cache[(seed, snr_label)]
            actual = prediction["y_true"].to_numpy()
            residual_matrix = np.column_stack(
                [prediction[key].to_numpy() - actual for key in SINGLE_KEYS]
            )
            second_moment = residual_matrix.T @ residual_matrix / len(residual_matrix)
            grid_mse = np.einsum("gi,ij,gj->g", grid_weights, second_moment, grid_weights)
            oracle_index = int(np.argmin(grid_mse))
            oracle_weight = grid_weights[oracle_index]
            oracle_rmse = float(np.sqrt(max(grid_mse[oracle_index], 0)))
            learned_weight = np.asarray(
                [weight_index.loc[(seed, snr_label, key), "inner_holdout_weight"] for key in SINGLE_KEYS]
            )
            evaluation_rmse = np.asarray(
                [metric_index.loc[(seed, snr_label, key), "rmse_all_mean"] for key in SINGLE_KEYS]
            )
            rank_alignment = float(
                pd.Series(learned_weight).corr(pd.Series(-evaluation_rmse), method="spearman")
            )
            oracle_rows.append(
                {
                    "seed": seed,
                    "snr": snr_label,
                    "best_single_key": best_single_key,
                    "best_single_rmse": metric_index.loc[(seed, snr_label, best_single_key), "rmse_all_mean"],
                    "oracle_weight_randomforest": oracle_weight[0],
                    "oracle_weight_conformer": oracle_weight[1],
                    "oracle_weight_alexnet": oracle_weight[2],
                    "oracle_rmse": oracle_rmse,
                    "learned_weight_l1_distance_to_oracle": float(np.abs(learned_weight - oracle_weight).sum()),
                    "max_weight_model_key": SINGLE_KEYS[int(np.argmax(learned_weight))],
                    "max_weight_matches_evaluation_best": bool(
                        SINGLE_KEYS[int(np.argmax(learned_weight))] == best_single_key
                    ),
                    "weight_vs_negative_evaluation_rmse_spearman": rank_alignment,
                }
            )
            for ensemble_key in ENSEMBLE_KEYS:
                ensemble = metric_index.loc[(seed, snr_label, ensemble_key)]
                rf = metric_index.loc[(seed, snr_label, "randomforest")]
                best = metric_index.loc[(seed, snr_label, best_single_key)]
                row = {
                    "seed": seed,
                    "snr": snr_label,
                    "ensemble_key": ensemble_key,
                    "ensemble": MODEL_LABELS[ensemble_key],
                    "best_single_key": best_single_key,
                    "best_single": MODEL_LABELS[best_single_key],
                    "delta_r2_vs_rf": ensemble["r2_mean"] - rf["r2_mean"],
                    "delta_rmse_vs_rf": ensemble["rmse_all_mean"] - rf["rmse_all_mean"],
                    "relative_rmse_vs_rf": ensemble["rmse_all_mean"] / rf["rmse_all_mean"] - 1,
                    "delta_rmse_vs_best_single": ensemble["rmse_all_mean"] - best["rmse_all_mean"],
                    "relative_rmse_vs_best_single": (
                        ensemble["rmse_all_mean"] / best["rmse_all_mean"] - 1
                    ),
                    "within_2pct_of_best_single": bool(
                        ensemble["rmse_all_mean"] <= 1.02 * best["rmse_all_mean"]
                    ),
                    "beats_rf": bool(ensemble["rmse_all_mean"] < rf["rmse_all_mean"]),
                    "beats_best_single": bool(ensemble["rmse_all_mean"] < best["rmse_all_mean"]),
                    "delta_onb_rmse_vs_rf": ensemble["rmse_onb_mean"] - rf["rmse_onb_mean"],
                    "delta_rmse_vs_evaluation_oracle": ensemble["rmse_all_mean"] - oracle_rmse,
                }
                for region in ["below", "near", "above"]:
                    row[f"delta_{region}_rmse_vs_rf"] = (
                        region_index.loc[(seed, snr_label, ensemble_key, region), "rmse"]
                        - region_index.loc[(seed, snr_label, "randomforest", region), "rmse"]
                    )
                    row[f"delta_{region}_false_alarms_vs_rf"] = (
                        region_index.loc[(seed, snr_label, ensemble_key, region), "false_alarms"]
                        - region_index.loc[(seed, snr_label, "randomforest", region), "false_alarms"]
                    )
                    row[f"delta_{region}_misses_vs_rf"] = (
                        region_index.loc[(seed, snr_label, ensemble_key, region), "misses"]
                        - region_index.loc[(seed, snr_label, "randomforest", region), "misses"]
                    )
                delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    oracle = pd.DataFrame(oracle_rows)

    # The five-way winner is descriptive only: all SNR cells share the same
    # evaluation WAVs, so 21 cells are not 21 independent experiments.
    winner_rows = []
    for (seed, snr_label), group in metrics.groupby(["seed", "snr"]):
        ordered = group.sort_values("rmse_all_mean")
        winner_rows.append(
            {
                "seed": seed,
                "snr": snr_label,
                "winner_key": ordered.iloc[0]["model_key"],
                "winner": ordered.iloc[0]["model"],
                "winner_rmse": ordered.iloc[0]["rmse_all_mean"],
                "runner_up_key": ordered.iloc[1]["model_key"],
                "runner_up": ordered.iloc[1]["model"],
                "runner_up_rmse": ordered.iloc[1]["rmse_all_mean"],
            }
        )
    winners = pd.DataFrame(winner_rows)

    criteria_rows = []
    for ensemble_key in ENSEMBLE_KEYS:
        ensemble_metrics = metrics[metrics["model_key"] == ensemble_key]
        rf_metrics = metrics[metrics["model_key"] == "randomforest"]
        noisy_ensemble_metrics = ensemble_metrics[ensemble_metrics["snr"] != "clean"]
        noisy_rf_metrics = rf_metrics[rf_metrics["snr"] != "clean"]
        ensemble_mean = ensemble_metrics["rmse_all_mean"].mean()
        rf_mean = rf_metrics["rmse_all_mean"].mean()
        noisy_ensemble_mean = noisy_ensemble_metrics["rmse_all_mean"].mean()
        noisy_rf_mean = noisy_rf_metrics["rmse_all_mean"].mean()
        ensemble_deltas = deltas[deltas["ensemble_key"] == ensemble_key]
        noisy_deltas = ensemble_deltas[ensemble_deltas["snr"] != "clean"]
        snr_relative = (
            ensemble_metrics.groupby("snr")["rmse_all_mean"].mean()
            / rf_metrics.groupby("snr")["rmse_all_mean"].mean()
            - 1
        )
        criteria_rows.append(
            {
                "ensemble_key": ensemble_key,
                "ensemble": MODEL_LABELS[ensemble_key],
                "mean_rmse_7snr_3seed": ensemble_mean,
                "rf_mean_rmse_7snr_3seed": rf_mean,
                "relative_mean_rmse_vs_rf": ensemble_mean / rf_mean - 1,
                "criterion_mean_better_than_rf": bool(ensemble_mean < rf_mean),
                "mean_rmse_noisy_6snr_3seed": noisy_ensemble_mean,
                "rf_mean_rmse_noisy_6snr_3seed": noisy_rf_mean,
                "relative_noisy_mean_rmse_vs_rf": noisy_ensemble_mean / noisy_rf_mean - 1,
                "criterion_noisy_mean_better_than_rf": bool(noisy_ensemble_mean < noisy_rf_mean),
                "cells_within_2pct_best_single": int(
                    ensemble_deltas["within_2pct_of_best_single"].sum()
                ),
                "criterion_at_least_14_of_21_within_2pct": bool(
                    ensemble_deltas["within_2pct_of_best_single"].sum() >= 14
                ),
                "cells_beating_rf": int(ensemble_deltas["beats_rf"].sum()),
                "cells_beating_best_single": int(ensemble_deltas["beats_best_single"].sum()),
                "noisy_cells_within_2pct_best_single": int(
                    noisy_deltas["within_2pct_of_best_single"].sum()
                ),
                "noisy_cells_beating_rf": int(noisy_deltas["beats_rf"].sum()),
                "noisy_cells_beating_best_single": int(noisy_deltas["beats_best_single"].sum()),
                "five_way_wins": int((winners["winner_key"] == ensemble_key).sum()),
                "noisy_five_way_wins": int(
                    ((winners["winner_key"] == ensemble_key) & (winners["snr"] != "clean")).sum()
                ),
                "snr_means_beating_rf": int((snr_relative < 0).sum()),
                "worst_snr_mean_relative_rmse_vs_rf": float(snr_relative.max()),
                "worst_snr": str(snr_relative.idxmax()),
                "criterion_no_snr_over_10pct_worse_than_rf": bool(snr_relative.max() <= 0.10),
            }
        )
    criteria = pd.DataFrame(criteria_rows)

    clean_metrics = read_csv(CLEAN_ANALYSIS_DIR / "metrics_by_seed.csv")
    comparison = metrics.merge(
        clean_metrics,
        on=["seed", "snr", "model_key", "model"],
        suffixes=("_matched", "_clean_only"),
        validate="one_to_one",
    )
    for measure in ["rmse_all_mean", "rmse_onb_mean", "r2_mean", "recall_mean", "f1_mean"]:
        comparison[f"delta_{measure}_matched_minus_clean_only"] = (
            comparison[f"{measure}_matched"] - comparison[f"{measure}_clean_only"]
        )
    comparison["relative_rmse_matched_vs_clean_only"] = (
        comparison["rmse_all_mean_matched"] / comparison["rmse_all_mean_clean_only"] - 1
    )

    clean_regions = read_csv(CLEAN_ANALYSIS_DIR / "region_metrics_by_seed.csv")
    region_comparison = regions.merge(
        clean_regions,
        on=["seed", "snr", "model_key", "model", "region", "chunks"],
        suffixes=("_matched", "_clean_only"),
        validate="one_to_one",
    )
    for measure in ["rmse", "mae", "mean_residual", "misses", "false_alarms"]:
        region_comparison[f"delta_{measure}_matched_minus_clean_only"] = (
            region_comparison[f"{measure}_matched"] - region_comparison[f"{measure}_clean_only"]
        )

    weight_stability = (
        weights.groupby(["snr", "model_key", "model"], sort=False)
        .agg(
            weight_mean=("inner_holdout_weight", "mean"),
            weight_std=("inner_holdout_weight", "std"),
            weight_min=("inner_holdout_weight", "min"),
            weight_max=("inner_holdout_weight", "max"),
            inner_error_mean=("inner_holdout_error_1_minus_r2", "mean"),
            inner_error_std=("inner_holdout_error_1_minus_r2", "std"),
            selected_epoch_mean=("selected_epoch", "mean"),
            selected_epoch_std=("selected_epoch", "std"),
        )
        .reset_index()
    )

    inner_transfer = oracle.merge(
        deltas[deltas["ensemble_key"] == "ensemble__inner_holdout"],
        on=["seed", "snr", "best_single_key"],
        validate="one_to_one",
    )
    transfer_summary_rows = []
    for scope, frame in [
        ("all", inner_transfer),
        ("clean", inner_transfer[inner_transfer["snr"] == "clean"]),
        ("noisy", inner_transfer[inner_transfer["snr"] != "clean"]),
    ]:
        for matches, subgroup in frame.groupby("max_weight_matches_evaluation_best"):
            transfer_summary_rows.append(
                {
                    "scope": scope,
                    "max_weight_matches_evaluation_best": bool(matches),
                    "cells": len(subgroup),
                    "mean_delta_rmse_vs_best_single": subgroup[
                        "delta_rmse_vs_best_single"
                    ].mean(),
                    "mean_delta_rmse_vs_rf": subgroup["delta_rmse_vs_rf"].mean(),
                    "cells_beating_best_single": int(subgroup["beats_best_single"].sum()),
                    "mean_weight_rank_alignment": subgroup[
                        "weight_vs_negative_evaluation_rmse_spearman"
                    ].mean(),
                }
            )
    transfer_summary = pd.DataFrame(transfer_summary_rows)

    # Held-out-day diagnostic only: identify whether excluding a model could
    # help. The winning subset must never be selected from these 6/18 values.
    subset_defs = {
        "rf": ["randomforest"],
        "conformer": ["conformer"],
        "alexnet": ["alexnet"],
        "rf+conformer": ["randomforest", "conformer"],
        "rf+alexnet": ["randomforest", "alexnet"],
        "conformer+alexnet": ["conformer", "alexnet"],
        "rf+conformer+alexnet": SINGLE_KEYS,
    }
    subset_rows = []
    for (seed, snr_label), prediction in prediction_cache.items():
        actual = prediction["y_true"].to_numpy()
        for subset_name, model_keys in subset_defs.items():
            predicted = prediction[model_keys].mean(axis=1).to_numpy()
            subset_rows.append(
                {
                    "seed": seed,
                    "snr": snr_label,
                    "subset": subset_name,
                    "models": ";".join(model_keys),
                    "model_count": len(model_keys),
                    "rmse": rmse(actual, predicted),
                }
            )
    subset_diagnostic = pd.DataFrame(subset_rows)
    best_subset_index = subset_diagnostic.groupby(["seed", "snr"])["rmse"].idxmin()
    subset_diagnostic["best_equal_subset_on_evaluation_day"] = False
    subset_diagnostic.loc[best_subset_index, "best_equal_subset_on_evaluation_day"] = True
    subset_aggregate = (
        subset_diagnostic.groupby(["subset", "models", "model_count"], as_index=False)
        .agg(
            mean_rmse_all=("rmse", "mean"),
            std_rmse_all=("rmse", "std"),
            evaluation_day_wins=("best_equal_subset_on_evaluation_day", "sum"),
        )
    )
    noisy_subset = (
        subset_diagnostic[subset_diagnostic["snr"] != "clean"]
        .groupby("subset")
        .agg(
            mean_rmse_noisy=("rmse", "mean"),
            noisy_evaluation_day_wins=("best_equal_subset_on_evaluation_day", "sum"),
        )
        .reset_index()
    )
    subset_aggregate = subset_aggregate.merge(noisy_subset, on="subset", validate="one_to_one")

    condition_audit.to_csv(OUTPUT_DIR / "condition_audit.csv", index=False)
    run_audit.to_csv(OUTPUT_DIR / "run_audit.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "metrics_seed_aggregate.csv", index=False)
    regions.to_csv(OUTPUT_DIR / "region_metrics_by_seed.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "residual_correlations_by_seed.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "weights_by_seed_snr.csv", index=False)
    weight_stability.to_csv(OUTPUT_DIR / "weight_stability_by_snr.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "ensemble_deltas_by_seed.csv", index=False)
    oracle.to_csv(OUTPUT_DIR / "evaluation_day_oracle_diagnostic.csv", index=False)
    inner_transfer.to_csv(OUTPUT_DIR / "inner_weight_transfer_diagnostic.csv", index=False)
    transfer_summary.to_csv(OUTPUT_DIR / "inner_weight_transfer_summary.csv", index=False)
    subset_diagnostic.to_csv(OUTPUT_DIR / "evaluation_day_equal_subset_diagnostic.csv", index=False)
    subset_aggregate.to_csv(OUTPUT_DIR / "evaluation_day_equal_subset_aggregate.csv", index=False)
    winners.to_csv(OUTPUT_DIR / "winner_by_seed_snr.csv", index=False)
    criteria.to_csv(OUTPUT_DIR / "provisional_success_criteria.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "matched_vs_clean_only.csv", index=False)
    region_comparison.to_csv(OUTPUT_DIR / "matched_vs_clean_only_by_region.csv", index=False)

    print("RUN AUDIT")
    print(run_audit.to_string(index=False))
    print("\nRMSE MEAN ACROSS SEEDS (kW/m2)")
    rmse_table = metrics.pivot_table(
        index="snr", columns="model", values="rmse_all_mean", aggfunc="mean"
    ) / 1000
    print(rmse_table.reindex([item[0] for item in SNR_PLAN]).to_string())
    print("\nPROVISIONAL SUCCESS CRITERIA")
    print(criteria.to_string(index=False))
    print("\nFIVE-WAY WIN COUNTS")
    print(winners["winner"].value_counts().to_string())
    print("\nWEIGHT ALIGNMENT")
    print(
        oracle.groupby("snr", sort=False)
        .agg(
            max_weight_match_rate=("max_weight_matches_evaluation_best", "mean"),
            rank_alignment_mean=("weight_vs_negative_evaluation_rmse_spearman", "mean"),
            learned_l1_to_oracle_mean=("learned_weight_l1_distance_to_oracle", "mean"),
        )
        .to_string()
    )
    print("\nMATCHED MINUS CLEAN-ONLY RMSE, MEAN ACROSS SEEDS (kW/m2)")
    print(
        (comparison.pivot_table(
            index="snr",
            columns="model",
            values="delta_rmse_all_mean_matched_minus_clean_only",
            aggfunc="mean",
        ) / 1000)
        .reindex([item[0] for item in SNR_PLAN])
        .to_string()
    )


if __name__ == "__main__":
    main()
