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
    run_audit = []
    metric_rows = []
    region_rows = []
    correlation_rows = []
    residual_second_moments = []

    run_dirs = sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        clean_dir = run_dir / "maxfreq=22kHz" / "heatflux_no_noise"
        manifest = read_json(clean_dir / "run_manifest.json")
        seed = int(manifest["validation_config"]["run"]["random_seed"])
        artifact = read_json(clean_dir / "fitted_state/fold1/artifact_manifest.json")
        selected_epochs = artifact["selected_epochs"]
        weights = artifact["ensemble_weights"]["ensemble__inner_holdout"]["weights"]
        verification_diffs = [
            float(details["verification"]["max_abs_prediction_difference_w_m2"])
            for details in artifact["models"].values()
        ]

        fit_ids = []
        completed_count = 0
        for snr_label, noise_dir, snr_value in SNR_PLAN:
            condition_dir = run_dir / "maxfreq=22kHz" / noise_dir
            completed_count += int(extended(condition_dir / "completed.json").exists())
            reference = read_json(condition_dir / "fitted_state_reference_f1.json")
            fit_ids.append(reference["fit_id"])

            suffix = suffix_for(snr_value)
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
            actual = prediction["y_true"].to_numpy()
            regions = {
                "below": actual < THRESHOLD * 0.9,
                "near": (actual >= THRESHOLD * 0.9) & (actual <= THRESHOLD * 1.1),
                "above": actual > THRESHOLD * 1.1,
                "all": np.ones(len(actual), dtype=bool),
            }
            for model_key, model_label in MODEL_LABELS.items():
                predicted = prediction[model_key].to_numpy()
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
                            "misses": int(np.sum((actual[mask] >= THRESHOLD) & (predicted[mask] < THRESHOLD))),
                            "false_alarms": int(np.sum((actual[mask] < THRESHOLD) & (predicted[mask] >= THRESHOLD))),
                        }
                    )

            residuals = {
                key: prediction[key].to_numpy() - actual
                for key in ["randomforest", "conformer", "alexnet"]
            }
            residual_matrix = np.column_stack(
                [residuals[key] for key in ["randomforest", "conformer", "alexnet"]]
            )
            residual_second_moments.append(
                {
                    "seed": seed,
                    "snr": snr_label,
                    "matrix": residual_matrix.T @ residual_matrix / len(residual_matrix),
                    "rf_rmse": rmse(actual, prediction["randomforest"].to_numpy()),
                }
            )
            for left, right in [("randomforest", "conformer"), ("randomforest", "alexnet"), ("conformer", "alexnet")]:
                correlation_rows.append(
                    {
                        "seed": seed,
                        "snr": snr_label,
                        "model_left": MODEL_LABELS[left],
                        "model_right": MODEL_LABELS[right],
                        "residual_correlation": float(np.corrcoef(residuals[left], residuals[right])[0, 1]),
                    }
                )

        run_audit.append(
            {
                "seed": seed,
                "run_instance_id": manifest["run_instance_id"],
                "run_hash": manifest["run_hash"],
                "fit_id": fit_ids[0],
                "completed_conditions": completed_count,
                "unique_fit_ids_across_snr": len(set(fit_ids)),
                "conformer_epoch": selected_epochs["conformer"],
                "alexnet_epoch": selected_epochs["alexnet"],
                "weight_randomforest": weights["randomforest"],
                "weight_conformer": weights["conformer"],
                "weight_alexnet": weights["alexnet"],
                "max_reload_prediction_difference_w_m2": max(verification_diffs),
            }
        )

    audit = pd.DataFrame(run_audit).sort_values("seed")
    metrics = pd.DataFrame(metric_rows)
    regions = pd.DataFrame(region_rows)
    correlations = pd.DataFrame(correlation_rows)

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

    delta_rows = []
    region_index = regions.set_index(["seed", "snr", "model_key", "region"])
    metric_index = metrics.set_index(["seed", "snr", "model_key"])
    for seed in sorted(metrics["seed"].unique()):
        for snr_label, _, _ in SNR_PLAN:
            singles = metrics[
                (metrics["seed"] == seed)
                & (metrics["snr"] == snr_label)
                & metrics["model_key"].isin(["randomforest", "conformer", "alexnet"])
            ]
            best_single_key = singles.sort_values("rmse_all_mean").iloc[0]["model_key"]
            for ensemble_key in ["ensemble__simple_equal", "ensemble__inner_holdout"]:
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
                    "delta_rmse_vs_best_single": ensemble["rmse_all_mean"] - best["rmse_all_mean"],
                    "delta_onb_rmse_vs_rf": ensemble["rmse_onb_mean"] - rf["rmse_onb_mean"],
                }
                for region in ["below", "near", "above"]:
                    row[f"delta_{region}_rmse_vs_rf"] = (
                        region_index.loc[(seed, snr_label, ensemble_key, region), "rmse"]
                        - region_index.loc[(seed, snr_label, "randomforest", region), "rmse"]
                    )
                delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    # Evaluation-day oracle diagnostic only. These weights must not be adopted
    # as learned weights because they use the held-out 6/18 predictions.
    weight_grid = []
    for rf_percent in range(101):
        for conformer_percent in range(101 - rf_percent):
            alexnet_percent = 100 - rf_percent - conformer_percent
            weight_grid.append(
                [rf_percent / 100, conformer_percent / 100, alexnet_percent / 100]
            )
    weight_grid = np.asarray(weight_grid)
    grid_rmse = []
    grid_excess = []
    grid_snr = []
    for item in residual_second_moments:
        mse_values = np.einsum("gi,ij,gj->g", weight_grid, item["matrix"], weight_grid)
        condition_rmse = np.sqrt(np.maximum(mse_values, 0))
        grid_rmse.append(condition_rmse)
        grid_excess.append(condition_rmse - item["rf_rmse"])
        grid_snr.append(item["snr"])
    grid_rmse = np.column_stack(grid_rmse)
    grid_excess = np.column_stack(grid_excess)
    grid_snr = np.asarray(grid_snr)
    grid_diagnostic = pd.DataFrame(
        {
            "weight_randomforest": weight_grid[:, 0],
            "weight_conformer": weight_grid[:, 1],
            "weight_alexnet": weight_grid[:, 2],
            "mean_rmse_all_conditions": grid_rmse.mean(axis=1),
            "mean_rmse_clean": grid_rmse[:, grid_snr == "clean"].mean(axis=1),
            "mean_rmse_noisy": grid_rmse[:, grid_snr != "clean"].mean(axis=1),
            "max_rmse_excess_vs_rf": grid_excess.max(axis=1),
            "conditions_beating_rf": (grid_excess < -1e-9).sum(axis=1),
            "conditions_not_worse_than_rf": (grid_excess <= 1e-9).sum(axis=1),
        }
    )
    oracle_rows = []
    for snr_label, _, _ in SNR_PLAN:
        mask = grid_snr == snr_label
        mean_mse = np.square(grid_rmse[:, mask]).mean(axis=1)
        best_index = int(np.argmin(mean_mse))
        oracle_rows.append(
            {
                "snr": snr_label,
                "weight_randomforest": weight_grid[best_index, 0],
                "weight_conformer": weight_grid[best_index, 1],
                "weight_alexnet": weight_grid[best_index, 2],
                "rmse_from_mean_mse": float(np.sqrt(mean_mse[best_index])),
            }
        )
    oracle = pd.DataFrame(oracle_rows)

    audit.to_csv(OUTPUT_DIR / "run_audit.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_seed.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "metrics_seed_aggregate.csv", index=False)
    regions.to_csv(OUTPUT_DIR / "region_metrics_by_seed.csv", index=False)
    deltas.to_csv(OUTPUT_DIR / "ensemble_deltas_by_seed.csv", index=False)
    correlations.to_csv(OUTPUT_DIR / "residual_correlations_by_seed.csv", index=False)
    grid_diagnostic.to_csv(OUTPUT_DIR / "evaluation_day_weight_grid_diagnostic.csv", index=False)
    oracle.to_csv(OUTPUT_DIR / "evaluation_day_oracle_weights_by_snr.csv", index=False)

    print("RUN AUDIT")
    print(audit.to_string(index=False))
    print("\nR2 MEAN ACROSS SEEDS")
    print(metrics.pivot_table(index="snr", columns="model", values="r2_mean", aggfunc="mean").reindex([x[0] for x in SNR_PLAN]).to_string())
    print("\nRMSE MEAN ACROSS SEEDS (kW/m2)")
    rmse_table = metrics.pivot_table(index="snr", columns="model", values="rmse_all_mean", aggfunc="mean") / 1000
    print(rmse_table.reindex([x[0] for x in SNR_PLAN]).to_string())
    print("\nENSEMBLE DELTAS VS RF: MEAN ACROSS SEEDS (kW/m2 except R2)")
    delta_summary = deltas.groupby(["snr", "ensemble"], sort=False).mean(numeric_only=True).reset_index()
    for column in [name for name in delta_summary if "rmse" in name]:
        delta_summary[column] /= 1000
    print(delta_summary.to_string(index=False))
    print("\nRESIDUAL CORRELATIONS")
    print(correlations.groupby(["snr", "model_left", "model_right"], sort=False)["residual_correlation"].mean().reset_index().to_string(index=False))
    print("\nEVALUATION-DAY WEIGHT DIAGNOSTIC (NOT FOR ADOPTION)")
    for objective in ["mean_rmse_all_conditions", "mean_rmse_noisy", "max_rmse_excess_vs_rf"]:
        best = grid_diagnostic.sort_values([objective, "mean_rmse_all_conditions"]).iloc[0]
        print(objective, best.to_dict())
    print("weights not worse than RF in all 21 seed/conditions:", int((grid_diagnostic["conditions_not_worse_than_rf"] == 21).sum()))
    print("\nORACLE WEIGHTS BY SNR (NOT FOR ADOPTION)")
    print(oracle.to_string(index=False))


if __name__ == "__main__":
    main()
