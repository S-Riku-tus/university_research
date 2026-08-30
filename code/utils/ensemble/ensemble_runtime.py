import csv
import gc
import os
from copy import deepcopy

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from utils.ensemble.ensemble_weighting import EnsembleWeighting
from utils.ensemble.strategy_catalog import resolve_ensemble_selection
from utils.ensemble.strategy_comparison import (
    aggregate_correction_rows,
    compute_strategy_outputs,
    correction_diagnostic_row,
    metric_comparison_rows,
    normalize_strategy_plan,
    pairwise_diversity_rows,
    strategy_plan_requires_inner_holdout,
)
from utils.experiment.run_helpers import open_text, path_exists
from utils.models.regression.base_regression import RegressionModelMaker


COMPARISON_HEADER = [
    "strategy_name", "strategy_type", "claim_safe",
    "metric", "better_direction", "ensemble_mean",
    "ensemble_se", "best_single_model", "best_single_mean",
    "best_single_se", "improvement_vs_best_single",
    "ensemble_better_than_best_single", "paired_improvement_se",
    "folds_improved", "folds_total",
]


class EnsembleManager:
    """Resolve reusable ensemble definitions and create per-run executors."""

    def __init__(self, selection_config, configured_model_keys, random_seed=42):
        self.selection_config = deepcopy(selection_config or {})
        self.configured_model_keys = list(configured_model_keys)
        self.random_seed = int(random_seed)
        self.resolved_config = resolve_ensemble_selection(self.selection_config)
        if (
            self.configured_model_keys
            and self.resolved_config["reference_model"]
            not in self.configured_model_keys
        ):
            self.resolved_config["reference_model"] = self.configured_model_keys[0]
        self.strategy_plan = (
            normalize_strategy_plan(self.resolved_config)
            if self.resolved_config["enabled"] else []
        )
        self.primary_strategy_name = self.resolved_config.get("primary_strategy")
        self.primary_result_key = next(
            (
                item["result_key"]
                for item in self.strategy_plan
                if item["name"] == self.primary_strategy_name
            ),
            None,
        )

    @property
    def enabled(self):
        return bool(self.strategy_plan)

    @property
    def reference_model(self):
        return self.resolved_config["reference_model"]

    @property
    def combine(self):
        return self.resolved_config["combine"]

    @property
    def inner_holdout_frac(self):
        return float(self.resolved_config["inner_holdout_frac"])

    @property
    def has_leaky_strategy(self):
        return any(not item["claim_safe"] for item in self.strategy_plan)

    @property
    def selected_strategy_names(self):
        return [item["name"] for item in self.strategy_plan]

    def validate(self, enabled_specs):
        if not self.enabled:
            return
        model_keys = [spec["key"] for spec in enabled_specs]
        if len(model_keys) < 2:
            return
        if self.primary_result_key is None:
            raise ValueError(
                "ensemble.primary_strategy_name must name one enabled strategy; got "
                f"{self.primary_strategy_name!r}."
            )
        if self.reference_model not in model_keys:
            raise ValueError(
                "The ensemble catalog reference model must be active; got "
                f"{self.reference_model!r}."
            )
        if self.combine not in {"mean", "min", "max"}:
            raise ValueError(f"Unknown ensemble combine method: {self.combine}")
        if strategy_plan_requires_inner_holdout(self.strategy_plan) and not (
                0 < self.inner_holdout_frac < 1):
            raise ValueError("Ensemble inner_holdout_frac must be between 0 and 1.")

    def snapshot(self):
        return {
            "selection": deepcopy(self.selection_config),
            **deepcopy(self.resolved_config),
            "resolved_strategy_plan": deepcopy(self.strategy_plan),
        }

    def description(self):
        return (
            f"strategies={self.selected_strategy_names} | "
            f"primary={self.primary_strategy_name} | combine={self.combine}"
        )

    def create_run(self, run_specs):
        return EnsembleRun(self, run_specs)


class EnsembleRun:
    """Execute and report ensemble mechanics for one resolved model run."""

    def __init__(self, manager, run_specs):
        self.manager = manager
        self.run_specs = list(run_specs)
        self.model_keys = [spec["key"] for spec in self.run_specs]
        self.enabled = bool(
            manager.enabled
            and len(self.run_specs) >= 2
            and manager.reference_model in self.model_keys
        )
        self.weighting = EnsembleWeighting()
        self.legacy_errors = {}
        self.weight_log = []
        self.correction_log = []
        self.diversity_log = []

    @property
    def strategy_plan(self):
        return self.manager.strategy_plan if self.enabled else []

    @property
    def result_keys(self):
        return [item["result_key"] for item in self.strategy_plan]

    @property
    def labels(self):
        return {item["result_key"]: item["label"] for item in self.strategy_plan}

    @property
    def primary_result_key(self):
        return self.manager.primary_result_key if self.enabled else None

    @property
    def strategy_tag(self):
        if len(self.strategy_plan) > 1:
            return "strategy_loop"
        if self.strategy_plan:
            return self.strategy_plan[0]["strategy"]
        return "simple"

    @property
    def needs_legacy_errors(self):
        return any(
            item["strategy"] == "val_fold_legacy"
            for item in self.strategy_plan
        )

    def description(self):
        if not self.enabled:
            return "ensemble_strategies=[] (single-model tuning run)"
        return (
            f"ensemble_strategies={self.manager.selected_strategy_names}, "
            f"primary={self.manager.primary_strategy_name}, "
            f"combine={self.manager.combine}"
        )

    def fit_inner_holdout_errors(
        self,
        trainer,
        x_train,
        y_train,
        pca_components,
        input_shape,
        epochs,
        fold,
        total_folds,
    ):
        """Fit temporary inner models only when the selected strategy needs them."""
        if not strategy_plan_requires_inner_holdout(self.strategy_plan):
            return {}

        x_inner_fit, x_inner, y_inner_fit, y_inner = train_test_split(
            x_train,
            y_train,
            test_size=self.manager.inner_holdout_frac,
            random_state=self.manager.random_seed + int(fold),
        )
        inner_scaler = MinMaxScaler()
        y_inner_fit_scaled = inner_scaler.fit_transform(
            y_inner_fit.reshape(-1, 1)
        )
        use_sklearn = any(spec["kind"] == "sklearn" for spec in self.run_specs)
        if use_sklearn:
            x_inner_fit_pca, (x_inner_pca,) = trainer.make_pca(
                x_inner_fit,
                [x_inner],
                pca_components,
            )
        else:
            x_inner_fit_pca = x_inner_pca = None

        model_maker = RegressionModelMaker(input_shape)
        errors = {}
        for spec in self.run_specs:
            print(
                f"[{spec['label']}] Fold {fold}/{total_folds} "
                "inner-holdout weight fit"
            )
            inner_spec = dict(spec)
            inner_spec["fit_verbose"] = 0
            inner_model, inner_history = trainer.train_one_model(
                inner_spec,
                model_maker,
                x_inner_fit,
                y_inner_fit_scaled,
                x_inner_fit_pca,
                epochs,
            )
            inner_pred = trainer.predict_one_model(
                inner_spec,
                inner_model,
                x_inner,
                x_inner_pca,
                inner_scaler,
            )
            errors[spec["key"]] = 1.0 - r2_score(y_inner, inner_pred)
            del inner_model, inner_history, inner_pred
            K.clear_session()
            gc.collect()
        return errors

    def record_validation_error(self, model_key, y_true, prediction):
        """Record outer-fold errors only for the explicitly selected legacy mode."""
        if self.needs_legacy_errors:
            self.legacy_errors[model_key] = 1.0 - r2_score(y_true, prediction)

    def combine_predictions(self, val_preds, inner_errors, fold):
        if not self.enabled:
            return {}
        outputs = compute_strategy_outputs(
            self.weighting,
            self.strategy_plan,
            self.run_specs,
            val_preds,
            self.manager.combine,
            inner_errors=inner_errors,
            legacy_errors=self.legacy_errors,
        )
        for output in outputs.values():
            item = output["strategy"]
            self.weight_log.append([
                fold,
                item["name"],
                item["strategy"],
                int(item["claim_safe"]),
                *[
                    output["weights"].get(key, 0.0)
                    for key in self.model_keys
                ],
            ])
        return outputs

    def merge_predictions(self, val_preds, ensemble_outputs):
        predictions = dict(val_preds)
        predictions.update({
            result_key: output["prediction"]
            for result_key, output in ensemble_outputs.items()
        })
        return predictions

    def record_diagnostics(
        self,
        fold,
        y_true,
        val_preds,
        ensemble_outputs,
        threshold,
        onb_band_frac,
    ):
        if not self.enabled:
            return
        for row in pairwise_diversity_rows(
            y_true,
            val_preds,
            threshold=threshold,
            onb_band_frac=onb_band_frac,
        ):
            self.diversity_log.append([fold, *row])
        reference_prediction = val_preds[self.manager.reference_model]
        for output in ensemble_outputs.values():
            self.correction_log.append([
                fold,
                *correction_diagnostic_row(
                    output["strategy"],
                    y_true,
                    self.manager.reference_model,
                    reference_prediction,
                    output["prediction"],
                    threshold=threshold,
                    onb_band_frac=onb_band_frac,
                ),
            ])

    def write_fold_weights(self, output_file, ensemble_outputs):
        for output in ensemble_outputs.values():
            output_file.write(
                f"  weights[{output['strategy']['name']}]="
                f"{ {key: round(value, 4) for key, value in output['weights'].items()} }\n"
            )

    def save_reports(
        self,
        save_path,
        base_save_path,
        snr_value,
        store,
        metrics,
        summary_metrics,
        plotter,
        run_instance_id,
        run_hash,
        run_dir,
        job,
    ):
        if not self.enabled:
            return None

        weights_csv = os.path.join(save_path, f"ensemble_weights_{snr_value}.csv")
        with open_text(weights_csv, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "fold", "strategy_name", "strategy_type", "claim_safe",
                *self.model_keys,
            ])
            writer.writerows(self.weight_log)

        comparison_rows = metric_comparison_rows(
            store,
            self.model_keys,
            self.strategy_plan,
            metrics.mean_se,
            summary_metrics,
        )
        comparison_path = os.path.join(
            save_path,
            f"ensemble_strategy_comparison_{snr_value}.csv",
        )
        with open_text(
                comparison_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(COMPARISON_HEADER)
            writer.writerows(comparison_rows)
        plotter.plot_ensemble_strategy_improvements(
            comparison_rows,
            save_path,
            snr_value,
        )

        correction_path = os.path.join(
            save_path,
            f"ensemble_correction_diagnostics_{snr_value}.csv",
        )
        with open_text(
                correction_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "fold", "strategy_name", "strategy_type", "claim_safe",
                "reference_model", "n_samples", "n_better", "n_worse",
                "n_tied", "mean_abs_error_delta", "n_onb",
                "mean_onb_abs_error_delta", "reference_false_negatives",
                "ensemble_false_negatives", "false_negative_delta",
                "recovered_false_negatives", "new_false_negatives",
            ])
            writer.writerows(self.correction_log)

        correction_summary_rows = aggregate_correction_rows(self.correction_log)
        correction_summary_path = os.path.join(
            save_path,
            f"ensemble_correction_summary_{snr_value}.csv",
        )
        with open_text(
                correction_summary_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "strategy_name", "strategy_type", "claim_safe",
                "reference_model", "folds", "n_samples",
                "n_better", "n_worse", "n_tied",
                "fraction_better", "fraction_worse",
                "mean_abs_error_delta", "n_onb",
                "mean_onb_abs_error_delta", "reference_false_negatives",
                "ensemble_false_negatives", "false_negative_delta",
                "recovered_false_negatives", "new_false_negatives",
            ])
            writer.writerows(correction_summary_rows)

        diversity_path = os.path.join(
            save_path,
            f"model_diversity_{snr_value}.csv",
        )
        with open_text(
                diversity_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "fold", "model_left", "model_right", "n_samples",
                "prediction_pearson", "residual_pearson",
                "left_lower_abs_error", "right_lower_abs_error", "ties",
                "n_onb", "onb_residual_pearson",
            ])
            writer.writerows(self.diversity_log)

        presentation_summary_path = os.path.join(
            base_save_path,
            "ensemble_presentation_summary.csv",
        )
        presentation_exists = path_exists(presentation_summary_path)
        with open_text(
                presentation_summary_path, "a", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            if not presentation_exists:
                writer.writerow([
                    "run_instance_id", "run_hash", "run_dir",
                    "experiment_name", "max_freq_hz", "noise_dir_name",
                    *COMPARISON_HEADER,
                ])
            for row in comparison_rows:
                writer.writerow([
                    run_instance_id,
                    run_hash,
                    run_dir,
                    job["experiment_name"],
                    job["max_freq_hz"],
                    job["noise_dir_name"],
                    *row,
                ])
        print(f"ensemble comparison saved: {comparison_path}")
        return comparison_path
