import csv
import gc
import json
import math
import os
from copy import deepcopy

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from utils.ensemble.ensemble_weighting import EnsembleWeighting
from utils.ensemble.crossfit_stacking import (
    CROSSFIT_STRATEGIES, fit_crossfit_strategy, group_crossfit_splits,
)
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
from utils.experiment.run_helpers import open_text, path_exists, set_global_seed
from utils.models.regression.base_regression import RegressionModelMaker


COMPARISON_HEADER = [
    "strategy_name", "strategy_type", "claim_safe",
    "metric", "better_direction", "ensemble_mean",
    "ensemble_se", "best_single_model", "best_single_mean",
    "best_single_se", "improvement_vs_best_single",
    "ensemble_better_than_best_single", "paired_improvement_se",
    "folds_improved", "folds_total",
]


def _group_disjoint_holdout_indices(y_true, groups, fraction, random_state):
    """Split inner-fit/holdout indices without sharing a source WAV."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if len(groups) != len(y_true):
        raise ValueError("groups and y_true must have the same length.")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("inner_holdout requires at least three source WAV groups.")
    holdout_group_count = min(
        len(unique_groups) - 1,
        max(2, int(math.ceil(len(unique_groups) * float(fraction)))),
    )
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_group_count,
        random_state=int(random_state),
    )
    fit_index, holdout_index = next(
        splitter.split(np.zeros(len(y_true)), y_true, groups=groups)
    )
    overlap = set(groups[fit_index]) & set(groups[holdout_index])
    if overlap:
        raise RuntimeError(f"Inner source-WAV leakage detected: {sorted(overlap)}")
    return fit_index, holdout_index


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
        if self.reference_model not in model_keys:
            raise ValueError(
                "The ensemble catalog reference model must be active; got "
                f"{self.reference_model!r}."
            )
        if self.combine not in {"mean", "min"}:
            raise ValueError(f"Unknown ensemble combine method: {self.combine}")
        if strategy_plan_requires_inner_holdout(self.strategy_plan) and not (
                0 < self.inner_holdout_frac < 1):
            raise ValueError("Ensemble inner_holdout_frac must be between 0 and 1.")
        crossfit = [item for item in self.strategy_plan if item["strategy"] in CROSSFIT_STRATEGIES]
        if crossfit:
            if self.combine != "mean":
                raise ValueError("Crossfit strategies require combine='mean'.")
            if len({item["crossfit"]["inner_folds"] for item in crossfit}) != 1:
                raise ValueError("Crossfit strategies must share inner_folds for OOF reuse.")
            if any(len(model_keys) > item["crossfit"]["max_models"] for item in crossfit):
                raise ValueError("Too many models for exhaustive crossfit subsets.")

    def snapshot(self):
        return {
            "selection": deepcopy(self.selection_config),
            **deepcopy(self.resolved_config),
            "resolved_strategy_plan": deepcopy(self.strategy_plan),
        }

    def description(self):
        return (
            f"strategies={self.selected_strategy_names} | "
            f"combine={self.combine}"
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
            f"combine={self.manager.combine}"
        )

    def fit_inner_holdout_errors(
        self,
        trainer,
        x_train,
        y_train,
        groups,
        pca_components,
        input_shape,
        epochs,
        fold,
        total_folds,
    ):
        """Fit temporary inner models only when the selected strategy needs them."""
        if any(item["name"] == "performance_kfold" for item in self.strategy_plan):
            raise ValueError("performance_kfold must use the training-only internal KFold runner")
        if not strategy_plan_requires_inner_holdout(self.strategy_plan):
            return {}

        if groups is None:
            raise ValueError(
                "inner_holdout requires source_wav_id groups; sample-level "
                "weight fitting would leak the same WAV across partitions."
            )
        groups = np.asarray(groups).ravel()
        inner_fit_index, inner_index = _group_disjoint_holdout_indices(
            y_train,
            groups,
            fraction=self.manager.inner_holdout_frac,
            random_state=self.manager.random_seed + int(fold),
        )
        x_inner_fit, x_inner = x_train[inner_fit_index], x_train[inner_index]
        y_inner_fit, y_inner = y_train[inner_fit_index], y_train[inner_index]
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
                epochs[spec["key"]] if isinstance(epochs, dict) else epochs,
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

    def fit_crossfit_weights(
        self, trainer, x_train, y_train, groups, pca_components,
        input_shape, epochs, fold, total_folds,
    ):
        """One shared inner OOF pass for all selected crossfit strategies.

        This API deliberately accepts no outer evaluation data. In clean_only,
        the caller reuses the returned bundle across every evaluation noise.
        """
        plan = [item for item in self.strategy_plan if item["strategy"] in CROSSFIT_STRATEGIES]
        if not plan:
            return None
        if len({item["crossfit"]["inner_folds"] for item in plan}) != 1:
            raise ValueError("Crossfit strategies must share inner_folds.")
        splits = group_crossfit_splits(groups, plan[0]["crossfit"]["inner_folds"],
                                      self.manager.random_seed + int(fold))
        groups = np.asarray(groups).ravel()
        y_train = np.asarray(y_train, dtype=float).ravel()
        if not (len(groups) == len(y_train) == len(x_train)):
            raise ValueError("Crossfit inputs, targets and groups must have equal lengths.")
        oof = {key: np.full(len(y_train), np.nan) for key in self.model_keys}
        coverage = np.zeros(len(y_train), dtype=int)
        inner_fold_ids = np.zeros(len(y_train), dtype=int)
        split_records = []
        use_pca = any(spec["kind"] == "sklearn" for spec in self.run_specs)
        for inner_fold, (fit_index, held_index) in enumerate(splits, 1):
            seed = self.manager.random_seed + 10000 * int(fold) + inner_fold
            set_global_seed(seed)
            x_fit, x_held = x_train[fit_index], x_train[held_index]
            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(y_train[fit_index].reshape(-1, 1))
            x_fit_pca = x_held_pca = None
            if use_pca:
                x_fit_pca, (x_held_pca,) = trainer.make_pca(x_fit, [x_held], pca_components)
            model_training = {}
            for spec in self.run_specs:
                set_global_seed(seed)
                print(f"[{spec['label']}] Fold {fold}/{total_folds} inner crossfit {inner_fold}/{len(splits)}")
                inner_spec = {**spec, "fit_verbose": 0}
                model = history = None
                try:
                    model, history = trainer.train_one_model(
                        inner_spec, RegressionModelMaker(input_shape), x_fit, y_scaled, x_fit_pca,
                        epochs[spec["key"]] if isinstance(epochs, dict) else epochs)
                    prediction = np.asarray(trainer.predict_one_model(
                        inner_spec, model, x_held, x_held_pca, scaler), dtype=float).ravel()
                    if len(prediction) != len(held_index) or not np.isfinite(prediction).all():
                        raise ValueError("Invalid inner OOF prediction shape or nonfinite values.")
                    oof[spec["key"]][held_index] = prediction
                    params = history.params if history is not None else {}
                    model_training[spec["key"]] = {key: params.get(key) for key in (
                        "epochs_completed", "actual_batch_size", "requested_batch_size", "stopped_by_memory_error")}
                finally:
                    del model, history
                    K.clear_session()
                    gc.collect()
            coverage[held_index] += 1
            inner_fold_ids[held_index] = inner_fold
            split_records.append({
                "inner_fold": inner_fold, "random_seed": seed,
                "training_wav_groups": sorted(set(groups[fit_index].tolist())),
                "heldout_wav_groups": sorted(set(groups[held_index].tolist())),
                "n_training_chunks": len(fit_index), "n_heldout_chunks": len(held_index),
                "model_training": model_training,
            })
            del x_fit, x_held, x_fit_pca, x_held_pca, y_scaled
        if not np.all(coverage == 1):
            raise RuntimeError("Every crossfit training chunk must have exactly one held-out prediction.")
        weights, diagnostics = {}, {}
        for item in plan:
            weights[item["name"]], diagnostics[item["name"]] = fit_crossfit_strategy(
                y_train, oof, groups, self.model_keys, item["strategy"], item["crossfit"])
        return {"fold": int(fold), "weights": weights, "diagnostics": diagnostics,
                "splits": split_records, "oof_predictions": oof, "targets": y_train,
                "groups": groups, "inner_fold_ids": inner_fold_ids}

    def save_crossfit_fit(self, save_path, fold, fit):
        if fit is None:
            return
        if fit["fold"] != fold:
            raise ValueError("Crossfit fit belongs to a different outer fold.")
        audit = {key: fit[key] for key in ("fold", "weights", "diagnostics", "splits")}
        with open_text(os.path.join(save_path, f"ensemble_crossfit_fit_f{fold}.json"),
                       "w", encoding="utf-8") as output:
            json.dump(audit, output, ensure_ascii=False, indent=2, allow_nan=False)
        with open_text(os.path.join(save_path, f"ensemble_inner_oof_f{fold}.csv"),
                       "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["training_row_index", "source_wav_group", "inner_fold", "y_true", *self.model_keys])
            for index, target in enumerate(fit["targets"]):
                writer.writerow([index, fit["groups"][index], fit["inner_fold_ids"][index], target,
                                 *[fit["oof_predictions"][key][index] for key in self.model_keys]])

    def combine_predictions(self, val_preds, inner_errors, fold, crossfit_fit=None):
        if not self.enabled:
            return {}
        if crossfit_fit is not None and crossfit_fit["fold"] != fold:
            raise ValueError("Crossfit fit belongs to a different outer fold.")
        outputs = compute_strategy_outputs(
            self.weighting,
            self.strategy_plan,
            self.run_specs,
            val_preds,
            self.manager.combine,
            inner_errors=inner_errors,
            legacy_errors=self.legacy_errors,
            fitted_weights=crossfit_fit["weights"] if crossfit_fit is not None else None,
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
