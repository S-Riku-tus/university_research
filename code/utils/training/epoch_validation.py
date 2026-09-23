"""Training-day-only epoch validation with source-WAV-disjoint folds."""

import csv
import gc
import json

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

from utils.experiment.learning_policy import wav_groups
from utils.experiment.run_helpers import json_default, open_text, set_global_seed
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.internal_validation import internal_splits


def checkpoint_epochs(total_epochs, interval):
    total_epochs = int(total_epochs)
    interval = int(interval)
    if total_epochs < 1 or interval < 1:
        raise ValueError("Epoch total and checkpoint interval must be positive.")
    return sorted({1, total_epochs, *range(interval, total_epochs + 1, interval)})


def _threshold_array(metadata, thresholds_by_experiment):
    values = []
    for row in metadata:
        experiment = row["experiment_name"]
        value = thresholds_by_experiment.get(experiment)
        if value is None or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(
                f"Training validation needs a positive ONB threshold for {experiment}."
            )
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _masked_rmse(residual, mask):
    return float(np.sqrt(np.mean(np.square(residual[mask])))) if np.any(mask) else None


def _masked_mean(residual, mask):
    return float(np.mean(residual[mask])) if np.any(mask) else None


def validation_metrics(y_true, prediction, thresholds, onb_band_frac):
    y_true = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    if not (len(y_true) == len(prediction) == len(thresholds)):
        raise ValueError("Validation targets, predictions, and thresholds must align.")
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        raise ValueError("Validation targets and predictions must be finite.")
    residual = prediction - y_true
    near = np.abs(y_true - thresholds) <= float(onb_band_frac) * thresholds
    below = (y_true < thresholds) & ~near
    above = (y_true >= thresholds) & ~near
    positive = y_true >= thresholds
    predicted_positive = prediction >= thresholds
    sse = float(np.sum(np.square(residual)))
    return {
        "r2": float(r2_score(y_true, prediction)) if np.var(y_true) > 0 else None,
        "rmse_all": float(np.sqrt(sse / len(y_true))),
        "mae_all": float(np.mean(np.abs(residual))),
        "mean_residual_all": float(np.mean(residual)),
        "rmse_below_onb": _masked_rmse(residual, below),
        "mean_residual_below_onb": _masked_mean(residual, below),
        "rmse_near_onb": _masked_rmse(residual, near),
        "mean_residual_near_onb": _masked_mean(residual, near),
        "rmse_above_onb": _masked_rmse(residual, above),
        "mean_residual_above_onb": _masked_mean(residual, above),
        "below_chunks": int(np.sum(below)),
        "near_chunks": int(np.sum(near)),
        "above_chunks": int(np.sum(above)),
        "misses": int(np.sum(positive & ~predicted_positive)),
        "false_alarms": int(np.sum(~positive & predicted_positive)),
    }


def fit_epoch_validation_cv(
    trainer,
    specs,
    x,
    y,
    metadata,
    selector,
    folds,
    seed,
    mode,
    pca_components,
    epochs,
    config,
    thresholds_by_experiment,
    onb_band_frac,
):
    """Return model-specific epochs chosen without using evaluation-day data."""
    if mode != "wav_kfold":
        raise ValueError("Training epoch validation must use mode='wav_kfold'.")
    folds = int(config.get("folds", folds))
    checkpoints = checkpoint_epochs(epochs, config.get("checkpoint_interval_epochs", 10))
    minimum_epoch = int(config.get("minimum_epoch", 1))
    metric_name = str(config.get("selection_metric", "rmse_all"))
    if metric_name != "rmse_all":
        raise ValueError("Only rmse_all is currently supported for epoch selection.")
    if minimum_epoch < 1 or minimum_epoch > int(epochs):
        raise ValueError("training_validation.minimum_epoch must be within the run epochs.")

    y = np.asarray(y, dtype=float).ravel()
    thresholds = _threshold_array(metadata, thresholds_by_experiment)
    groups = wav_groups(metadata)
    splits = internal_splits(metadata, folds, seed, mode)
    predictions = {spec["key"]: {} for spec in specs}
    for spec in specs:
        epochs_for_model = [0] if spec["kind"] != "keras" else checkpoints
        predictions[spec["key"]] = {
            epoch: np.full(len(y), np.nan, dtype=float)
            for epoch in epochs_for_model
        }

    split_records = []
    for fold, (fit_index, held_index) in enumerate(splits, 1):
        retained, selection = selector.select([metadata[i] for i in fit_index])
        selected_fit = fit_index[retained]
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y[selected_fit].reshape(-1, 1))
        x_fit, x_held = x[selected_fit], x[held_index]
        x_fit_pca = x_held_pca = None
        if any(spec["kind"] != "keras" for spec in specs):
            x_fit_pca, (x_held_pca,) = trainer.make_pca(
                x_fit, [x_held], pca_components
            )
        model_training = {}
        for spec in specs:
            set_global_seed(seed + fold)
            inner_spec = {**spec, "fit_verbose": 0}
            model = history = None
            try:
                if spec["kind"] == "keras":
                    model, history, held_predictions = (
                        trainer.train_one_model_with_epoch_validation(
                            inner_spec,
                            RegressionModelMaker(tuple(x.shape[1:])),
                            x_fit,
                            y_scaled,
                            x_fit_pca,
                            epochs,
                            x_held,
                            x_held_pca,
                            y[held_index],
                            scaler,
                            checkpoints,
                        )
                    )
                    for epoch, prediction in held_predictions.items():
                        predictions[spec["key"]][int(epoch)][held_index] = prediction
                else:
                    model, history = trainer.train_one_model(
                        inner_spec,
                        RegressionModelMaker(tuple(x.shape[1:])),
                        x_fit,
                        y_scaled,
                        x_fit_pca,
                        epochs,
                    )
                    predictions[spec["key"]][0][held_index] = trainer.predict_one_model(
                        inner_spec, model, x_held, x_held_pca, scaler
                    )
                params = history.params if history is not None else {}
                model_training[spec["key"]] = {
                    key: params.get(key)
                    for key in (
                        "epochs_completed",
                        "actual_batch_size",
                        "requested_batch_size",
                        "stopped_by_memory_error",
                    )
                }
            finally:
                del model, history
                K.clear_session()
                gc.collect()
        split_records.append(
            {
                "fold": fold,
                "random_seed": seed + fold,
                "training_wav_groups": sorted(set(groups[selected_fit].tolist())),
                "heldout_wav_groups": sorted(set(groups[held_index].tolist())),
                "shared_source_wavs": len(
                    set(groups[selected_fit]) & set(groups[held_index])
                ),
                "n_training_chunks_before_selection": int(len(fit_index)),
                "n_training_chunks_after_selection": int(len(selected_fit)),
                "n_heldout_chunks": int(len(held_index)),
                "selection": selection,
                "model_training": model_training,
            }
        )

    rows = []
    for spec in specs:
        for epoch, prediction in predictions[spec["key"]].items():
            if not np.isfinite(prediction).all():
                continue
            metrics = validation_metrics(y, prediction, thresholds, onb_band_frac)
            rows.append(
                {
                    "model": spec["key"],
                    "epoch": int(epoch),
                    "chunks": int(len(y)),
                    "source_wavs": int(len(set(groups.tolist()))),
                    "folds": int(folds),
                    **metrics,
                }
            )

    selected_epochs = {}
    if config.get("select_epochs", True):
        for spec in specs:
            if spec["kind"] != "keras":
                continue
            candidates = [
                row for row in rows
                if row["model"] == spec["key"] and row["epoch"] >= minimum_epoch
            ]
            if not candidates:
                raise RuntimeError(
                    f"No fully covered epoch checkpoint for {spec['key']}."
                )
            chosen = min(candidates, key=lambda row: (row[metric_name], row["epoch"]))
            selected_epochs[spec["key"]] = int(chosen["epoch"])

    audit = {
        "method": mode,
        "scope": "training_days_only",
        "test_used": False,
        "folds": folds,
        "checkpoint_epochs": checkpoints,
        "minimum_epoch": minimum_epoch,
        "selection_metric": metric_name,
        "select_epochs": bool(config.get("select_epochs", True)),
        "selected_epochs": selected_epochs,
        "splits": split_records,
        "rows": rows,
    }
    return selected_epochs, rows, audit


def save_epoch_validation_outputs(path, outer_fold, rows, audit):
    csv_path = path / f"training_validation_curve_f{outer_fold}.csv"
    json_path = path / f"training_validation_f{outer_fold}.json"
    if rows:
        with open_text(csv_path, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open_text(json_path, "w", encoding="utf-8") as output:
        json.dump(audit, output, ensure_ascii=False, indent=2, default=json_default)


def model_epochs(epochs, selected_epochs, spec):
    value = selected_epochs.get(spec["key"], epochs)
    value = int(value)
    if value < 1:
        raise ValueError(f"Invalid epoch count for {spec['key']}: {value}")
    return value
