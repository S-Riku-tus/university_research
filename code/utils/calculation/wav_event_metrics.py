"""WAV-level OOF aggregation and predicted-event diagnostics.

The heat-flux target belongs to a source recording/operating condition, while
the model consumes one-second chunks.  This module keeps the chunk predictions
but adds one evaluation unit per ``source_wav_id``.  This avoids treating many
chunks cut from the same recording as independent evidence; WAVs collected on
the same experiment day can still be correlated.

"Event" outputs in this module are deliberately named *predicted* events.  A
threshold crossing of the predicted heat flux is not ground-truth evidence of
a bubble event.  Event precision/recall must wait for synchronized audio/video
annotations.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics


PREDICTION_METADATA_COLUMNS = [
    "fold",
    "sample_index",
    "sample_filename",
    "experiment_name",
    "source_wav_id",
    "source_wav_name",
    "chunk_index",
    "chunk_start_seconds",
    "chunk_duration_seconds",
    "y_true",
]

SUPPORTED_AGGREGATIONS = {
    "mean": lambda values: float(np.mean(values)),
    "median": lambda values: float(np.median(values)),
    "p90": lambda values: float(np.quantile(values, 0.90)),
    "p95": lambda values: float(np.quantile(values, 0.95)),
}


def _has_threshold(threshold):
    try:
        return threshold is not None and np.isfinite(float(threshold))
    except (TypeError, ValueError):
        return False


def _to_float(value, default=np.nan):
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        # Preserve enough precision for exact ONB-boundary labels.  The old
        # 10-significant-digit format could round a true onset value just
        # below its registered threshold.
        return f"{float(value):.17g}"
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _long_path(path):
    path = os.path.abspath(os.fspath(path))
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path
    return path


def _open_text(path, mode="r", **kwargs):
    """Open a path while tolerating long paths on Windows."""
    return open(_long_path(path), mode, **kwargs)


def _makedirs(path):
    os.makedirs(_long_path(path), exist_ok=True)


def _write_csv(path, fieldnames, rows):
    _makedirs(os.path.dirname(os.path.abspath(os.fspath(path))))
    with _open_text(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def build_fold_prediction_rows(
    val_indices,
    y_true,
    predictions,
    sample_metadata,
    fold,
):
    """Attach source-WAV/time provenance to one outer-fold prediction set."""
    val_indices = np.asarray(val_indices).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    model_keys = list(predictions)
    prediction_arrays = {
        key: np.asarray(values, dtype=float).ravel()
        for key, values in predictions.items()
    }
    expected = len(val_indices)
    if len(y_true) != expected:
        raise ValueError("val_indices and y_true must have the same length.")
    for key, values in prediction_arrays.items():
        if len(values) != expected:
            raise ValueError(
                f"Prediction length mismatch for {key}: {len(values)} != {expected}."
            )

    rows = []
    for row_i, sample_index in enumerate(val_indices):
        sample_index = int(sample_index)
        if not 0 <= sample_index < len(sample_metadata):
            raise IndexError(f"sample_index outside metadata: {sample_index}")
        metadata = dict(sample_metadata[sample_index])
        source_wav_id = str(metadata.get("source_wav_id", "")).strip()
        if not source_wav_id:
            raise ValueError(
                f"source_wav_id is missing for sample_index={sample_index}."
            )
        duration = _to_float(metadata.get("chunk_duration_seconds"), default=1.0)
        chunk_index = _to_int(metadata.get("chunk_index"), default=row_i)
        start_seconds = _to_float(
            metadata.get("chunk_start_seconds"),
            default=chunk_index * duration,
        )
        row = {
            "fold": int(fold),
            "sample_index": sample_index,
            "sample_filename": metadata.get("sample_filename", ""),
            "experiment_name": metadata.get("experiment_name", ""),
            "source_wav_id": source_wav_id,
            "source_wav_name": metadata.get("source_wav_name", ""),
            "chunk_index": chunk_index,
            "chunk_start_seconds": start_seconds,
            "chunk_duration_seconds": duration,
            "y_true": float(y_true[row_i]),
        }
        row.update({key: float(prediction_arrays[key][row_i]) for key in model_keys})
        rows.append(row)
    return rows


def write_fold_prediction_csv(path, rows, model_keys):
    """Write enriched fold predictions in a stable, post-processable schema."""
    _write_csv(path, PREDICTION_METADATA_COLUMNS + list(model_keys), rows)


def _predicted_event_stats(rows, prediction_values, threshold):
    """Summarize contiguous predicted-threshold crossings within one WAV."""
    if not _has_threshold(threshold):
        return {
            "predicted_positive_chunk_count": None,
            "predicted_positive_chunk_rate": np.nan,
            "predicted_positive_run_count": None,
            "first_predicted_positive_seconds": np.nan,
            "longest_predicted_positive_run_seconds": np.nan,
        }

    threshold = float(threshold)
    positive_count = 0
    run_count = 0
    first_positive = np.nan
    longest_duration = 0.0
    current_start = None
    current_end = None

    for row, prediction in zip(rows, prediction_values):
        start = _to_float(row.get("chunk_start_seconds"))
        duration = _to_float(row.get("chunk_duration_seconds"), default=1.0)
        if not np.isfinite(start):
            start = _to_int(row.get("chunk_index")) * duration
        end = start + max(duration, 0.0)
        is_positive = np.isfinite(prediction) and prediction >= threshold

        if not is_positive:
            if current_start is not None:
                longest_duration = max(longest_duration, current_end - current_start)
                current_start = current_end = None
            continue

        positive_count += 1
        if not np.isfinite(first_positive):
            first_positive = start

        tolerance = max(1e-9, abs(duration) * 1e-6)
        contiguous = (
            current_start is not None
            and start <= current_end + tolerance
            and start >= current_end - tolerance
        )
        if not contiguous:
            if current_start is not None:
                longest_duration = max(longest_duration, current_end - current_start)
            run_count += 1
            current_start = start
            current_end = end
        else:
            current_end = max(current_end, end)

    if current_start is not None:
        longest_duration = max(longest_duration, current_end - current_start)

    return {
        "predicted_positive_chunk_count": int(positive_count),
        "predicted_positive_chunk_rate": (
            float(positive_count / len(rows)) if rows else np.nan
        ),
        "predicted_positive_run_count": int(run_count),
        "first_predicted_positive_seconds": first_positive,
        "longest_predicted_positive_run_seconds": float(longest_duration),
    }


def aggregate_oof_predictions_by_wav(
    chunk_rows,
    model_keys,
    threshold,
    aggregations=("mean", "median", "p90"),
):
    """Return one row per source WAV from pooled outer-fold predictions."""
    aggregations = tuple(aggregations)
    unknown = [name for name in aggregations if name not in SUPPORTED_AGGREGATIONS]
    if unknown:
        raise ValueError(
            f"Unknown WAV aggregations {unknown}; supported={list(SUPPORTED_AGGREGATIONS)}"
        )

    grouped = defaultdict(list)
    seen_indices = set()
    for row in chunk_rows:
        source_wav_id = str(row.get("source_wav_id", "")).strip()
        if not source_wav_id:
            raise ValueError("Every OOF chunk row requires source_wav_id.")
        sample_index = _to_int(row.get("sample_index"), default=-1)
        if sample_index >= 0:
            if sample_index in seen_indices:
                raise ValueError(
                    f"Duplicate OOF sample_index={sample_index}; each chunk must occur once."
                )
            seen_indices.add(sample_index)
        experiment_name = str(row.get("experiment_name", ""))
        grouped[(experiment_name, source_wav_id)].append(dict(row))

    wav_rows = []
    for (experiment_name, source_wav_id), rows in grouped.items():
        rows.sort(
            key=lambda row: (
                _to_float(row.get("chunk_start_seconds"), default=math.inf),
                _to_int(row.get("chunk_index")),
                _to_int(row.get("sample_index")),
            )
        )
        y_values = np.asarray([_to_float(row.get("y_true")) for row in rows])
        if not np.all(np.isfinite(y_values)):
            raise ValueError(f"Non-finite y_true in source_wav_id={source_wav_id}.")
        tolerance = max(1e-6, float(np.max(np.abs(y_values))) * 1e-9)
        if not np.allclose(y_values, y_values[0], rtol=0.0, atol=tolerance):
            raise ValueError(
                f"One source WAV has multiple heat-flux labels: {source_wav_id}."
            )
        folds = sorted({_to_int(row.get("fold")) for row in rows})
        if len(folds) != 1:
            raise ValueError(
                f"source_wav_id={source_wav_id} appears in multiple outer folds: {folds}."
            )

        wav_row = {
            "fold": folds[0],
            "experiment_name": experiment_name,
            "source_wav_id": source_wav_id,
            "source_wav_name": rows[0].get("source_wav_name", ""),
            "y_true": float(y_values[0]),
            "true_onb_state": (
                int(y_values[0] >= float(threshold))
                if _has_threshold(threshold) else None
            ),
            "n_chunks": len(rows),
            "first_chunk_start_seconds": _to_float(
                rows[0].get("chunk_start_seconds")
            ),
            "last_chunk_start_seconds": _to_float(
                rows[-1].get("chunk_start_seconds")
            ),
        }
        for model_key in model_keys:
            values = np.asarray(
                [_to_float(row.get(model_key)) for row in rows], dtype=float
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Missing/non-finite {model_key} prediction in {source_wav_id}."
                )
            for aggregation in aggregations:
                wav_row[f"{model_key}_pred_{aggregation}"] = (
                    SUPPORTED_AGGREGATIONS[aggregation](values)
                )
            wav_row[f"{model_key}_pred_std"] = float(np.std(values, ddof=0))
            event_stats = _predicted_event_stats(rows, values, threshold)
            for name, value in event_stats.items():
                wav_row[f"{model_key}_{name}"] = value
        wav_rows.append(wav_row)

    wav_rows.sort(key=lambda row: (row["experiment_name"], row["y_true"], row["source_wav_id"]))
    return wav_rows


def summarize_pooled_oof_wav_metrics(
    wav_rows,
    model_keys,
    threshold,
    band_frac,
    aggregations=("mean", "median", "p90"),
    claim_safe_by_model=None,
    claim_note_by_model=None,
):
    """Compute metrics once over all held-out WAVs, never mean tiny-fold R2s."""
    calculator = RegressionDetectionMetrics()
    claim_safe_by_model = dict(claim_safe_by_model or {})
    claim_note_by_model = dict(claim_note_by_model or {})
    y_true = np.asarray([row["y_true"] for row in wav_rows], dtype=float)
    summary_rows = []
    for model_key in model_keys:
        for aggregation in aggregations:
            column = f"{model_key}_pred_{aggregation}"
            y_pred = np.asarray([row[column] for row in wav_rows], dtype=float)
            combined = {}
            combined.update(
                calculator.regression_metrics(y_true, y_pred, threshold, band_frac)
            )
            combined.update(
                calculator.detection_metrics_continuous(y_true, y_pred, threshold)
            )
            combined.update(
                calculator.detection_metrics_binary(y_true, y_pred, threshold)
            )
            summary_rows.append({
                "evaluation_unit": "source_wav_pooled_oof",
                "model_key": model_key,
                "aggregation": aggregation,
                "n_wavs": len(wav_rows),
                "claim_safe": int(claim_safe_by_model.get(model_key, True)),
                "claim_note": claim_note_by_model.get(model_key, ""),
                **combined,
            })
    return summary_rows


def predicted_event_summary_rows(wav_rows, model_keys, threshold):
    """Create a long-form table of descriptive predicted-event quantities."""
    output = []
    for row in wav_rows:
        for model_key in model_keys:
            output.append({
                "fold": row["fold"],
                "experiment_name": row["experiment_name"],
                "source_wav_id": row["source_wav_id"],
                "source_wav_name": row["source_wav_name"],
                "y_true": row["y_true"],
                "true_onb_state_from_heat_flux": row["true_onb_state"],
                "model_key": model_key,
                "threshold": float(threshold) if _has_threshold(threshold) else None,
                "n_chunks": row["n_chunks"],
                "predicted_positive_chunk_count": row[
                    f"{model_key}_predicted_positive_chunk_count"
                ],
                "predicted_positive_chunk_rate": row[
                    f"{model_key}_predicted_positive_chunk_rate"
                ],
                "predicted_positive_run_count": row[
                    f"{model_key}_predicted_positive_run_count"
                ],
                "first_predicted_positive_seconds": row[
                    f"{model_key}_first_predicted_positive_seconds"
                ],
                "longest_predicted_positive_run_seconds": row[
                    f"{model_key}_longest_predicted_positive_run_seconds"
                ],
                "ground_truth_event_annotation_available": 0,
                "interpretation": "predicted_heat_flux_threshold_crossing_only",
            })
    return output


def _first_persistent_positive_index(states, persistence_wavs):
    """Return the start of the first positive run of the requested length."""
    persistence_wavs = int(persistence_wavs)
    if persistence_wavs <= 0:
        raise ValueError("persistence_wavs must be a positive integer.")
    run_start = None
    run_length = 0
    for index, state in enumerate(states):
        if bool(state):
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= persistence_wavs:
                return int(run_start)
        else:
            run_start = None
            run_length = 0
    return None


def onb_transition_summary_rows(
    wav_rows,
    model_keys,
    threshold,
    aggregations=("mean", "median", "p90"),
    persistence_wavs=(1, 2),
    claim_safe_by_model=None,
    claim_note_by_model=None,
):
    """Evaluate the threshold-defined ONB transition along operating points.

    The result is an error in measured heat-flux levels/operating-point steps,
    not a time delay in seconds.  A synchronized physical-event annotation is
    required before a temporal lead/lag can be reported.
    """
    if not _has_threshold(threshold):
        return []
    threshold = float(threshold)
    aggregations = tuple(aggregations)
    persistence_wavs = tuple(int(value) for value in persistence_wavs)
    unknown = [name for name in aggregations if name not in SUPPORTED_AGGREGATIONS]
    if unknown:
        raise ValueError(f"Unknown WAV aggregations: {unknown}")
    if not persistence_wavs or any(value <= 0 for value in persistence_wavs):
        raise ValueError("persistence_wavs must contain positive integers.")

    claim_safe_by_model = dict(claim_safe_by_model or {})
    claim_note_by_model = dict(claim_note_by_model or {})
    by_experiment = defaultdict(list)
    for row in wav_rows:
        by_experiment[str(row.get("experiment_name", ""))].append(row)

    output = []
    for experiment_name, experiment_rows in sorted(by_experiment.items()):
        ordered = sorted(
            experiment_rows,
            key=lambda row: (float(row["y_true"]), str(row["source_wav_id"])),
        )
        true_states = [float(row["y_true"]) >= threshold for row in ordered]
        true_index = next(
            (index for index, state in enumerate(true_states) if state), None
        )
        true_heat_flux = (
            float(ordered[true_index]["y_true"]) if true_index is not None else np.nan
        )

        for model_key in model_keys:
            for aggregation in aggregations:
                prediction_column = f"{model_key}_pred_{aggregation}"
                predictions = [float(row[prediction_column]) for row in ordered]
                predicted_states = [value >= threshold for value in predictions]
                false_positive_count = (
                    int(sum(predicted_states[:true_index]))
                    if true_index is not None else int(sum(predicted_states))
                )

                for persistence in persistence_wavs:
                    predicted_index = _first_persistent_positive_index(
                        predicted_states, persistence
                    )
                    if true_index is None:
                        post_onb_index = None
                    else:
                        relative_index = _first_persistent_positive_index(
                            predicted_states[true_index:], persistence
                        )
                        post_onb_index = (
                            true_index + relative_index
                            if relative_index is not None else None
                        )

                    predicted_heat_flux = (
                        float(ordered[predicted_index]["y_true"])
                        if predicted_index is not None else np.nan
                    )
                    post_onb_heat_flux = (
                        float(ordered[post_onb_index]["y_true"])
                        if post_onb_index is not None else np.nan
                    )
                    output.append({
                        "experiment_name": experiment_name,
                        "evaluation_unit": "threshold_defined_onb_transition",
                        "prediction_scope": "pooled_outer_fold_oof",
                        "model_key": model_key,
                        "aggregation": aggregation,
                        "persistence_wavs": persistence,
                        "n_operating_points": len(ordered),
                        "threshold": threshold,
                        "true_transition_available": int(true_index is not None),
                        "true_transition_index_0based": true_index,
                        "true_transition_heat_flux": true_heat_flux,
                        "predicted_transition_available": int(
                            predicted_index is not None
                        ),
                        "predicted_transition_index_0based": predicted_index,
                        "predicted_transition_at_true_heat_flux": predicted_heat_flux,
                        "signed_transition_step_error": (
                            predicted_index - true_index
                            if predicted_index is not None and true_index is not None
                            else None
                        ),
                        "signed_transition_heat_flux_error": (
                            predicted_heat_flux - true_heat_flux
                            if np.isfinite(predicted_heat_flux)
                            and np.isfinite(true_heat_flux) else np.nan
                        ),
                        "false_positive_wavs_before_onb": false_positive_count,
                        "post_onb_detection_available": int(
                            post_onb_index is not None
                        ),
                        "first_post_onb_detection_index_0based": post_onb_index,
                        "post_onb_detection_at_true_heat_flux": post_onb_heat_flux,
                        "post_onb_detection_delay_steps": (
                            post_onb_index - true_index
                            if post_onb_index is not None and true_index is not None
                            else None
                        ),
                        "post_onb_detection_delay_heat_flux": (
                            post_onb_heat_flux - true_heat_flux
                            if np.isfinite(post_onb_heat_flux)
                            and np.isfinite(true_heat_flux) else np.nan
                        ),
                        "detected_at_true_onb_operating_point": int(
                            true_index is not None and post_onb_index == true_index
                        ),
                        "claim_safe": int(claim_safe_by_model.get(model_key, True)),
                        "claim_note": claim_note_by_model.get(model_key, ""),
                        "temporal_delay_seconds_available": 0,
                        "interpretation": (
                            "heat_flux_operating_point_transition; "
                            "not_synchronized_time_delay"
                        ),
                    })
    return output


def save_wav_event_evaluation(
    save_path,
    snr_value,
    chunk_rows,
    model_keys,
    threshold,
    band_frac,
    aggregations=("mean", "median", "p90"),
    primary_aggregation="median",
    save_predicted_event_summary=True,
    onb_transition_persistence_wavs=(1, 2),
    claim_safe_by_model=None,
    claim_note_by_model=None,
    threshold_provenance=None,
):
    """Persist enriched OOF chunks, WAV metrics, and predicted-event summaries."""
    aggregations = tuple(aggregations)
    if primary_aggregation not in aggregations:
        raise ValueError("primary_aggregation must be included in aggregations.")
    if not chunk_rows:
        raise ValueError("No OOF chunk predictions were supplied.")

    # Keep the directory short because run paths are already near the Windows
    # legacy path-length limit.
    evaluation_dir = os.path.join(os.fspath(save_path), "wav_eval")
    _makedirs(evaluation_dir)
    wav_rows = aggregate_oof_predictions_by_wav(
        chunk_rows,
        model_keys,
        threshold,
        aggregations=aggregations,
    )
    metric_rows = summarize_pooled_oof_wav_metrics(
        wav_rows,
        model_keys,
        threshold,
        band_frac,
        aggregations=aggregations,
        claim_safe_by_model=claim_safe_by_model,
        claim_note_by_model=claim_note_by_model,
    )
    event_rows = (
        predicted_event_summary_rows(wav_rows, model_keys, threshold)
        if save_predicted_event_summary else []
    )
    transition_rows = onb_transition_summary_rows(
        wav_rows,
        model_keys,
        threshold,
        aggregations=aggregations,
        persistence_wavs=onb_transition_persistence_wavs,
        claim_safe_by_model=claim_safe_by_model,
        claim_note_by_model=claim_note_by_model,
    )

    chunk_path = os.path.join(
        evaluation_dir, f"oof_chunk_predictions_{snr_value}.csv"
    )
    wav_path = os.path.join(
        evaluation_dir, f"wav_predictions_{snr_value}.csv"
    )
    metrics_path = os.path.join(
        evaluation_dir, f"wav_metrics_{snr_value}.csv"
    )
    event_path = os.path.join(
        evaluation_dir, f"predicted_event_summary_{snr_value}.csv"
    )
    transition_path = os.path.join(
        evaluation_dir, f"onb_transition_summary_{snr_value}.csv"
    )

    _write_csv(chunk_path, PREDICTION_METADATA_COLUMNS + list(model_keys), chunk_rows)
    wav_base = [
        "fold", "experiment_name", "source_wav_id", "source_wav_name",
        "y_true", "true_onb_state", "n_chunks",
        "first_chunk_start_seconds", "last_chunk_start_seconds",
    ]
    wav_model_columns = []
    for model_key in model_keys:
        wav_model_columns.extend(
            [f"{model_key}_pred_{aggregation}" for aggregation in aggregations]
        )
        wav_model_columns.extend([
            f"{model_key}_pred_std",
            f"{model_key}_predicted_positive_chunk_count",
            f"{model_key}_predicted_positive_chunk_rate",
            f"{model_key}_predicted_positive_run_count",
            f"{model_key}_first_predicted_positive_seconds",
            f"{model_key}_longest_predicted_positive_run_seconds",
        ])
    _write_csv(wav_path, wav_base + wav_model_columns, wav_rows)

    metric_names = [
        "r2", "rmse_all", "mae_all", "r2_high", "rmse_high", "mae_high",
        "rmse_onb", "mae_onb", "n_onb", "roc_auc_cont", "pr_auc_cont",
        "accuracy", "precision", "recall", "f1", "auc_binary",
    ]
    _write_csv(
        metrics_path,
        [
            "evaluation_unit", "model_key", "aggregation", "n_wavs",
            "claim_safe", "claim_note",
        ] + metric_names,
        metric_rows,
    )
    event_columns = [
        "fold", "experiment_name", "source_wav_id", "source_wav_name",
        "y_true", "true_onb_state_from_heat_flux", "model_key", "threshold",
        "n_chunks", "predicted_positive_chunk_count",
        "predicted_positive_chunk_rate", "predicted_positive_run_count",
        "first_predicted_positive_seconds",
        "longest_predicted_positive_run_seconds",
        "ground_truth_event_annotation_available", "interpretation",
    ]
    if save_predicted_event_summary:
        _write_csv(event_path, event_columns, event_rows)

    transition_columns = [
        "experiment_name", "evaluation_unit", "prediction_scope", "model_key",
        "aggregation", "persistence_wavs", "n_operating_points", "threshold",
        "true_transition_available", "true_transition_index_0based",
        "true_transition_heat_flux", "predicted_transition_available",
        "predicted_transition_index_0based",
        "predicted_transition_at_true_heat_flux",
        "signed_transition_step_error", "signed_transition_heat_flux_error",
        "false_positive_wavs_before_onb", "post_onb_detection_available",
        "first_post_onb_detection_index_0based",
        "post_onb_detection_at_true_heat_flux",
        "post_onb_detection_delay_steps",
        "post_onb_detection_delay_heat_flux",
        "detected_at_true_onb_operating_point", "claim_safe", "claim_note",
        "temporal_delay_seconds_available", "interpretation",
    ]
    _write_csv(transition_path, transition_columns, transition_rows)

    manifest = {
        "evaluation_unit": "source_wav_id",
        "generalization_scope": (
            "within_experiment source-WAV OOF; not unseen-experiment-day validation"
        ),
        "same_day_wav_independence_assumed": False,
        "fold_handling": "pooled out-of-fold WAV predictions",
        "n_oof_chunks": len(chunk_rows),
        "n_wavs": len(wav_rows),
        "model_keys": list(model_keys),
        "aggregations": list(aggregations),
        "primary_aggregation": primary_aggregation,
        "onb_transition_persistence_wavs": list(onb_transition_persistence_wavs),
        "claim_safe_by_model": dict(claim_safe_by_model or {}),
        "claim_note_by_model": dict(claim_note_by_model or {}),
        "threshold": float(threshold) if _has_threshold(threshold) else None,
        "threshold_provenance": threshold_provenance,
        "within_wav_event_output_status": (
            "descriptive predictions only"
            if save_predicted_event_summary else "disabled"
        ),
        "within_wav_event_caveat": (
            "Threshold crossings are not ground-truth bubble events. "
            "Event precision/recall require synchronized annotations."
        ),
        "onb_transition_output_status": (
            "threshold-defined heat-flux operating-point transition"
        ),
        "onb_transition_caveat": (
            "Step and heat-flux offsets are available; temporal delay in "
            "seconds requires synchronized physical-event annotations."
        ),
    }
    manifest_path = os.path.join(
        evaluation_dir, f"evaluation_manifest_{snr_value}.json"
    )
    with _open_text(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)

    return {
        "chunk_path": chunk_path,
        "wav_path": wav_path,
        "metrics_path": metrics_path,
        "event_path": event_path if save_predicted_event_summary else None,
        "transition_path": transition_path,
        "manifest_path": manifest_path,
        "n_chunk_rows": len(chunk_rows),
        "wav_rows": wav_rows,
        "metric_rows": metric_rows,
        "transition_rows": transition_rows,
    }


def load_sample_metadata_without_arrays(data_path):
    """Load metadata in the exact sorted-.npy order used by the training loader."""
    data_path = os.fspath(data_path)
    manifest_path = os.path.join(data_path, "chunk_manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"chunk_manifest.csv not found: {manifest_path}")
    with _open_text(manifest_path, newline="", encoding="utf-8-sig") as source:
        manifest_by_filename = {
            row["sample_filename"]: row
            for row in csv.DictReader(source)
            if row.get("sample_filename")
        }
    filenames = sorted(
        entry.name for entry in os.scandir(data_path)
        if entry.is_file() and entry.name.endswith(".npy")
    )
    metadata = []
    for filename in filenames:
        if filename not in manifest_by_filename:
            raise ValueError(f"Manifest row missing for {filename} in {data_path}")
        metadata.append({"sample_filename": filename, **manifest_by_filename[filename]})
    return metadata


def read_saved_fold_predictions(prediction_dir, snr_value, sample_metadata):
    """Read new or legacy fold CSVs and backfill metadata by sample_index."""
    filename_pattern = re.compile(
        rf"^pred_f(?P<fold>\d+)_{re.escape(str(snr_value))}\.csv$"
    )
    files = []
    for entry in os.scandir(_long_path(prediction_dir)):
        if entry.is_file():
            match = filename_pattern.match(entry.name)
            if match:
                files.append((int(match.group("fold")), entry.path))
    files.sort()
    if not files:
        raise FileNotFoundError(
            f"No fold predictions for SNR={snr_value} in {prediction_dir}"
        )

    base_columns = set(PREDICTION_METADATA_COLUMNS)
    output_rows = []
    model_keys = None
    for fold, path in files:
        with _open_text(path, newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            current_model_keys = [
                name for name in (reader.fieldnames or []) if name not in base_columns
            ]
            if model_keys is None:
                model_keys = current_model_keys
            elif current_model_keys != model_keys:
                raise ValueError(f"Model columns differ across fold CSVs: {path}")

            for saved_row in reader:
                sample_index = _to_int(saved_row.get("sample_index"), default=-1)
                if not 0 <= sample_index < len(sample_metadata):
                    raise IndexError(
                        f"sample_index={sample_index} cannot map to dataset metadata."
                    )
                metadata = sample_metadata[sample_index]
                duration = _to_float(
                    saved_row.get("chunk_duration_seconds")
                    or metadata.get("chunk_duration_seconds"),
                    default=1.0,
                )
                chunk_index = _to_int(
                    saved_row.get("chunk_index") or metadata.get("chunk_index")
                )
                row = {
                    "fold": _to_int(saved_row.get("fold"), default=fold),
                    "sample_index": sample_index,
                    "sample_filename": saved_row.get("sample_filename") or metadata.get("sample_filename", ""),
                    "experiment_name": saved_row.get("experiment_name") or metadata.get("experiment_name", ""),
                    "source_wav_id": saved_row.get("source_wav_id") or metadata.get("source_wav_id", ""),
                    "source_wav_name": saved_row.get("source_wav_name") or metadata.get("source_wav_name", ""),
                    "chunk_index": chunk_index,
                    "chunk_start_seconds": _to_float(
                        saved_row.get("chunk_start_seconds")
                        or metadata.get("chunk_start_seconds"),
                        default=chunk_index * duration,
                    ),
                    "chunk_duration_seconds": duration,
                    # Historical fold CSVs formatted y_true with only 10
                    # significant digits.  Prefer the manifest heat_flux so
                    # an exact ONB sample cannot move below the threshold.
                    "y_true": _to_float(
                        metadata.get("heat_flux"),
                        default=_to_float(saved_row.get("y_true")),
                    ),
                }
                row.update({
                    key: _to_float(saved_row.get(key)) for key in (model_keys or [])
                })
                output_rows.append(row)
    return output_rows, list(model_keys or [])
