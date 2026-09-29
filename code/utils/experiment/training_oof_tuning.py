"""Hyperparameter tuning confined to the outer-training partition.

The normal ONB experiment runner evaluates fixed parameters on its outer test
partition.  This module is a separate mode: candidates are ranked only by
source-WAV-disjoint out-of-fold predictions made inside the outer-training
partition.  It deliberately does not load outer-test arrays or produce a test
score, preventing parameter selection on the final evaluation data.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.config.parameter_sets import parameter_set_tag, resolve_parameter_set
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import (
    build_learning_families,
    checked_metadata,
    experiment_split_kind,
    outer_splits,
    wav_groups,
)
from utils.experiment.result_paths import result_scope_dir_name
from utils.experiment.run_helpers import json_default, safe_tag, set_global_seed
from utils.training.internal_validation import fit_individual_performance_cv


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, default=json_default)


def _source_day(row):
    return str(row.get("source_experiment_name") or row["experiment_name"])


def _finite_metric(value):
    return float(value) if np.isfinite(value) else None


def oof_regression_metrics(y, prediction, metadata, thresholds, onb_band_frac):
    """Compute ranking and ONB diagnostics from training-only OOF predictions."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or len(y) != len(metadata):
        raise ValueError("OOF targets, predictions, and metadata must have equal lengths.")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("OOF targets and predictions must be finite.")
    source_days = np.asarray([_source_day(row) for row in metadata])
    missing = sorted(set(source_days) - set(thresholds))
    if missing:
        raise ValueError(f"Missing source-day ONB thresholds for tuning: {missing}")
    sample_thresholds = np.asarray([float(thresholds[day]) for day in source_days])
    if not np.isfinite(sample_thresholds).all():
        raise ValueError("Tuning ONB thresholds must be finite.")

    pre = y < sample_thresholds
    post = ~pre
    near = np.abs(y - sample_thresholds) <= np.abs(sample_thresholds) * float(onb_band_frac)

    def rmse(mask):
        return (float(np.sqrt(mean_squared_error(y[mask], prediction[mask])))
                if np.any(mask) else None)

    def mae(mask):
        return float(mean_absolute_error(y[mask], prediction[mask])) if np.any(mask) else None

    def bias(mask):
        return float(np.mean(prediction[mask] - y[mask])) if np.any(mask) else None

    r2 = r2_score(y, prediction) if len(y) >= 2 and np.var(y) > 0 else np.nan
    result = {
        "n_samples": int(len(y)),
        "n_source_wavs": int(len(set(wav_groups(metadata)))),
        "r2": _finite_metric(r2),
        "rmse_all": rmse(np.ones(len(y), dtype=bool)),
        "mae_all": mae(np.ones(len(y), dtype=bool)),
        "bias_all": bias(np.ones(len(y), dtype=bool)),
        "n_pre_onb": int(pre.sum()),
        "rmse_pre_onb": rmse(pre),
        "bias_pre_onb": bias(pre),
        "false_positive_rate": (float(np.mean(prediction[pre] >= sample_thresholds[pre]))
                                if np.any(pre) else None),
        "n_post_onb": int(post.sum()),
        "rmse_post_onb": rmse(post),
        "bias_post_onb": bias(post),
        "recall": (float(np.mean(prediction[post] >= sample_thresholds[post]))
                   if np.any(post) else None),
        "n_onb_band": int(near.sum()),
        "rmse_onb": rmse(near),
        "mae_onb": mae(near),
    }
    return result


def _output_directory(family, config):
    configured = config["tuning"].get("output_dir")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path

    job = family["evaluation_jobs"][0]
    experiment_root = Path(job["experiment_root"])
    ensemble_root = experiment_root / "regression_result" / "npy" / "ensemble"
    save_base = Path(job["save_base_path"])
    try:
        relative = save_base.relative_to(ensemble_root)
    except ValueError:
        relative = Path(save_base.name)
    scope_name = result_scope_dir_name(
        "tuning",
        job,
        config,
        config["output"]["run_instance_id"],
        "training-oof",
    )
    return (experiment_root / "regression_result" / "npy" / "tuning"
            / relative.parent / scope_name)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_training_oof_tuning(jobs, policy, config, enabled_specs, parameter_sets, trainer):
    """Rank independent model candidates without scoring the outer test set."""
    if config["tuning"].get("mode") != "training_oof":
        raise ValueError("Only tuning.mode='training_oof' is supported.")
    if policy["training_noise"] != "clean_only":
        raise ValueError(
            "training_oof tuning currently requires clean_only so one candidate ranking "
            "is not silently fitted separately for every noise condition."
        )
    families = build_learning_families(jobs, policy, config["data"]["experiment_names"])
    if len(families) != 1:
        raise ValueError(
            "training_oof tuning requires exactly one learning family. "
            "Tune one experiment/frequency scope at a time."
        )
    family = families[0]
    loader = DataLoadingConversion()
    train_x, train_y, train_metadata = [], [], []
    for training_job in family["training_jobs"]:
        x, y, metadata = loader.load_npy_data(training_job["data_path"], return_metadata=True)
        train_x.append(x)
        train_y.append(y)
        train_metadata.extend(checked_metadata(metadata, training_job["experiment_name"]))
    x = train_x[0] if len(train_x) == 1 else np.concatenate(train_x)
    y = train_y[0] if len(train_y) == 1 else np.concatenate(train_y)

    evaluation_job = family["evaluation_jobs"][0]
    evaluation_metadata = checked_metadata(
        load_sample_metadata_without_arrays(evaluation_job["data_path"]),
        evaluation_job["experiment_name"],
    )
    split_kind = experiment_split_kind(policy)
    outer_fold_count = config["run"]["folds"] if split_kind == "within_day" else 1
    splits = outer_splits(
        train_metadata,
        evaluation_metadata,
        outer_fold_count,
        policy=policy,
        threshold=evaluation_job["threshold"],
    )
    if len(splits) != 1:
        raise ValueError(
            "training_oof tuning needs one fixed outer holdout. Use within_wav_chunk, "
            "within_day holdout, or cross_day; do not rank candidates across outer CV folds."
        )
    fit_indices = np.asarray(splits[0][0], dtype=int)
    metadata_fit = [train_metadata[int(index)] for index in fit_indices]
    if len(set(wav_groups(metadata_fit))) < int(config["run"]["folds"]):
        raise ValueError("The outer-training partition has fewer WAV groups than tuning folds.")

    selector = AcousticTrainingSelector(
        config.get("acoustic_selection"),
        config["thresholds"].get("by_experiment", {}),
    )
    output_dir = _output_directory(family, config)
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_dir = output_dir / "candidate_audits"
    audit_dir.mkdir()

    metric_name = config["tuning"].get("selection_metric", "rmse_all")
    allowed_metrics = {"rmse_all", "mae_all", "rmse_onb"}
    if metric_name not in allowed_metrics:
        raise ValueError(f"selection_metric must be one of {sorted(allowed_metrics)}")
    epochs = int(config["run"]["epochs"])
    all_rows, day_rows = [], []
    spec_by_key = {spec["key"]: spec for spec in enabled_specs}
    for candidate_index, parameter_set in enumerate(parameter_sets, 1):
        requested = list(parameter_set.get("active_model_keys") or [])
        if len(requested) != 1:
            raise ValueError(
                "Each training_oof candidate must select exactly one model via active_model_keys."
            )
        model_key = requested[0]
        if model_key not in spec_by_key:
            raise ValueError(f"Tuning candidate selects inactive model: {model_key}")
        specs = resolve_parameter_set(enabled_specs, parameter_set)
        if len(specs) != 1 or specs[0]["key"] != model_key:
            raise RuntimeError("Independent tuning candidate did not resolve to one model.")
        tag = parameter_set_tag(parameter_set, specs, safe_tag)
        print(
            f"[training OOF tuning] {candidate_index}/{len(parameter_sets)}: "
            f"{model_key} | {parameter_set['name']}",
            flush=True,
        )
        set_global_seed(config["run"]["random_seed"])
        _, audit = fit_individual_performance_cv(
            trainer,
            specs,
            x[fit_indices],
            y[fit_indices],
            metadata_fit,
            selector,
            config["run"]["folds"],
            config["run"]["random_seed"],
            config["features"]["pca_components"],
            {model_key: epochs},
        )
        prediction = np.asarray([sample[model_key] for sample in audit["samples"]], dtype=float)
        metrics = oof_regression_metrics(
            y[fit_indices],
            prediction,
            metadata_fit,
            config["thresholds"]["by_experiment"],
            config["thresholds"]["onb_band_frac"],
        )
        row = {
            "candidate_index": candidate_index,
            "model_key": model_key,
            "candidate_name": parameter_set["name"],
            "candidate_tag": tag,
            "parameters_json": json.dumps(parameter_set.get("models", {}).get(model_key, {}),
                                          ensure_ascii=False, sort_keys=True),
            **metrics,
        }
        all_rows.append(row)
        for source_day in sorted({_source_day(item) for item in metadata_fit}):
            mask = np.asarray([_source_day(item) == source_day for item in metadata_fit])
            day_metrics = oof_regression_metrics(
                y[fit_indices][mask], prediction[mask],
                [item for item, keep in zip(metadata_fit, mask) if keep],
                config["thresholds"]["by_experiment"],
                config["thresholds"]["onb_band_frac"],
            )
            day_rows.append({
                "candidate_index": candidate_index,
                "model_key": model_key,
                "candidate_name": parameter_set["name"],
                "source_experiment_name": source_day,
                **day_metrics,
            })
        _write_json(audit_dir / f"{candidate_index:02d}_{tag}.json", {
            "parameter_set": parameter_set,
            "metrics": metrics,
            "internal_validation": audit,
        })

    _write_csv(output_dir / "candidate_metrics.csv", all_rows)
    _write_csv(output_dir / "candidate_metrics_by_source_day.csv", day_rows)
    tolerance = float(config["tuning"].get("relative_tolerance", 0.02))
    if not 0 <= tolerance < 1:
        raise ValueError("tuning.relative_tolerance must be between 0 and 1.")
    selected = {}
    for model_key in sorted({row["model_key"] for row in all_rows}):
        candidates = [row for row in all_rows if row["model_key"] == model_key]
        if any(row[metric_name] is None for row in candidates):
            raise ValueError(f"{metric_name} is undefined for a {model_key} candidate.")
        best = min(candidates, key=lambda row: (row[metric_name], row["candidate_index"]))
        limit = best[metric_name] * (1.0 + tolerance)
        equivalent = [row["candidate_name"] for row in candidates if row[metric_name] <= limit]
        selected[model_key] = {
            "candidate_name": best["candidate_name"],
            "candidate_tag": best["candidate_tag"],
            "selection_metric": metric_name,
            "selection_value": best[metric_name],
            "parameters": json.loads(best["parameters_json"]),
            "within_relative_tolerance": equivalent,
        }
    summary = {
        "mode": "training_oof",
        "outer_test_used": False,
        "candidate_ranking_scope": "outer-training partition, source-WAV-disjoint OOF",
        "evaluation_scheme": split_kind,
        "n_outer_training_samples": int(len(fit_indices)),
        "n_outer_test_samples_not_scored": int(len(splits[0][1])),
        "folds": int(config["run"]["folds"]),
        "selection_metric": metric_name,
        "relative_tolerance": tolerance,
        "selected": selected,
        "normal_run_unchanged": True,
        "next_step": (
            "Copy the chosen parameters into models.parameter_sets, set tuning.enabled=False, "
            "and run the normal pipeline once for the untouched outer test."
        ),
    }
    _write_json(output_dir / "tuning_config.json", config)
    _write_json(output_dir / "selected_candidates.json", summary)
    _write_json(output_dir / "completed.json", {"completed": True, **summary})
    print(f"training-only OOF tuning results saved: {output_dir}", flush=True)
    return {"output_dir": output_dir, **summary}
