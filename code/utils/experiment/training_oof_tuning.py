"""Hyperparameter tuning confined to the outer-training partition.

The normal ONB experiment runner evaluates fixed parameters on its outer test
partition.  This module is a separate mode: candidates are ranked only by
shuffled chunk out-of-fold predictions made inside the outer-training
partition.  Outer-test samples enter neither fitting nor candidate ranking,
and this module produces no outer-test score.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.config.parameter_sets import parameter_set_tag, resolve_parameter_set
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.experiment.acoustic_selection import AcousticTrainingSelector
from utils.experiment.learning_policy import (
    experiment_split_kind,
    build_learning_families,
    checked_metadata,
    outer_splits,
    wav_groups,
)
from utils.experiment.result_paths import result_scope_dir_name
from utils.experiment.run_helpers import json_default, safe_tag, set_global_seed
from utils.training.internal_validation import fit_individual_performance_cv, internal_splits


SELECTION_METRIC = "rmse_all"
RELATIVE_TOLERANCE = 0.02


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, default=json_default)
    temporary.replace(path)


def tuning_resume_context(config, specs):
    """Bind cached OOF predictions to the data, fit policy and model code."""
    root = Path(__file__).resolve().parents[3]
    code_paths = [
        "code/utils/models/regression/base_regression.py",
        "code/utils/models/regression/acoustic_regression.py",
        "code/utils/config/onb_defaults.py",
        "code/utils/training/model_training.py",
        "code/utils/training/internal_validation.py",
        "code/utils/dataloading/dataloading_and_conversion.py",
        "code/utils/experiment/learning_policy.py",
        "code/utils/experiment/acoustic_selection.py",
    ]
    context = {key: config.get(key, {}) for key in (
        "run", "data", "learning_policy", "acoustic_selection", "thresholds", "features")}
    context["models"] = [{key: value for key, value in spec.items() if key != "builder"}
                         for spec in specs]
    context["model_code_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in code_paths}
    return json.loads(json.dumps(context, default=json_default))


def _candidate_identity(parameter_set):
    # Candidate names, indices and reporting defaults can change after a
    # completed model is reduced to its adopted parameters.
    defaults = {k: v for k, v in parameter_set.get("default_keras", {}).items()
                if k not in {"fit_verbose", "progress_interval_epochs"}}
    return json.dumps({"model_key": parameter_set["model_key"],
                       "models": parameter_set.get("models", {}),
                       "training_defaults": defaults}, sort_keys=True)


def _resume_candidates(manifest_path, context):
    root = Path(__file__).resolve().parents[3]
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest["context"] != context:
        raise ValueError("Tuning resume context differs: data, split, epochs or model code changed")
    return {_candidate_identity(item["parameter_set"]): (root / item["path"], item["sha256"])
            for item in manifest["candidates"]}


def validate_resume_audit(saved, parameter_set, specs, metadata, targets, selector,
                          folds, seed, epochs, method):
    """Require the same training identities, folds and completed fit budget."""
    if _candidate_identity(saved["parameter_set"]) != _candidate_identity(parameter_set):
        raise ValueError("Tuning resume candidate parameters differ")
    audit = saved["internal_validation"]
    if (audit.get("test_used") is not False or audit.get("method") != method
            or audit.get("random_state") != seed or len(audit["folds"]) != folds):
        raise ValueError("Tuning resume internal validation policy differs")
    samples = audit["samples"]
    identity = lambda row: (str(row["experiment_name"]), str(row["source_wav_id"]), int(row["chunk_index"]))
    if ([identity(row) for row in samples] != [identity(row) for row in metadata]
            or not np.array_equal([row["heat_flux"] for row in samples], np.asarray(targets).ravel())):
        raise ValueError("Tuning resume training sample identities or targets differ")
    key = parameter_set["model_key"]
    if not np.isfinite([row[key] for row in samples]).all():
        raise ValueError("Tuning resume predictions are nonfinite")
    for number, ((fit, held), record) in enumerate(zip(
            internal_splits(metadata, folds, seed, method), audit["folds"]), 1):
        retained, _ = selector.select([metadata[i] for i in fit])
        if (record["fold"] != number or record["fit_indices"] != fit[retained].tolist()
                or record["validation_indices"] != held.tolist()
                or record["shared_samples"] != 0):
            raise ValueError("Tuning resume training/validation partition differs")
        if specs[0]["kind"] == "keras":
            training = record["model_training"][key]
            if (training.get("requested_epochs") != epochs
                    or training.get("epochs_completed") != epochs
                    or training.get("requested_batch_size") != specs[0]["batch_size"]
                    or training.get("stopped_by_memory_error")):
                raise ValueError("Tuning resume candidate did not complete the configured training budget")
    return audit


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


def _output_directory(family, config, configured_output_dir=None):
    if configured_output_dir:
        path = Path(configured_output_dir)
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


def run_training_oof_tuning(
        jobs, policy, config, enabled_specs, parameter_sets, trainer,
        output_dir=None, selection_metric=SELECTION_METRIC,
        relative_tolerance=RELATIVE_TOLERANCE):
    """Rank independent model candidates without scoring the outer test set."""
    if policy["training_noise"] != "clean_only":
        raise ValueError(
            "training_oof tuning currently requires clean_only so one candidate ranking "
            "is not silently fitted separately for every noise condition."
        )
    families = build_learning_families(jobs, policy, config["data"]["experiment_names"])
    if len(families) != 1:
        scopes = sorted({
            (
                str(job["experiment_name"]),
                str(job["max_freq_hz"]),
            )
            for family in families
            for job in family["evaluation_jobs"]
        })
        scope_text = ", ".join(
            f"{experiment}/{frequency}" for experiment, frequency in scopes
        )
        raise ValueError(
            "training_oof tuning requires exactly one experiment/frequency scope, "
            f"but found {len(families)}: {scope_text}. "
            "Keep exactly one value in data.max_freq_hz_list and tune one scope at a time."
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
    splits = outer_splits(
        train_metadata,
        evaluation_metadata,
        policy=policy,
    )
    if len(splits) != 1:
        raise ValueError(
            "training_oof tuning needs one fixed outer holdout. Use within_wav_chunk, "
            "or cross_day; do not rank candidates across outer CV folds."
        )
    fit_indices = np.asarray(splits[0][0], dtype=int)
    metadata_fit = [train_metadata[int(index)] for index in fit_indices]
    internal_method = config["run"].get("internal_validation_split", "chunk_kfold")
    internal_units = (len(set(wav_groups(metadata_fit))) if internal_method == "wav_kfold"
                      else len(metadata_fit))
    if internal_units < int(config["run"]["folds"]):
        raise ValueError("The outer-training partition has fewer split units than tuning folds.")

    selector = AcousticTrainingSelector(
        config.get("acoustic_selection"),
        config["thresholds"].get("by_experiment", {}),
    )
    resume_config = config.get("models", {}).get("tuning_resume", {})
    context = tuning_resume_context(config, enabled_specs)
    cached = (_resume_candidates(resume_config["source_manifest"], context)
              if resume_config.get("source_manifest") else {})
    output_dir = _output_directory(family, config, output_dir or resume_config.get("output_dir"))
    if output_dir.exists():
        context_path = output_dir / "resume_context.json"
        if not resume_config or not context_path.is_file():
            raise ValueError("Existing tuning output requires an explicit resume configuration and context")
        if json.loads(context_path.read_text(encoding="utf-8")) != context:
            raise ValueError("Existing tuning output has a different resume context")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(output_dir / "resume_context.json", context)
    audit_dir = output_dir / "candidate_audits"
    audit_dir.mkdir(exist_ok=True)
    # New complete candidates take precedence over the imported run. Atomic
    # writes leave interrupted .tmp files out of this cache.
    for path in sorted(audit_dir.glob("*.json")):
        raw = path.read_bytes()
        saved = json.loads(raw)
        cached[_candidate_identity(saved["parameter_set"])] = (path, hashlib.sha256(raw).hexdigest())
    _write_json(output_dir / "parameter_search_config.json", config)

    metric_name = str(selection_metric)
    allowed_metrics = {"rmse_all", "mae_all", "rmse_onb"}
    if metric_name not in allowed_metrics:
        raise ValueError(f"selection_metric must be one of {sorted(allowed_metrics)}")
    epochs = int(config["run"]["epochs"])
    all_rows, day_rows = [], []
    reused_candidates, trained_candidates = 0, 0
    spec_by_key = {spec["key"]: spec for spec in enabled_specs}
    for candidate_index, parameter_set in enumerate(parameter_sets, 1):
        model_key = parameter_set.get("model_key")
        if not model_key:
            raise ValueError(
                "Each training_oof candidate must select exactly one model via model_key."
            )
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
        source = None
        identity = _candidate_identity(parameter_set)
        if identity in cached:
            source, expected_hash = cached[identity]
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"Tuning resume audit checksum differs: {source}")
            audit = validate_resume_audit(
                json.loads(raw), parameter_set, specs, metadata_fit, y[fit_indices], selector,
                config["run"]["folds"], config["run"]["random_seed"], epochs, internal_method)
            reused_candidates += 1
            print(f"  reused completed OOF candidate: {source.name}", flush=True)
        else:
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
                method=internal_method,
            )
            trained_candidates += 1
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
            "reused_from": str(source) if source else None,
        })

    _write_csv(output_dir / "candidate_metrics.csv", all_rows)
    _write_csv(output_dir / "candidate_metrics_by_source_day.csv", day_rows)
    tolerance = float(relative_tolerance)
    if not 0 <= tolerance < 1:
        raise ValueError("relative_tolerance must be between 0 and 1.")
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
        "candidate_ranking_scope": f"outer-training partition, {internal_method} OOF",
        "internal_validation_split": internal_method,
        "evaluation_scheme": split_kind,
        "n_outer_training_samples": int(len(fit_indices)),
        "n_outer_test_samples_not_scored": int(len(splits[0][1])),
        "folds": int(config["run"]["folds"]),
        "selection_metric": metric_name,
        "relative_tolerance": tolerance,
        "reused_candidates": reused_candidates,
        "trained_candidates": trained_candidates,
        "selected": selected,
        "normal_run_unchanged": True,
        "next_step": (
            "Copy the chosen parameters into models.parameter_sets.by_max_freq_hz"
            f"['{config.get('models', {}).get('active_max_freq_hz', evaluation_job['max_freq_hz'])}'], "
            "leave exactly one value in every candidate list, and run the normal pipeline "
            "once for the untouched outer test."
        ),
    }
    _write_json(output_dir / "parameter_search_config.json", config)
    _write_json(output_dir / "selected_candidates.json", summary)
    _write_json(output_dir / "completed.json", {"completed": True, **summary})
    print(f"training-only OOF tuning results saved: {output_dir}", flush=True)
    return {"output_dir": output_dir, **summary}
