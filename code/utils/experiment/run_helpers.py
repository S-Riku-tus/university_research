import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


WEIGHT_STRATEGY_TAGS = {
    "simple": "simp",
    "fixed": "fix",
    "inner_holdout": "ih",
    "val_fold_legacy": "vleg",
<<<<<<< HEAD
    "strategy_loop": "cmp",
=======
>>>>>>> 51979ea46b47fa367df94150a3b3739b1f36b65e
}


def format_param_value(value):
    if isinstance(value, float):
        text = f"{value:.8f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def safe_tag(text, max_len=32):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")[:max_len] or "params"


def snr_value_from_noise_dir(noise_dir_name):
    if noise_dir_name == "heatflux_no_noise":
        return "no_noise"
    if "SNR=" in noise_dir_name:
        return noise_dir_name.split("SNR=", 1)[1]
    return safe_tag(noise_dir_name)


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def has_threshold(threshold):
    if threshold is None:
        return False
    try:
        return np.isfinite(float(threshold))
    except (TypeError, ValueError):
        return False


def short_digest(payload, length=8):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def compact_weight_strategy_tag(weight_strategy):
    return WEIGHT_STRATEGY_TAGS.get(weight_strategy, safe_tag(weight_strategy, max_len=8))


def model_param_summary(resolved_specs):
    summary = {}
    for spec in resolved_specs:
        if spec["kind"] == "keras":
            item = {
                "lr": spec["lr"],
                "batch_size": spec["batch_size"],
            }
            if spec.get("builder_params"):
                item["builder_params"] = spec["builder_params"]
            if spec.get("input_axes_assumption"):
                item["input_axes_assumption"] = spec["input_axes_assumption"]
            if spec.get("actual_npy_axes"):
                item["actual_npy_axes"] = spec["actual_npy_axes"]
            if spec.get("architecture"):
                item["architecture"] = spec["architecture"]
            if spec.get("note"):
                item["note"] = spec["note"]
            summary[spec["key"]] = item
        else:
            summary[spec["key"]] = {
                "kind": spec["kind"],
                "params": spec.get("builder_params", {}),
            }
    return summary


def serializable_run_specs(run_specs):
    return [
        {
            key: value
            for key, value in spec.items()
            if key not in {"builder"}
        }
        for spec in run_specs
    ]


def run_config_digest(validation_config, parameter_set, run_specs, model_tag, save_fold_predictions):
    config = dict(validation_config)
    config["output"] = {
        "save_fold_predictions": save_fold_predictions,
    }
    return short_digest({
        "validation_config": config,
        "parameter_set": parameter_set,
        "model_tag": model_tag,
        "model_params": model_param_summary(run_specs),
    }, length=6)


def run_dir_name(epoch_num, param_tag, model_tag, ensemble_enabled, weight_strategy):
    parts = [
        f"e{epoch_num}",
        safe_tag(param_tag, max_len=24),
        safe_tag(model_tag, max_len=12),
    ]
    if ensemble_enabled:
        parts.append(compact_weight_strategy_tag(weight_strategy))
    return "_".join(parts)


def write_run_manifest(
    save_path,
    job,
    parameter_set,
    run_specs,
    param_tag,
    model_tag,
    run_hash,
    run_dir,
    run_instance_id,
    validation_config,
):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_instance_id": run_instance_id,
        "run_hash": run_hash,
        "run_dir": run_dir,
        "folder_naming": {
            "scheme": "e{epochs}_{param}_{models}[_{weight_when_ensemble_enabled}]",
            "reason": "Keep Windows paths short while keeping parameter folders readable.",
            "details": "Full conditions are stored in this manifest and validation_results_*.txt.",
        },
        "dataset": {
            "experiment_name": job["experiment_name"],
            "source_dir": str(job["source_dir"]),
            "data_path": str(job["data_path"]),
            "max_freq_hz": job["max_freq_hz"],
            "noise_dir_name": job["noise_dir_name"],
            "snr_value": job["snr_value"],
            "threshold": job["threshold"],
        },
        "parameter_set_name": parameter_set.get("name"),
        "parameter_set_tag": param_tag,
        "model_tag": model_tag,
        "model_params": model_param_summary(run_specs),
        "run_specs": serializable_run_specs(run_specs),
        "validation_config": validation_config,
    }
    manifest_path = os.path.join(save_path, "run_manifest.json")
    with open_text(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2, default=json_default)


def windows_long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        os.makedirs(windows_long_path(path), exist_ok=True)


def path_exists(path):
    return os.path.exists(path) or os.path.exists(windows_long_path(path))


def open_text(path, mode, **kwargs):
    try:
        return open(path, mode, **kwargs)
    except OSError:
        return open(windows_long_path(path), mode, **kwargs)


def csv_metric(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(value):
        return ""
    return f"{value:.10g}"


def join_unique(values):
    clean = [str(v) for v in values if v not in (None, "")]
    return "|".join(dict.fromkeys(clean))


def set_global_seed(seed):
    """Keep KFold, sklearn, and Keras runs as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def append_tuning_summary(
    summary_path,
    job,
    parameter_set,
    run_specs,
    store,
    train_meta,
    summary_metrics,
    metrics,
    param_tag,
    run_dir,
    run_hash,
    save_path,
    model_keys,
    save_tuning_summary,
    run_instance_id,
):
    if not save_tuning_summary:
        return

    makedirs(os.path.dirname(summary_path))
    header = [
        "created_at", "run_instance_id", "run_hash", "run_dir", "save_path",
        "experiment_name", "data_source_dir", "max_freq_hz", "noise_dir_name",
        "snr_value", "threshold_available", "threshold",
        "parameter_set", "model_key", "model_label",
        "lr", "batch_size",
        "requested_batch_size", "actual_batch_sizes", "min_actual_batch_size",
        "epochs_completed", "stopped_by_memory_error",
    ]
    for metric_name in summary_metrics:
        header.extend([f"{metric_name}_mean", f"{metric_name}_se"])

    spec_by_key = {spec["key"]: spec for spec in run_specs}
    file_exists = path_exists(summary_path)
    with open_text(summary_path, "a", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        if not file_exists:
            writer.writerow(header)
        for key in model_keys:
            spec = spec_by_key.get(key, {})
            meta = train_meta.get(key, {})
            actual_batch_sizes = meta.get("actual_batch_size", [])
            min_actual_batch_size = min(actual_batch_sizes) if actual_batch_sizes else ""
            stopped_flags = meta.get("stopped_by_memory_error", [])
            row = [
                datetime.now().isoformat(timespec="seconds"),
                run_instance_id,
                run_hash,
                run_dir,
                save_path,
                job["experiment_name"],
                job["source_dir"],
                job["max_freq_hz"],
                job["noise_dir_name"],
                job["snr_value"],
                int(has_threshold(job["threshold"])),
                job["threshold"] if has_threshold(job["threshold"]) else "",
                parameter_set.get("name", param_tag),
                key,
                spec.get("label", key),
                spec.get("lr", ""),
                spec.get("batch_size", ""),
                join_unique(meta.get("requested_batch_size", [])),
                join_unique(actual_batch_sizes),
                min_actual_batch_size,
                join_unique(meta.get("epochs_completed", [])),
                int(any(bool(flag) for flag in stopped_flags)),
            ]
            for metric_name in summary_metrics:
                mean, se = metrics.mean_se(store[key][metric_name])
                row.extend([csv_metric(mean), csv_metric(se)])
            writer.writerow(row)


<<<<<<< HEAD
def is_completed_run(summary_path, run_dir, save_path, snr_value,
                     resume_completed_runs, save_tuning_summary,
                     run_hash=None):
=======
def is_completed_run(summary_path, run_dir, save_path, snr_value, resume_completed_runs, save_tuning_summary):
>>>>>>> 51979ea46b47fa367df94150a3b3739b1f36b65e
    """Return True only when the per-run metrics and tuning summary both exist."""
    if not resume_completed_runs:
        return False

    metrics_path = os.path.join(save_path, f"metrics_summary_{snr_value}.csv")
    if not path_exists(metrics_path):
        return False

    if not save_tuning_summary:
        return True
    if not path_exists(summary_path):
        return False

    try:
        with open_text(summary_path, "r", newline="", encoding="utf-8") as sf:
            for row in csv.DictReader(sf):
<<<<<<< HEAD
                same_directory = row.get("run_dir") == run_dir
                same_config = run_hash is None or row.get("run_hash") == run_hash
                if same_directory and same_config:
=======
                if row.get("run_dir") == run_dir:
>>>>>>> 51979ea46b47fa367df94150a3b3739b1f36b65e
                    return True
    except (OSError, csv.Error):
        return False
    return False
