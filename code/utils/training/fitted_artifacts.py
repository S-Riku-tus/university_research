"""Persist final fitted regression state and verify it can be reloaded."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np

from utils.experiment.run_helpers import json_default, makedirs, open_text, windows_long_path
from utils.training.model_training import PCA_FEATURE_DECIMALS


def _file_digest(path):
    digest = hashlib.sha256()
    with open(windows_long_path(path), "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path, root):
    path = Path(path)
    return {
        "path": os.path.relpath(path, root),
        "bytes": os.path.getsize(windows_long_path(path)),
        "sha256": _file_digest(path),
    }


def environment_snapshot():
    packages = {}
    for name in ("numpy", "scikit-learn", "tensorflow", "xgboost", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tensorflow_build = None
    tensorflow_devices = []
    try:
        import tensorflow as tf

        tensorflow_build = tf.sysconfig.get_build_info()
        tensorflow_devices = [
            device.name for device in tf.config.list_physical_devices()
        ]
    except Exception:
        # Package versions are still useful if device enumeration is not
        # available in a CPU-only inspection process.
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "tensorflow_build": tensorflow_build,
        "tensorflow_physical_devices": tensorflow_devices,
        "determinism_environment": {
            key: os.environ.get(key)
            for key in (
                "TF_DETERMINISTIC_OPS",
                "TF_CUDNN_DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def begin_fitted_state(
    directory,
    fit_id,
    fold,
    random_seed,
    training_wav_groups,
    selected_epochs,
    pca,
    scaler,
    deterministic_ops=True,
):
    directory = Path(directory)
    makedirs(directory)
    manifest = {
        "schema_version": 1,
        "fit_id": fit_id,
        "fold": int(fold),
        "random_seed": int(random_seed),
        "deterministic_ops": bool(deterministic_ops),
        "training_wav_groups": sorted({str(value) for value in training_wav_groups}),
        "selected_epochs": {key: int(value) for key, value in selected_epochs.items()},
        "pca_feature_decimals": PCA_FEATURE_DECIMALS if pca is not None else None,
        "environment": environment_snapshot(),
        "preprocessors": {},
        "models": {},
        "ensemble_weights": {},
    }
    scaler_path = directory / "target_scaler.joblib"
    joblib.dump(scaler, windows_long_path(scaler_path))
    loaded_scaler = joblib.load(windows_long_path(scaler_path))
    if not np.allclose(loaded_scaler.data_min_, scaler.data_min_) or not np.allclose(
        loaded_scaler.data_max_, scaler.data_max_
    ):
        raise RuntimeError("Reloaded target scaler does not match the fitted scaler.")
    manifest["preprocessors"]["target_scaler"] = _file_record(scaler_path, directory)
    if pca is not None:
        pca_path = directory / "pca.joblib"
        joblib.dump(pca, windows_long_path(pca_path))
        loaded_pca = joblib.load(windows_long_path(pca_path))
        if not np.allclose(loaded_pca.components_, pca.components_):
            raise RuntimeError("Reloaded PCA does not match the fitted PCA.")
        manifest["preprocessors"]["pca"] = _file_record(pca_path, directory)
    return manifest


def save_and_verify_model(
    directory,
    manifest,
    spec,
    model,
    model_maker,
    trainer,
    x_probe,
    scaler,
    pca,
    verify=True,
):
    directory = Path(directory)
    key = spec["key"]
    loaded_scaler = joblib.load(windows_long_path(directory / "target_scaler.joblib"))
    loaded_pca = None
    x_probe_pca = None
    if spec["kind"] != "keras":
        loaded_pca = joblib.load(windows_long_path(directory / "pca.joblib"))
        x_probe_pca = trainer.transform_pca(loaded_pca, x_probe)

    if spec["kind"] == "keras":
        path = directory / f"{key}.weights.h5"
        model.save_weights(windows_long_path(path))
        loaded_model = spec["builder"](model_maker, **spec.get("builder_params", {}))
        loaded_model.load_weights(windows_long_path(path))
    else:
        # Persist the complete sklearn wrapper as well as its booster state.
        # XGBoost's raw model format can omit wrapper attributes that affect
        # sklearn-level prediction in some installed versions.
        path = directory / f"{key}.joblib"
        joblib.dump(model, windows_long_path(path))
        loaded_model = joblib.load(windows_long_path(path))

    verification = {"performed": bool(verify)}
    if verify:
        original_pca = None
        if spec["kind"] != "keras":
            original_pca = trainer.transform_pca(pca, x_probe)
        original = trainer.predict_one_model(
            spec, model, x_probe, original_pca, scaler
        )
        restored = trainer.predict_one_model(
            spec, loaded_model, x_probe, x_probe_pca, loaded_scaler
        )
        difference = np.abs(np.asarray(restored) - np.asarray(original))
        verification.update(
            {
                "probe_samples": int(len(x_probe)),
                "max_abs_prediction_difference_w_m2": float(np.max(difference)),
                "mean_abs_prediction_difference_w_m2": float(np.mean(difference)),
                "allclose_rtol": 1e-6,
                "allclose_atol_w_m2": 1e-3,
                "passed": bool(np.allclose(restored, original, rtol=1e-6, atol=1e-3)),
            }
        )
        if not verification["passed"]:
            raise RuntimeError(
                f"Reloaded model prediction mismatch: {key}; "
                f"max_abs={verification['max_abs_prediction_difference_w_m2']:.9g} W/m2, "
                f"mean_abs={verification['mean_abs_prediction_difference_w_m2']:.9g} W/m2"
            )

    manifest["models"][key] = {
        "kind": spec["kind"],
        "builder_params": spec.get("builder_params", {}),
        "fitted_random_state": (
            model.get_params().get("random_state")
            if spec["kind"] != "keras" and hasattr(model, "get_params")
            else manifest["random_seed"]
        ),
        "artifact": _file_record(path, directory),
        "verification": verification,
    }
    del loaded_model
    return manifest["models"][key]


def finish_fitted_state(directory, manifest, ensemble_outputs):
    for result_key, output in (ensemble_outputs or {}).items():
        manifest["ensemble_weights"][result_key] = {
            "strategy_name": output["strategy"]["name"],
            "strategy_type": output["strategy"]["strategy"],
            "weights": {key: float(value) for key, value in output["weights"].items()},
        }
    weights_path = Path(directory) / "ensemble_weights.json"
    with open_text(weights_path, "w", encoding="utf-8") as output:
        json.dump(
            manifest["ensemble_weights"], output, ensure_ascii=False, indent=2,
            default=json_default,
        )
    manifest["ensemble_weights_file"] = _file_record(weights_path, directory)
    manifest_path = Path(directory) / "artifact_manifest.json"
    with open_text(manifest_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2, default=json_default)
    return manifest_path
