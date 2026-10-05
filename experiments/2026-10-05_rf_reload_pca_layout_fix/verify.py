"""Reproduce the failed run's PCA layout error without retraining its models."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import run_ensemble_regression_onb as onb
from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.experiment.learning_policy import checked_metadata, outer_splits
from utils.experiment.run_helpers import run_config_digest, windows_long_path
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.fitted_artifacts import begin_fitted_state, save_and_verify_model
from utils.training.model_training import ModelTrainer, PCA_FEATURE_DECIMALS


def read_json(path):
    return json.loads(Path(windows_long_path(path)).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(windows_long_path(path)).read_bytes()).hexdigest()


def main():
    condition = read_json(ROOT / "configs/experiments/2026-10-05_rf_reload_pca_layout_fix.json")
    source = ROOT / condition["source_failed_run"]
    state = source / "fitted_state/fold1"
    filenames = ("pca.joblib", "randomforest.joblib", "target_scaler.joblib")
    source_digests = {filename: digest(state / filename) for filename in filenames}
    manifest = read_json(source / "run_manifest.json")
    dataset = manifest["dataset"]
    metadata = checked_metadata(load_sample_metadata_without_arrays(dataset["data_path"]), dataset["experiment_name"])
    fit, _ = outer_splits(metadata, metadata, manifest["validation_config"]["learning_policy"])[0]
    x_probe, _ = DataLoadingConversion().load_npy_data(dataset["data_path"], sample_indices=fit[:8])
    pca = joblib.load(windows_long_path(state / "pca.joblib"))
    model = joblib.load(windows_long_path(state / "randomforest.joblib"))
    scaler = joblib.load(windows_long_path(state / "target_scaler.joblib"))

    # Recreate the padded Fortran view returned by randomized SVD before
    # joblib converts it to a contiguous array. Values and dtype are equal.
    pca_strided = copy.copy(pca)
    buffer = np.empty((pca.components_.shape[0] + 10, pca.components_.shape[1]), dtype=pca.components_.dtype, order="F")
    buffer[: pca.components_.shape[0]] = pca.components_
    pca_strided.components_ = buffer[: pca.components_.shape[0]]
    np.testing.assert_array_equal(pca_strided.components_, pca.components_)
    flat = np.ascontiguousarray(x_probe.reshape(len(x_probe), -1))
    old_strided = np.round(pca_strided.transform(flat), decimals=PCA_FEATURE_DECIMALS)
    old_contiguous = np.round(pca.transform(flat), decimals=PCA_FEATURE_DECIMALS)
    old_predictions = [scaler.inverse_transform(model.predict(features).reshape(-1, 1)).ravel() for features in (old_strided, old_contiguous)]
    difference = np.abs(old_predictions[0] - old_predictions[1])
    assert np.isclose(difference.max(), condition["reported_max_difference_W_m2"], atol=1e-3, rtol=0)

    trainer = ModelTrainer(42)
    fixed_strided = trainer.transform_pca(pca_strided, x_probe)
    fixed_contiguous = trainer.transform_pca(pca, x_probe)
    np.testing.assert_array_equal(fixed_strided, fixed_contiguous)
    fixed_predictions = [scaler.inverse_transform(model.predict(features).reshape(-1, 1)).ravel() for features in (fixed_strided, fixed_contiguous)]
    np.testing.assert_array_equal(*fixed_predictions)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp) / "fitted_state"
        state_manifest = begin_fitted_state(directory, "failed-run-probe", 1, 43, [], {}, pca_strided, scaler)
        spec = next(spec for spec in manifest["run_specs"] if spec["key"] == "randomforest")
        spec["builder"] = onb.MODEL_SPECS[0]["builder"]
        verification = save_and_verify_model(directory, state_manifest, spec, model, RegressionModelMaker(tuple(x_probe.shape[1:])), trainer, x_probe, scaler, pca_strided, verify=True)["verification"]
    assert verification["passed"] and verification["max_abs_prediction_difference_w_m2"] == 0

    onb.validate_validation_config(onb.MODEL_SPECS)
    config = onb.validation_config_snapshot("maxfreq=3kHz")
    parameter_set = onb.parameter_plan_for_max_freq("maxfreq=3kHz")["parameter_sets"][0]
    specs = onb.resolve_parameter_set(onb.MODEL_SPECS, parameter_set)
    new_hash = run_config_digest(config, parameter_set, specs, "-".join(onb.MODEL_KEYS), True)
    assert new_hash != manifest["run_hash"]
    for filename in filenames:
        assert digest(state / filename) == source_digests[filename]
    audit = read_json(source / "internal_validation_fold1.json")
    result = {
        "source_failed_run": condition["source_failed_run"],
        "source_artifacts_sha256": source_digests,
        "probe_outer_training_indices": fit[:8].tolist(),
        "legacy_layout_changed_feature_count": int(np.count_nonzero(old_strided != old_contiguous)),
        "legacy_max_prediction_difference_W_m2": float(difference.max()),
        "legacy_mean_prediction_difference_W_m2": float(difference.mean()),
        "fixed_feature_arrays_identical": True,
        "actual_saved_model_reload_verification": verification,
        "pca_transform_version": config["features"]["pca_transform_version"],
        "old_run_hash": manifest["run_hash"],
        "fixed_run_hash": new_hash,
        "run_identity_changed": True,
        "original_run_overwritten": False,
        "source_run_complete": Path(windows_long_path(source / "completed.json")).is_file(),
        "source_internal_audit_keys": list(audit),
        "research_model_retraining": False,
        "q100_improvement_verified": False,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("legacy_max_prediction_difference_W_m2", "legacy_mean_prediction_difference_W_m2", "fixed_feature_arrays_identical", "actual_saved_model_reload_verification", "run_identity_changed", "original_run_overwritten")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
