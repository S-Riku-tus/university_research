"""Preflight and collect the fixed clean-only ONB comparison without fitting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.calculation.onb_comparison_report import _csv, _json, noise_key, write_comparison_report
from utils.calculation.prediction_records import load_sample_metadata_without_arrays
from utils.calculation.source_day_metrics import source_day_metric_rows, source_experiment
from utils.experiment.learning_policy import checked_metadata, outer_splits
from utils.experiment.run_helpers import makedirs, open_text, windows_long_path


def _read_json(path):
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path):
    with open_text(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _require(condition, message):
    if not condition:
        raise RuntimeError("ONB main comparison: "+message)


def _fingerprint(metadata, indices):
    rows = sorted((source_experiment(metadata[int(i)]), str(metadata[int(i)]["source_wav_id"]),
                   int(metadata[int(i)]["chunk_index"]), float(metadata[int(i)]["heat_flux"])) for i in indices)
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def standard_configuration(config, protocol):
    expected = protocol["standard_run"]
    run, data, policy = config["run"], config["data"], config["learning_policy"]
    parameters = config["models"]["parameter_sets"]
    parameter_match = len(parameters) == 1 and parameters[0].get("models") == protocol["standard_model_parameters"]
    return bool(not run["smoke_test"] and run["epochs"] == expected["epochs"]
        and run["folds"] == expected["internal_folds"] and run["random_seed"] == expected["random_seed"]
        and run["cpu_threads"] == expected["cpu_threads"]
        and run["internal_validation_split"] == expected["internal_split"]
        and data["chunk_seconds"] == expected["chunk_seconds"]
        and data["max_freq_hz_list"] == [expected["max_freq_hz"]]
        and data["noise_source"] == "waterflow" and run["color_channel"] == expected["color_channels"]
        and policy["training_noise"] == "clean_only" and policy.get("evaluation_mode") == "within_wav_chunk"
        and policy.get("test_fraction") == .25 and policy.get("test_split_seed") == expected["random_seed"]
        and config.get("acoustic_selection", {}).get("peak_height_threshold") is None
        and config["features"]["pca_components"] == protocol["pca_components"] and parameter_match
        and set(config["models"]["fixed_model_keys"]) == set(protocol["model_keys"])
        and [noise_key(n) for n in data["noise_dir_names"]] == protocol["noise_conditions"])


def validate_main_comparison_jobs(jobs, config):
    """Read only manifests/filenames; reject missing conditions before CNN fits."""
    protocol = config["output"]["main_comparison_protocol"]
    expected_jobs = {(e, f, n) for e in config["data"]["experiment_names"]
                     for f in config["data"]["max_freq_hz_list"] for n in config["data"]["noise_dir_names"]}
    observed_jobs = [(j["experiment_name"], j["max_freq_hz"], j["noise_dir_name"]) for j in jobs]
    _require(len(observed_jobs) == len(set(observed_jobs)) and set(observed_jobs) == expected_jobs,
             f"configured evaluation datasets are incomplete; missing={sorted(expected_jobs-set(observed_jobs))}")
    matches = standard_configuration(config, protocol)
    dataset_rows, fingerprints = [], {}
    for job in jobs:
        metadata = checked_metadata(load_sample_metadata_without_arrays(job["data_path"]), job["experiment_name"])
        fit, test = outer_splits(metadata, metadata, policy=config["learning_policy"])[0]
        _require(len(np.intersect1d(fit, test)) == 0, "training and evaluation chunks overlap")
        for day in {source_experiment(r) for r in metadata}:
            _require(day in protocol["source_day_thresholds_W_m2"] and np.isclose(
                config["thresholds"]["by_experiment"][day], protocol["source_day_thresholds_W_m2"][day], rtol=0, atol=1e-8),
                f"source-day threshold differs from the evaluation protocol: {day}")
        fingerprint = _fingerprint(metadata, test)
        unit = (job["experiment_name"], job["max_freq_hz"])
        _require(unit not in fingerprints or fingerprints[unit] == fingerprint, "noise conditions do not share evaluation identities/targets")
        fingerprints[unit] = fingerprint
        wavs = {r["source_wav_id"] for r in metadata}
        if matches:
            expected = protocol["standard_run"]
            _require(len(wavs) == expected["source_wavs"] and len(fit) == expected["train_chunks"]
                and len(test) == expected["evaluation_chunks"]
                and all(sum(r["source_wav_id"] == w for r in metadata) == expected["chunks_per_wav"] for w in wavs),
                "data counts do not match the standard 36-WAV / 1620-train / 540-evaluation protocol")
            _require(all(len({r["source_wav_id"] for r in metadata if source_experiment(r) == day}) == 18
                         for day in protocol["source_day_thresholds_W_m2"]), "each source day must have 18 WAVs")
            _require(all(sum(metadata[int(i)]["source_wav_id"] == w for i in fit) == 45
                         and sum(metadata[int(i)]["source_wav_id"] == w for i in test) == 15 for w in wavs),
                     "each WAV must have 45 training and 15 evaluation chunks")
        dataset_rows.append({"noise_dir_name": job["noise_dir_name"], "data_path": str(job["data_path"]),
            "source_wavs": len(wavs), "input_chunks": len(metadata), "training_chunks": len(fit),
            "evaluation_chunks": len(test), "evaluation_identity_sha256": fingerprint})
    return {"status": "passed", "standard_configuration": matches, "datasets": dataset_rows,
            "configured_noise_conditions": [noise_key(n) for n in config["data"]["noise_dir_names"]]}


def export_main_comparison(recorders, config):
    directory = recorders[0].path.parent / "onb_comparison"
    try:
        result = _export(recorders, config, directory)
    except Exception as error:
        makedirs(directory)
        _json(directory/"verification.json", {"status": "failed", "error": str(error)})
        raise
    print(f"ONBの主比較表と実行確認を保存: {result}", flush=True)
    return result


def _export(recorders, config, directory):
    protocol = config["output"]["main_comparison_protocol"]
    _require(config["learning_policy"]["training_noise"] == "clean_only", "main comparison collection requires clean_only")
    _require(len({r.run_hash for r in recorders}) == 1, "evaluation conditions have different run hashes")
    _require({noise_key(r.snr) for r in recorders} == {noise_key(n) for n in config["data"]["noise_dir_names"]},
             "configured noise conditions are incomplete")
    rows, weight_rows, reload_rows, epoch_rows, ensemble_checks, sources = [], [], [], [], [], []
    evaluation_identities, artifact_paths, fit_ids = {}, set(), set()
    epoch_complete, batch_sizes_match = True, True
    for recorder in recorders:
        _require(Path(windows_long_path(recorder.path/"completed.json")).exists(), "an evaluation condition is incomplete")
        split = _read_json(recorder.path/"split_manifest.json")
        saved_metrics = _read_csv(recorder.path/"metrics_by_source_day.csv")
        for row in saved_metrics:
            rows.append({"seed": config["run"]["random_seed"], "noise": recorder.snr,
                         "source_result": str(recorder.path), **row})
        metadata = checked_metadata(load_sample_metadata_without_arrays(recorder.job["data_path"]), recorder.job["experiment_name"])
        for fold_record in split["folds"]:
            fold = int(fold_record["fold"])
            fit_ids.add(fold_record["fit_id"])
            pred_path = recorder.path/"fold_pred"/f"pred_f{fold}_{recorder.snr}.csv"
            predictions = _read_csv(pred_path)
            indices = np.asarray([int(r["sample_index"]) for r in predictions])
            _require(indices.tolist() == fold_record["evaluation_sample_indices"], "saved prediction and split indices differ")
            fingerprint = _fingerprint(metadata, indices)
            _require(fold not in evaluation_identities or evaluation_identities[fold] == fingerprint,
                     "evaluation chunks or targets differ across noise")
            evaluation_identities[fold] = fingerprint
            targets = np.asarray([float(r["y_true"]) for r in predictions])
            np.testing.assert_allclose(targets, [float(metadata[i]["heat_flux"]) for i in indices], rtol=0, atol=1e-8)
            reference = _read_json(recorder.path/f"fitted_state_reference_f{fold}.json")
            artifact_dir = Path(reference["canonical_directory"])
            artifact = _read_json(artifact_dir/"artifact_manifest.json")
            _require(artifact["fit_id"] == fold_record["fit_id"], "saved fitted state and prediction fit IDs differ")
            _require(set(artifact["models"]) == set(protocol["model_keys"]), "saved five-model state is incomplete")
            _require(set(artifact["ensemble_weights"]) == {"ensemble__"+s for s in protocol["ensemble_names"]}, "saved comparison weights are incomplete")
            columns = protocol["model_keys"] + list(artifact["ensemble_weights"])
            matrix = {key: np.asarray([float(r[key]) for r in predictions]) for key in columns}
            recalculated = source_day_metric_rows(targets, matrix, [metadata[i] for i in indices],
                config["thresholds"]["by_experiment"], fold, band_frac=config["thresholds"]["onb_band_frac"])
            saved_lookup = {(int(r["fold"]), r["model_key"], r["source_day"]): r for r in saved_metrics}
            for computed in recalculated:
                before = saved_lookup[fold, computed["model_key"], computed["source_day"]]
                for key, value in computed.items():
                    if isinstance(value, (float, int, np.integer)):
                        np.testing.assert_allclose(float(before[key]), value, rtol=1e-10, atol=1e-6, equal_nan=True)
            for key, item in artifact["ensemble_weights"].items():
                w = item["weights"]
                restored = sum(matrix[k]*float(w[k]) for k in protocol["model_keys"])
                np.testing.assert_allclose(restored, matrix[key], rtol=1e-6, atol=1e-3)
                ensemble_checks.append({"noise": noise_key(recorder.snr), "fold": fold, "model_key": key,
                    "samples": len(targets), "max_abs_difference_W_m2": float(np.max(np.abs(restored-matrix[key]))), "passed": True})
            if artifact_dir not in artifact_paths:
                artifact_paths.add(artifact_dir)
                core = artifact["ensemble_weights"][protocol["baseline_model_key"]]["weights"]
                for key, item in artifact["ensemble_weights"].items():
                    w = item["weights"]
                    _require(np.isclose(sum(w.values()), 1, rtol=0, atol=1e-8) and all(v >= 0 for v in w.values()), "invalid saved convex weights")
                    if key in {protocol["primary_model_key"], "ensemble__original3_hgb", "ensemble__original3_extra_trees"}:
                        core_total = sum(w[k] for k in protocol["model_keys"][:3])
                        _require(core_total >= .25-1e-8 and all(w[k] > 0 for k in protocol["model_keys"][:3]), "original-three retention constraint failed")
                        np.testing.assert_allclose([w[k]/core_total for k in protocol["model_keys"][:3]], [core[k] for k in protocol["model_keys"][:3]], atol=1e-8)
                        additions = [k for k in protocol["model_keys"][3:] if w[k] > 0]
                        _require(all(w[k] >= .05-1e-8 for k in additions), "added-model retention constraint failed")
                        _require(len(additions) == (2 if key == protocol["primary_model_key"] else 1), "wrong active member count")
                    weight_rows.extend({"seed": config["run"]["random_seed"], "fold": fold,
                        "strategy_name": item["strategy_name"], "model_key": k, "weight": v,
                        "fit_scope": "clean training OOF; fixed across all noise"} for k, v in w.items())
                for key, item in artifact["models"].items():
                    _require(item["verification"].get("passed", False), f"probe reload verification absent/failed: {key}")
                    evaluated = item.get("evaluation_verification", {})
                    _require(evaluated.get("passed", False), f"full evaluation reload verification absent/failed: {key}")
                    _require({r["noise_dir_name"] for r in evaluated["conditions"]} == {r.job["noise_dir_name"] for r in recorders}, f"reload noise coverage is incomplete: {key}")
                    for check in evaluated["conditions"]:
                        _require(check["passed"] and check["samples"] == len(targets), f"reload sample coverage failed: {key}")
                        reload_rows.append({"fold": fold, "model_key": key, **check})
                    if item["kind"] == "keras":
                        history = item["training_history"]
                        complete = history["epochs_completed"] == config["run"]["epochs"] and not history.get("stopped_by_memory_error", False)
                        epoch_complete &= complete
                        batch_sizes_match &= history["actual_batch_size"] == history["requested_batch_size"]
                        epoch_rows.append({"stage": "final", "fold": fold, "model_key": key, "requested_epochs": config["run"]["epochs"], **history, "complete": complete})
                audit = _read_json(recorder.path/f"internal_validation_fold{fold}.json")
                _require(audit["test_used"] is False, "outer test was used for internal weighting")
                _require(audit["method"] == config["run"]["internal_validation_split"]
                         and len(audit["folds"]) == config["run"]["folds"], "internal split or fold count differs from configuration")
                keras_keys = {k for k, item in artifact["models"].items() if item["kind"] == "keras"}
                coverage = np.zeros(len(audit["samples"]), dtype=int)
                for inner in audit["folds"]:
                    _require(set(inner["model_training"]) == keras_keys, "internal CNN training-history coverage is incomplete")
                    _require(not set(inner["fit_indices"]) & set(inner["validation_indices"]), "internal fit/held chunks overlap")
                    np.add.at(coverage, inner["validation_indices"], 1)
                    for key, history in inner["model_training"].items():
                        complete = history["epochs_completed"] == config["run"]["epochs"] and not history["stopped_by_memory_error"]
                        epoch_complete &= complete
                        batch_sizes_match &= history["actual_batch_size"] == history["requested_batch_size"]
                        epoch_rows.append({"stage": "internal", "fold": inner["fold"], "model_key": key, **history, "complete": complete})
                _require(np.all(coverage == 1), "internal OOF coverage is not exactly one per training chunk")
            sources.append({"noise_dir_name": recorder.job["noise_dir_name"], "fold": fold,
                "run_directory": str(recorder.path), "prediction_csv": str(pred_path), "fit_id": fold_record["fit_id"],
                "fitted_state": str(artifact_dir), "evaluation_chunks": len(indices), "evaluation_identity_sha256": fingerprint})
    _require(len(fit_ids) == 1, "clean-only evaluation conditions use different fits")
    verification = {"status": "passed" if epoch_complete else "incomplete_epochs", "normal_150_epoch_main_run_completed":
        bool(epoch_complete and batch_sizes_match and standard_configuration(config, protocol)), "epochs_complete": bool(epoch_complete),
        "requested_batch_sizes_completed": bool(batch_sizes_match),
        "standard_configuration": standard_configuration(config, protocol), "smoke_test": config["run"]["smoke_test"],
        "noise_conditions": [noise_key(r.snr) for r in recorders], "source_day_metric_rows": len(rows),
        "fitted_models": len(reload_rows)//len(recorders), "reloaded_evaluation_predictions": sum(r["samples"] for r in reload_rows),
        "same_clean_fit_across_noise": True, "same_evaluation_chunks_and_targets": True,
        "internal_oof_coverage_and_no_shared_chunks": True, "saved_weights_and_ensemble_predictions_verified": True,
        "performance_improvement_required_for_completion": False, "sources": sources}
    verification["scope"] = ("standard seed42/150-epoch known-recording development evaluation" if verification["normal_150_epoch_main_run_completed"]
                             else "software smoke or nonstandard configuration; not the standard 150-epoch research result")
    result = write_comparison_report(directory, rows, protocol, config, weight_rows, verification)
    _csv(directory/"fitted_reload_checks.csv", reload_rows)
    _csv(directory/"ensemble_reconstruction_checks.csv", ensemble_checks)
    _csv(directory/"fit_epoch_audit.csv", epoch_rows)
    if not epoch_complete:
        print("ONB主比較は保存済みですが、要求epochを完了していない学習があります。fit_epoch_audit.csvを確認してください。", flush=True)
    return result
