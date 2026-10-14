"""Apply one saved clean fit to an unseen day without fitting any state."""
import csv
import gc
import json
from pathlib import Path

import joblib
import numpy as np

from utils.calculation.prediction_records import (
    build_fold_prediction_rows, load_sample_metadata_without_arrays, write_fold_prediction_csv,
)
from utils.calculation.source_day_metrics import source_day_metric_rows, source_experiment
from utils.calculation.onb_comparison_report import comparison_tables, noise_key, _csv
from utils.config.onb_defaults import onb_model_specs
from utils.experiment.learning_policy import aligned_indices, checked_metadata
from utils.experiment.run_helpers import (
    makedirs, open_text, short_digest, windows_long_path, json_default, set_global_seed,
)
from utils.training.fitted_artifacts import _file_digest
from utils.training.model_training import PCA_FEATURE_DECIMALS, PCA_TRANSFORM_VERSION


def read_json(path):
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with open_text(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)
        stream.write("\n")


def _require(condition, message):
    if not condition:
        raise ValueError("Saved-fit evaluation: " + message)


def _rooted(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _metadata(path, experiment):
    rows = checked_metadata(load_sample_metadata_without_arrays(path), experiment)
    with open_text(path / "chunk_manifest.csv", "r", encoding="utf-8-sig", newline="") as stream:
        filenames = [r["sample_filename"] for r in csv.DictReader(stream)]
    _require(len(filenames) == len(set(filenames)) == len(rows)
             and set(filenames) == {r["sample_filename"] for r in rows},
             "input files and manifest differ or contain duplicate filenames")
    for row in rows:
        q = float(row["heat_flux"])
        _require(np.isfinite(q) and np.isclose(q, float(row["sample_filename"].split("_")[0]),
                                               rtol=0, atol=1e-6), "filename/manifest label mismatch")
        _require(float(row["chunk_duration_seconds"]) == 1., "expected one-second chunks")
    return rows


def validate_saved_weights(artifact, model_keys, strategy_names):
    """Use saved numbers verbatim; never renormalize or fit on evaluation labels."""
    selected = {}
    for name in strategy_names:
        key = "ensemble__" + name
        _require(key in artifact["ensemble_weights"], f"missing saved weights: {key}")
        weights = artifact["ensemble_weights"][key]["weights"]
        _require(set(weights) == set(model_keys), f"saved weight members differ: {key}")
        values = np.asarray([weights[k] for k in model_keys], float)
        _require(np.isfinite(values).all() and (values >= 0).all()
                 and np.isclose(values.sum(), 1, rtol=0, atol=1e-8), f"invalid saved weights: {key}")
        selected[key] = {k: float(weights[k]) for k in model_keys}
    return selected


def _preprocessing_records(jobs, config, manifest, train_days, hashes):
    """Record noise references separately from the shared STFT/filter contract."""
    paths = {str(j["data_path"].parent / "preprocess_manifest.json") for j in jobs}
    data = manifest["validation_config"]["data"]
    mapping = data.get("data_source_dir_by_experiment", {})
    experiment_root = Path(config["data"]["experiment_root"])
    for day in train_days:
        tag = str(mapping[day]).replace("{chunk_tag}", "1s")
        paths.add(str(experiment_root / day / "data/npy" / tag / "maxfreq=3kHz/preprocess_manifest.json"))
    records = []
    signature = None
    for path in sorted(paths):
        item = read_json(path)
        current = {k: item[k] for k in ["samplerate_hz", "chunk_seconds", "max_freq_hz", "stft", "highpass_filter"]}
        if signature is None:
            signature = current
        _require(current == signature, "STFT/highpass preprocessing differs between source and target")
        hashes[path] = _file_digest(path)
        records.append({"path": path, "sha256": hashes[path], "input_contract": current, "noise": item["noise"]})
    return records


def prepare_saved_fit(jobs, config, root):
    """Read-only preflight, including provenance, hashes and noise alignment."""
    root = Path(root)
    settings = config["execution"]
    _require(config["learning_policy"]["training_noise"] == "clean_only", "requires clean_only")
    _require(config["learning_policy"]["evaluation_mode"] == "cross_day", "requires cross_day")
    _require(config["data"]["max_freq_hz_list"] == ["maxfreq=3kHz"]
             and config["data"]["chunk_seconds"] == 1 and config["run"]["color_channel"] == 1,
             "saved frequency34 models require one-second, 3 kHz, one-channel power")
    _require(not config["acoustic_selection"].get("enabled"), "acoustic selection must stay disabled")
    _require(not config["models"].get("parameter_search_enabled"), "parameter search is incompatible with saved_fit")
    _require(not config["explainability"].get("enabled"), "use the existing training mode for additional XAI")
    _require(int(settings["prediction_batch_size"]) == 32 and int(settings["source_probe_samples"]) > 0,
             "use the source runner's inference batch size 32 and a positive probe count")
    _require(jobs and len({j["experiment_name"] for j in jobs}) == 1, "select one new evaluation experiment")
    noises = [j["noise_dir_name"] for j in jobs]
    _require(len(noises) == len(set(noises)) and "heatflux_no_noise" in noises,
             "unique noise conditions including clean are required")
    source = _rooted(root, settings["source_condition_dir"])
    fold = int(settings["source_fold"])
    directory = source / "fitted_state" / f"fold{fold}"
    artifact_path = directory / "artifact_manifest.json"
    artifact = read_json(artifact_path)
    manifest = read_json(source / "run_manifest.json")
    completed = read_json(source / "completed.json")
    _require(completed["run_hash"] == manifest["run_hash"] and artifact["fit_id"] in completed["fit_ids"],
             "source condition is incomplete or has another fit ID")
    _require(manifest["learning_context"]["training_noise"] == "clean_only"
             and manifest["dataset"]["noise_dir_name"] == "heatflux_no_noise", "source is not a clean-only fit")
    _require(manifest["dataset"]["max_freq_hz"] == "maxfreq=3kHz"
             and manifest["validation_config"]["data"]["chunk_seconds"] == 1
             and manifest["validation_config"]["run"]["color_channel"] == 1, "source input representation differs")
    _require(artifact["fold"] == fold and artifact["pca_transform_version"] == PCA_TRANSFORM_VERSION
             and artifact["pca_feature_decimals"] == PCA_FEATURE_DECIMALS, "saved PCA implementation differs")
    keys = config["models"]["fixed_model_keys"]
    _require(set(keys) == set(artifact["models"]), "active models differ from saved models")
    specs = onb_model_specs(keys)
    for spec in specs:
        record = artifact["models"][spec["key"]]
        _require(spec["kind"] == record["kind"], "saved model kind differs")
        if spec["kind"] == "sklearn_summary":
            _require(spec["feature_version"] == record["feature_version"], "summary feature version differs")
        spec["builder_params"] = record["builder_params"]
    strategies = [s["name"] for s in config["ensemble"]["resolved_strategy_plan"]]
    weights = validate_saved_weights(artifact, keys, strategies)
    required = {"ensemble__original3_mse", "ensemble__original3_hgb", "ensemble__original3_extra_trees",
                "ensemble__original3_hgb_extra_trees"}
    _require(required <= set(weights), "include the original3, HGB4, ET4 and five-model comparisons")
    records = [*artifact["preprocessors"].values(), artifact["ensemble_weights_file"],
               *[r["artifact"] for r in artifact["models"].values()]]
    hashes = {str(artifact_path): _file_digest(artifact_path)}
    for record in records:
        path = directory / record["path"]
        value = _file_digest(path)
        _require(value == record["sha256"], f"artifact SHA-256 mismatch: {record['path']}")
        hashes[str(path)] = value
    for name in ["run_manifest.json", "completed.json", "split_manifest.json", f"internal_validation_fold{fold}.json"]:
        hashes[str(source / name)] = _file_digest(source / name)

    # The current source fit used one combined recording collection. Select its
    # exact training chunks from the saved internal audit, not the whole folder.
    data_path = Path(manifest["dataset"]["data_path"])
    source_meta = _metadata(data_path, manifest["dataset"]["experiment_name"])
    hashes[str(data_path / "chunk_manifest.csv")] = _file_digest(data_path / "chunk_manifest.csv")
    audit = read_json(source / f"internal_validation_fold{fold}.json")
    lookup = {(r["source_wav_id"], int(r["chunk_index"])): r for r in source_meta}
    _require(len(lookup) == len(source_meta), "ambiguous source WAV/chunk identities")
    training_meta = [lookup[r["source_wav_id"], int(r["chunk_index"])] for r in audit["samples"]]
    _require(len({(r["source_wav_id"], int(r["chunk_index"])) for r in training_meta}) == len(training_meta),
             "duplicate source training identities")
    np.testing.assert_allclose([float(r["heat_flux"]) for r in training_meta],
                               [float(r["heat_flux"]) for r in audit["samples"]], rtol=0, atol=1e-6)
    splits = read_json(source / "split_manifest.json")["folds"]
    split = next(r for r in splits if r["fold"] == fold)
    _require(split["fit_id"] == artifact["fit_id"] and split["n_training_chunks"] == len(training_meta),
             "source split/audit/fit mismatch")
    train_days = {source_experiment(r) for r in training_meta}
    train_wavs = {(source_experiment(r), r["source_wav_id"]) for r in training_meta}
    clean_job = next(j for j in jobs if j["noise_dir_name"] == "heatflux_no_noise")
    reference = _metadata(clean_job["data_path"], clean_job["experiment_name"])
    evaluations = []
    for job in jobs:
        metadata = _metadata(job["data_path"], job["experiment_name"])
        order = aligned_indices(reference, metadata)
        metadata = [metadata[int(i)] for i in order]
        _require(not train_days & {source_experiment(r) for r in metadata}, "training and evaluation days overlap")
        _require(not train_wavs & {(source_experiment(r), r["source_wav_id"]) for r in metadata},
                 "training and evaluation WAVs overlap")
        for day in {source_experiment(r) for r in metadata}:
            threshold = float(config["thresholds"]["by_experiment"][day])
            _require(np.isfinite(threshold) and threshold > 0, "missing/invalid target ONB threshold")
        label_values = np.asarray([float(r["heat_flux"]) for r in metadata])
        csv_path = job["data_path"] / "chunk_manifest.csv"
        hashes[str(csv_path)] = _file_digest(csv_path)
        evaluations.append({"job": job, "metadata": metadata, "targets": label_values})
    preprocessing = _preprocessing_records(jobs, config, manifest, train_days, hashes)
    for day in {source_experiment(r) for r in reference}:
        provenance = config["thresholds"].get("provenance_by_experiment", {}).get(day, {})
        if provenance.get("heat_flux_csv_sha256"):
            label_path = _rooted(root, provenance["heat_flux_csv"])
            _require(_file_digest(label_path) == provenance["heat_flux_csv_sha256"], "ONB label CSV changed")
            hashes[str(label_path)] = provenance["heat_flux_csv_sha256"]
    probe_csv = source / "fold_pred" / f"pred_f{fold}_no_noise.csv"
    with open_text(probe_csv, "r", encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    probe_count = min(int(settings["source_probe_samples"]), len(saved))
    _require(probe_count > 0, "source predictions are empty")
    # The original runner called Keras predict with its default batch size 32.
    # GPU results can change with batch shape. Restore entire original batches
    # around the selected probes, including the original final partial batch.
    starts = sorted({int(i)//32*32 for i in np.linspace(0, len(saved)-1, probe_count, dtype=int)})
    probe = [row for start in starts for row in saved[start:start+32]]
    _require(all(int(r["fold"]) == fold for r in saved), "source probe fold differs")
    hashes[str(probe_csv)] = _file_digest(probe_csv)
    _require({int(r["sample_index"]) for r in saved} == set(split["evaluation_sample_indices"]),
             "saved evaluation predictions and split differ")
    for row in saved:
        original = source_meta[int(row["sample_index"])]
        _require(row["sample_filename"] == original["sample_filename"]
                 and row["source_wav_id"] == original["source_wav_id"]
                 and int(row["chunk_index"]) == int(original["chunk_index"])
                 and np.isclose(float(row["y_true"]), float(original["heat_flux"]), rtol=0, atol=1e-6),
                 "source saved prediction metadata differs from input manifest")
    _require(not {(r["source_wav_id"], int(r["chunk_index"])) for r in training_meta}
             & {(r["source_wav_id"], int(r["chunk_index"])) for r in saved}, "source training/evaluation chunks overlap")
    return {"source": source, "directory": directory, "artifact": artifact, "manifest": manifest,
            "specs": specs, "weights": weights, "training_metadata": training_meta,
            "source_data_path": data_path, "source_metadata": source_meta, "probe": probe,
            "evaluations": evaluations, "hashes": hashes, "preprocessing": preprocessing,
            "summary": {"source_run_hash": manifest["run_hash"], "fit_id": artifact["fit_id"],
                "source_training_chunks": len(training_meta), "source_days": sorted(train_days),
                "target_days": sorted({source_experiment(r) for r in reference}),
                "evaluation_chunks_per_condition": len(reference), "noise_conditions": noises,
                "models": keys, "ensembles": list(weights), "training_performed": False,
                "preprocessing_and_weights_frozen": True}}


def input_batches(data_path, metadata, batch_size):
    for start in range(0, len(metadata), batch_size):
        rows = metadata[start:start+batch_size]
        values = [np.load(windows_long_path(data_path / r["sample_filename"]), allow_pickle=False) for r in rows]
        x = np.asarray(values, np.float32)
        if x.ndim == 3:
            x = x[..., None]
        _require(x.shape == (len(rows), 224, 224, 1) and np.isfinite(x).all() and (x >= 0).all(),
                 "invalid power array (expected finite nonnegative 224x224)")
        yield rows, x


def _predict(model, spec, trainer, scaler, pca, data_path, metadata, batch_size):
    values = []
    for _, x in input_batches(data_path, metadata, batch_size):
        transformed = trainer.transform_pca(pca, x) if spec["kind"] == "sklearn" else None
        current = np.asarray(trainer.predict_one_model(spec, model, x, transformed, scaler), float).ravel()
        _require(len(current) == len(x) and np.isfinite(current).all(), f"invalid predictions: {spec['key']}")
        values.extend(current)
    return np.asarray(values)


def run_saved_fit_evaluation(jobs, config, root, preflight=False):
    prepared = prepare_saved_fit(jobs, config, root)
    print(json.dumps(prepared["summary"], ensure_ascii=False, indent=2), flush=True)
    if preflight:
        print("Preflight passed: no training, prediction or output performed.", flush=True)
        return prepared["summary"]
    from tensorflow.keras import backend as keras_backend
    from utils.models.regression.base_regression import RegressionModelMaker
    from utils.training.model_training import ModelTrainer
    from utils.calculation.saved_fit_report import write_transfer_diagnostics, write_feature_comparison

    artifact, directory = prepared["artifact"], prepared["directory"]
    set_global_seed(artifact["random_seed"], deterministic_ops=True)
    batch_size = int(config["execution"]["prediction_batch_size"])
    job0 = jobs[0]
    tag = short_digest({"fit": artifact["fit_id"], "data": config["data"], "weights": prepared["weights"]})
    output = job0["save_base_path"] / f"onb_saved_{tag}_{config['output']['run_instance_id']}" / job0["max_freq_hz"]
    # Each application is a new record. Existing predictions and source fits
    # are never overwritten, including a manually repeated RUN_ID.
    Path(windows_long_path(output)).mkdir(parents=True, exist_ok=False)
    write_json(output / "run_conditions.json", {"evaluation": prepared["summary"], "requested_config": config,
        "source_condition": str(prepared["source"]), "source_validation_config": prepared["manifest"]["validation_config"],
        "source_artifact_manifest": artifact, "source_sha256": prepared["hashes"],
        "preprocessing_manifests": prepared["preprocessing"],
        "reference_snr_scope": "Each generation's reference RMS; not equal absolute added-noise amplitude across days."})
    scaler = joblib.load(windows_long_path(directory / artifact["preprocessors"]["target_scaler"]["path"]))
    pca = joblib.load(windows_long_path(directory / artifact["preprocessors"]["pca"]["path"]))
    trainer = ModelTrainer(random_seed=artifact["random_seed"])
    results = {j["noise_dir_name"]: {} for j in jobs}
    checks = []
    for spec in prepared["specs"]:
        key = spec["key"]
        keras_backend.clear_session()
        path = directory / artifact["models"][key]["artifact"]["path"]
        if spec["kind"] == "keras":
            maker = RegressionModelMaker((224, 224, 1))
            model = spec["builder"](maker, **spec.get("builder_params", {}))
            model.load_weights(windows_long_path(path))
        else:
            model = joblib.load(windows_long_path(path))
        probe = prepared["probe"]
        actual = _predict(model, spec, trainer, scaler, pca, prepared["source_data_path"], probe, batch_size)
        expected = np.asarray([float(r[key]) for r in probe])
        _require(np.allclose(actual, expected, rtol=1e-6, atol=1e-3), f"source prediction reproduction failed: {key}")
        checks.append({"model_key": key, "probe_samples": len(probe), "inference_batch_size": batch_size, "passed": True,
                       "max_abs_difference_W_m2": float(abs(actual-expected).max())})
        for item in prepared["evaluations"]:
            job = item["job"]
            noise = job["noise_dir_name"]
            results[noise][key] = _predict(model, spec, trainer, scaler, pca, job["data_path"], item["metadata"], batch_size)
            print(f"[saved-fit] {key} / {noise}: {len(item['metadata'])} predictions", flush=True)
        del model
        keras_backend.clear_session()
        gc.collect()

    keys = [s["key"] for s in prepared["specs"]]
    metric_rows = []
    for item in prepared["evaluations"]:
        job, metadata, y = item["job"], item["metadata"], item["targets"]
        noise = job["noise_dir_name"]
        predictions = results[noise]
        for key, weights in prepared["weights"].items():
            predictions[key] = sum(predictions[k]*weights[k] for k in keys)
        condition_dir = output / noise
        makedirs(condition_dir)
        records = build_fold_prediction_rows(np.arange(len(y)), y, predictions, metadata, artifact["fold"])
        write_fold_prediction_csv(condition_dir / "predictions.csv", records, predictions)
        rows = source_day_metric_rows(y, predictions, metadata, config["thresholds"]["by_experiment"], artifact["fold"],
                                     band_frac=config["thresholds"]["onb_band_frac"])
        metric_rows.extend({**r, "seed": prepared["manifest"]["validation_config"]["run"]["random_seed"],
                            "noise": noise_key(noise)} for r in rows)
        write_transfer_diagnostics(condition_dir, y, predictions, metadata, config["thresholds"]["by_experiment"],
            prepared["training_metadata"], config["thresholds"]["onb_band_frac"])
        # Keep the actual per-condition noise and label provenance in the output.
        fields = ["sample_filename", "source_wav_id", "chunk_index", "heat_flux", "signal_chunk_power",
                  "scaled_noise_chunk_power", "realized_snr_db", "noise_scaling_mode", "fixed_reference_signal_rms"]
        _csv(condition_dir / "input_provenance.csv", [{k: r.get(k, "") for k in fields} for r in metadata])
    protocol = {"model_keys": keys, "ensemble_names": [k.removeprefix("ensemble__") for k in prepared["weights"]],
                "comparison_baselines": ["ensemble__original3_mse", "ensemble__original3_extra_trees", "extra_trees"],
                # Keep the full planned set so a clean-only run cannot be
                # reported as a completed six-noise average.
                "noise_conditions": ["clean", "0", "-4", "-8", "-12", "-16", "-20"]}
    for name, rows in comparison_tables(metric_rows, protocol).items():
        if rows:
            _csv(output / name, rows)
    write_json(output / "frozen_weights.json", prepared["weights"])
    _csv(output / "source_reload_checks.csv", checks)
    if config["execution"].get("export_feature_comparison", True):
        clean = next(v for v in prepared["evaluations"] if v["job"]["noise_dir_name"] == "heatflux_no_noise")
        write_feature_comparison(output, prepared, clean, config["thresholds"]["by_experiment"], batch_size)
    _require(all(_file_digest(p) == sha for p, sha in prepared["hashes"].items()), "source files changed during evaluation")
    write_json(output / "completed.json", {**prepared["summary"], "status": "passed", "source_files_unchanged": True,
        "source_reload_checks": checks, "base_predictions": len(keys)*sum(len(r["metadata"]) for r in prepared["evaluations"])})
    print(f"Saved-fit evaluation completed: {output}", flush=True)
    return output
