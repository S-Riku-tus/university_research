"""Full-population common interventions of the saved five models, no fitting."""
import gc
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from common import (ARTIFACTS, CONFIG, FIVE, ET4, MEMBERS, OUT, POOLED, THRESHOLDS,
    deltas, frozen_fit, load_raw, metric_rows, read_json, save_csv, save_json,
    source_columns, source_fingerprint, source_metadata, source_predictions, windows_long_path)
from utils.dataloading.acoustic_summary_features import AcousticFrequency34


def apply_intervention(raw, intervention):
    kind = intervention["type"]
    if kind == "identity":
        return raw
    if kind == "time_permutation":
        order = np.random.default_rng(intervention["seed"]).permutation(224)
        return np.ascontiguousarray(raw[:, order, :, :])
    if kind == "global_gain":
        return raw*np.float32(intervention["power_factor"])
    changed = raw.copy()
    start, stop = [round(224*intervention[k]/3000) for k in ["low_hz", "high_hz"]]
    changed[:, :, start:stop, :] *= np.float32(intervention["power_factor"])
    return changed


def main():
    directory = OUT / "stage1"
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = source_fingerprint()
    save_json(directory / "started.json", {"protocol": CONFIG, "input_files": fingerprint})
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("Saved GPU endpoint comparisons require the recorded GPU path")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    from utils.experiment.run_helpers import set_global_seed
    set_global_seed(43, deterministic_ops=True)
    from utils.models.regression.base_regression import RegressionModelMaker
    from utils.training.model_training import ModelTrainer
    trainer = ModelTrainer(42)
    scaler = joblib.load(windows_long_path(ARTIFACTS / "target_scaler.joblib"))
    pca = joblib.load(windows_long_path(ARTIFACTS / "pca.joblib"))
    artifact = read_json(ARTIFACTS / "artifact_manifest.json")
    interventions = CONFIG["stage1"]["interventions"]
    noises = CONFIG["stage1"]["noise_conditions"]
    frames = {noise: source_predictions(noise) for noise in noises}
    raw_inputs = {noise: load_raw(frames[noise], noise) for noise in noises}
    identity_columns = ["sample_index", "source_wav_id", "chunk_index", "y_true"]
    assert frames[noises[0]][identity_columns].equals(frames[noises[1]][identity_columns])
    features = AcousticFrequency34()
    feature_names = features.get_feature_names_out()
    feature_audit, checks = [], []
    for noise in noises:
        baseline = features.transform(raw_inputs[noise])
        for intervention in interventions:
            changed = features.transform(apply_intervention(raw_inputs[noise], intervention))
            if intervention["type"] == "time_permutation":
                np.testing.assert_allclose(changed, baseline, rtol=1e-12, atol=1e-9)
                checks.append({"noise": noise, "check": "feature34_time_invariance", "max_abs_difference": float(abs(changed-baseline).max()), "passed": True})
            if intervention["type"] == "global_gain":
                assert np.all(baseline[:, :5] > -20) and np.all(changed[:, :5] > -20)
                expected = baseline.copy()
                expected[:, :5] += np.log10(intervention["power_factor"])
                np.testing.assert_allclose(changed, expected, rtol=1e-12, atol=1e-9)
                checks.append({"noise": noise, "check": intervention["name"]+"_absolute_power_only", "max_abs_difference": float(abs(changed-expected).max()), "passed": True})
            for j, name in enumerate(feature_names):
                feature_audit.append({"noise": noise, "intervention": intervention["name"], "feature": name,
                    "mean_change": float(np.mean(changed[:, j]-baseline[:, j])),
                    "mean_abs_change": float(np.mean(abs(changed[:, j]-baseline[:, j]))),
                    "max_abs_change": float(np.max(abs(changed[:, j]-baseline[:, j])))})
    save_csv(directory / "feature_intervention_checks.csv", feature_audit)
    predictions = np.empty((len(noises), len(interventions), 540, len(MEMBERS)), dtype=float)
    for member_index, key in enumerate(MEMBERS):
        tf.keras.backend.clear_session()
        if artifact["models"][key]["kind"] == "keras":
            maker = RegressionModelMaker((224, 224, 1))
            model = maker.cnn_transformer_v2() if key == "conformer" else maker.alexnet()
            model.load_weights(windows_long_path(ARTIFACTS / f"{key}.weights.h5"))
        else:
            model = joblib.load(windows_long_path(ARTIFACTS / f"{key}.joblib"))
        spec = {"key": key, "kind": artifact["models"][key]["kind"]}
        for noise_index, noise in enumerate(noises):
            for intervention_index, intervention in enumerate(interventions):
                raw = apply_intervention(raw_inputs[noise], intervention)
                projection = trainer.transform_pca(pca, raw) if key == "randomforest" else None
                predicted = trainer.predict_one_model(spec, model, raw, projection, scaler)
                predictions[noise_index, intervention_index, :, member_index] = predicted
                if intervention["type"] == "identity":
                    expected = frames[noise][key].to_numpy()
                    np.testing.assert_allclose(predicted, expected, rtol=1e-6, atol=1e-3)
                    checks.append({"noise": noise, "check": key+"_unchanged_matches_saved", "max_abs_difference_W_m2": float(abs(predicted-expected).max()), "passed": True})
                if key in ["hgb", "extra_trees"] and intervention["type"] == "time_permutation":
                    baseline = predictions[noise_index, 0, :, member_index]
                    np.testing.assert_allclose(predicted, baseline, rtol=1e-12, atol=1e-6)
                    checks.append({"noise": noise, "check": key+"_time_invariance", "max_abs_difference_W_m2": float(abs(predicted-baseline).max()), "passed": True})
                print(f"[stage1] {key} / {noise} / {intervention['name']}: 540 predictions", flush=True)
                del projection, raw
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        # Persist progress after each model to keep completed inference reviewable.
        np.savez_compressed(directory / f"{key}_predictions.npz", predictions=predictions[:, :, :, member_index], noises=noises,
            interventions=[r["name"] for r in interventions], heat_flux_unit="W/m2")
    weights = frozen_fit()["weights"]
    weights["simple_equal"] = {key: .2 for key in MEMBERS}
    metrics, paired, representatives, population = [], [], [], []
    for noise_index, noise in enumerate(noises):
        frame = frames[noise]
        days = np.array([r["experiment_name"] for r in source_metadata(frame)])
        threshold = np.array([THRESHOLDS[d] for d in days])
        y = frame.y_true.to_numpy()
        actual = y >= threshold
        baseline_five = frame[FIVE].to_numpy()
        baseline_matrix = frame[MEMBERS].to_numpy()
        all_common = actual & np.all(baseline_matrix < threshold[:, None], axis=1)
        et4_changed = (frame[ET4].to_numpy() >= threshold) != (baseline_five >= threshold)
        last = np.zeros(len(y), bool)
        for day in THRESHOLDS:
            fn = (days == day) & actual & (baseline_five < threshold)
            if fn.any():
                last |= fn & (y == y[fn].max())
        selected = (all_common | et4_changed | last) if noise == "clean" else last
        for intervention_index, intervention in enumerate(interventions):
            matrix = predictions[noise_index, intervention_index]
            current = {key: matrix[:, j] for j, key in enumerate(MEMBERS)}
            for profile, w in weights.items():
                current["ensemble__"+profile] = matrix@np.array([w[key] for key in MEMBERS])
            if intervention_index == 0:
                for key in source_columns(frame):
                    np.testing.assert_allclose(current[key], frame[key], rtol=1e-6, atol=1e-3)
            metrics.extend(metric_rows(frame, current, noise, intervention=intervention["name"]))
            for key, prediction in current.items():
                before = frame[key].to_numpy()
                before_fn, after_fn = actual & (before < threshold), actual & (prediction < threshold)
                before_fp, after_fp = ~actual & (before >= threshold), ~actual & (prediction >= threshold)
                population.append({"noise": noise, "intervention": intervention["name"], "model_key": key,
                    "mean_prediction_change_kW_m2": float(np.mean(prediction-before)/1000),
                    "mean_abs_prediction_change_kW_m2": float(np.mean(abs(prediction-before))/1000)})
                paired.append({"noise": noise, "intervention": intervention["name"], "model_key": key,
                    "corrected_fn": int((before_fn & ~after_fn).sum()), "added_fn": int((~before_fn & after_fn).sum()),
                    "corrected_fp": int((before_fp & ~after_fp).sum()), "added_fp": int((~before_fp & after_fp).sum()),
                    "remaining_original_common_fn": int((all_common & after_fn).sum())})
                for i in np.flatnonzero(selected):
                    representatives.append({"noise": noise, "intervention": intervention["name"], "model_key": key,
                        "source_wav_id": frame.source_wav_id.iloc[i], "chunk_index": int(frame.chunk_index.iloc[i]),
                        "source_day": days[i], "target_kW_m2": y[i]/1000, "threshold_kW_m2": threshold[i]/1000,
                        "original_common_fn": bool(all_common[i]), "et4_five_detection_difference": bool(et4_changed[i]),
                        "last_five_fn": bool(last[i]), "before_prediction_kW_m2": before[i]/1000,
                        "after_prediction_kW_m2": prediction[i]/1000, "positive_after": bool(prediction[i] >= threshold[i])})
    save_csv(directory / "metrics.csv", metrics)
    save_csv(directory / "metric_deltas.csv", deltas(metrics))
    save_csv(directory / "paired_detection_changes.csv", paired)
    save_csv(directory / "prediction_change_summary.csv", population)
    save_csv(directory / "representative_case_predictions.csv", representatives)
    np.savez_compressed(directory / "base_predictions.npz", predictions=predictions, members=MEMBERS, noises=noises,
        interventions=[r["name"] for r in interventions], heat_flux_unit="W/m2")
    assert source_fingerprint() == fingerprint
    save_json(directory / "verification.json", {"status": "passed", "model_fits": 0,
        "model_predictions": int(predictions.size), "saved_models_and_weights_unchanged": True,
        "metrics_rows": len(metrics), "checks": checks, "input_files": fingerprint,
        "frequency_coordinates": CONFIG["stage1"]["frequency_coordinates"],
        "scope": CONFIG["stage1"]["interpretation"]})
    print(f"[stage1] passed: {predictions.size} model predictions, {len(metrics)} metric rows", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=CONFIG["cpu_threads"]):
        main()
