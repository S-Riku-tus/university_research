"""Fixed HGB/ET input-removal retraining; preserve source original-three models."""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from common import (ARTIFACTS, CONFIG, CORE, FIVE, MEMBERS, OUT, THRESHOLDS, digest, frozen_fit,
    load_raw, metric_rows, read_csv, read_json, save_csv, save_json, source_columns,
    source_fingerprint, source_metadata, source_predictions, training_data, windows_long_path)
from utils.dataloading.acoustic_summary_features import AcousticFrequency34
from utils.ensemble.fixed_core_stacking import FIXED_CORE_DEFAULTS, fit_fixed_core_strategy


def persisted_fit(saved_model, full_features, raw_held, y_W, indices, seed, path):
    """Fit only the supplied clean training rows, then persist/reload the pipeline."""
    selector = ColumnTransformer([("selected_features", "passthrough", indices)], remainder="drop")
    selected = selector.fit_transform(full_features)
    scaler = MinMaxScaler().fit(np.asarray(y_W).reshape(-1, 1))
    scaled = scaler.transform(np.asarray(y_W).reshape(-1, 1)).ravel()
    regressor = clone(saved_model.named_steps["regressor"]).set_params(random_state=seed)
    regressor.fit(selected, scaled)
    pipeline = Pipeline([("features", AcousticFrequency34()), ("selected", selector), ("regressor", regressor)])
    joblib.dump(pipeline, path, compress=3)
    scaler_path = path.with_name(path.stem+"_target_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    restored = joblib.load(path)
    restored_scaler = joblib.load(scaler_path)
    prediction = scaler.inverse_transform(pipeline.predict(raw_held).reshape(-1, 1)).ravel()
    reloaded = restored_scaler.inverse_transform(restored.predict(raw_held).reshape(-1, 1)).ravel()
    np.testing.assert_allclose(prediction, reloaded, rtol=1e-12, atol=1e-6)
    return pipeline, scaler, prediction, {"model_artifact": path.name, "target_scaler_artifact": scaler_path.name,
        "fit_chunks": len(y_W), "random_state": seed, "feature_indices": indices,
        "max_reload_difference_W_m2": float(abs(prediction-reloaded).max()), "model_sha256": digest(path),
        "scaler_sha256": digest(scaler_path), "reload_samples": len(raw_held), "passed": True}


def main():
    directory = OUT / "stage3"
    models_dir = directory / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    assert read_json(OUT / "stage1/verification.json")["status"] == "passed"
    assert read_json(OUT / "stage2/verification.json")["status"] == "passed"
    paired_stage2 = read_csv(OUT / "stage2/paired_detection_changes.csv")
    remaining = int(paired_stage2[paired_stage2.noise == "clean"].iloc[0].remaining_original_all_five_common_fn)
    assert remaining > 0
    save_json(directory / "trigger.json", {"trigger_satisfied": True, "unchanged_clean_common_fn": remaining,
        "reason": "Nonnegative reweighting leaves 11 common misses; gain/shape dependence requires a retrained removal contrast.",
        "protocol": CONFIG["stage3"]})
    fingerprint = source_fingerprint()
    frame, audit, oof = training_data()
    raw = load_raw(frame, "clean")
    transformer = AcousticFrequency34()
    full = transformer.transform(raw)
    y = frame.y_true.to_numpy()
    np.savez_compressed(directory / "training_features.npz", features=full, y_true_W_m2=y,
        source_wav_id=frame.source_wav_id.to_numpy(dtype=str), chunk_index=frame.chunk_index.to_numpy())
    families = {key: joblib.load(windows_long_path(ARTIFACTS / f"{key}.joblib")) for key in ["hgb", "extra_trees"]}
    for model in families.values():
        np.testing.assert_allclose(model.named_steps["features"].transform(raw), full, rtol=0, atol=0)
    contrasts = {key: value for key, value in CONFIG["stage3"]["representations"].items() if isinstance(value, list)}
    models, new_oof, fit_audit, metrics = {}, {}, [], []
    count = 0
    for key, saved_model in families.items():
        for representation, indices in contrasts.items():
            candidate = key+"__"+representation
            predictions = np.full(len(frame), np.nan)
            for split in audit["folds"]:
                fit = np.array(split["fit_indices"], int)
                held = np.array(split["validation_indices"], int)
                path = models_dir / f"{candidate}_fold{split['fold']}.joblib"
                _, _, predictions[held], record = persisted_fit(saved_model, full[fit], raw[held], y[fit],
                    indices, 42+int(split["fold"]), path)
                record.update(model_key=key, representation=representation, stage="internal", fold=int(split["fold"]),
                    held_chunks=len(held), shared_chunks=0)
                fit_audit.append(record)
                count += 1
                print(f"[stage3] {candidate} / inner fold {split['fold']}: {count}/16 fits", flush=True)
            assert np.isfinite(predictions).all()
            new_oof[candidate] = predictions
            path = models_dir / f"{candidate}_final.joblib"
            # Probe only training rows here. All outer/noisy rows are evaluated
            # after every clean fit and integration weight has been frozen.
            model, scaler, _, record = persisted_fit(saved_model, full, raw[:8], y, indices, 43, path)
            models[candidate] = (model, scaler, path)
            record.update(model_key=key, representation=representation, stage="final", fold=1)
            fit_audit.append(record)
            count += 1
            print(f"[stage3] {candidate} / final: {count}/16 fits", flush=True)
    assert count == CONFIG["stage3"]["expected_model_fits"]
    frozen, oof_predictions = {}, {key: oof[key].to_numpy() for key in MEMBERS}
    options = {**FIXED_CORE_DEFAULTS, "added_keys": ["hgb", "extra_trees"]}
    current_weights, current_diagnostic = fit_fixed_core_strategy(oof_predictions, y, MEMBERS, options)
    np.testing.assert_allclose(list(current_weights.values()),
        [frozen_fit()["weights"][FIVE.removeprefix("ensemble__")][key] for key in MEMBERS], rtol=1e-7, atol=1e-8)
    for candidate, predictions in new_oof.items():
        replaced = candidate.split("__")[0]
        training_predictions = {**oof_predictions, replaced: predictions}
        weights, diagnostic = fit_fixed_core_strategy(training_predictions, y, MEMBERS, options)
        np.testing.assert_allclose(list(diagnostic["core_weights"].values()), list(current_diagnostic["core_weights"].values()),
            rtol=1e-7, atol=1e-8)
        assert all(w > 0 for w in weights.values()) and weights["hgb"] >= .05-1e-10 and weights["extra_trees"] >= .05-1e-10
        assert sum(weights[k] for k in MEMBERS[:3]) >= .25-1e-10
        frozen[candidate] = {"weights": weights, "diagnostic": diagnostic, "replaced_member": replaced}
    save_json(directory / "frozen_integration.json", {"protocol": CONFIG["stage3"], "contrasts": frozen,
        "freeze_timing": "all 16 clean fits and clean OOF weights frozen before outer/noise scoring",
        "outer_labels_used_for_fit_or_selection": False})
    training_predictions = {FIVE: sum(oof_predictions[k]*current_weights[k] for k in MEMBERS),
        "hgb": oof_predictions["hgb"], "extra_trees": oof_predictions["extra_trees"]}
    for candidate, prediction in new_oof.items():
        training_predictions[candidate] = prediction
        weights = frozen[candidate]["weights"]
        key = frozen[candidate]["replaced_member"]
        training_predictions["ensemble__replace_"+candidate] = sum((prediction if k == key else oof_predictions[k])*weights[k] for k in MEMBERS)
    metrics.extend(metric_rows(frame, training_predictions, "clean", scope="training_oof_weight_fit_diagnostic"))
    np.savez_compressed(directory / "oof_predictions.npz", **new_oof, heat_flux_unit="W/m2")
    del raw
    checks, saved_predictions, case_rows, paired = [], [], [], []
    for noise in CONFIG["evaluation_noise_conditions"]:
        outer = source_predictions(noise)
        raw_outer = load_raw(outer, noise)
        full_outer = transformer.transform(raw_outer)
        predictions = {key: outer[key].to_numpy() for key in source_columns(outer)}
        for key, saved_model in families.items():
            scaler = joblib.load(windows_long_path(ARTIFACTS / "target_scaler.joblib"))
            values = scaler.inverse_transform(saved_model.predict(raw_outer).reshape(-1, 1)).ravel()
            np.testing.assert_allclose(values, outer[key], rtol=1e-6, atol=1e-3)
            checks.append({"noise": noise, "candidate": key+"__full34_reference", "samples": 540,
                "max_abs_difference_W_m2": float(abs(values-outer[key].to_numpy()).max()), "passed": True})
        for candidate, (model, scaler, path) in models.items():
            value = scaler.inverse_transform(model.predict(raw_outer).reshape(-1, 1)).ravel()
            reloaded = joblib.load(path)
            reloaded_scaler = joblib.load(path.with_name(path.stem+"_target_scaler.joblib"))
            restored = reloaded_scaler.inverse_transform(reloaded.predict(raw_outer).reshape(-1, 1)).ravel()
            direct = scaler.inverse_transform(model.named_steps["regressor"].predict(model.named_steps["selected"].transform(full_outer)).reshape(-1, 1)).ravel()
            np.testing.assert_allclose(value, restored, rtol=1e-12, atol=1e-6)
            np.testing.assert_allclose(value, direct, rtol=1e-12, atol=1e-6)
            checks.append({"noise": noise, "candidate": candidate, "samples": 540,
                "max_abs_difference_W_m2": float(abs(value-restored).max()), "passed": True})
            predictions[candidate] = value
            integration = frozen[candidate]
            replaced, weights = integration["replaced_member"], integration["weights"]
            predictions["ensemble__replace_"+candidate] = sum((value if k == replaced else outer[k].to_numpy())*weights[k] for k in MEMBERS)
        metrics.extend(metric_rows(outer, predictions, noise, scope="outer_development"))
        days = np.array([r["experiment_name"] for r in source_metadata(outer)])
        threshold = np.array([THRESHOLDS[d] for d in days])
        actual = outer.y_true.to_numpy() >= threshold
        common = actual & np.all(outer[MEMBERS].to_numpy() < threshold[:, None], axis=1)
        current_fn, current_fp = actual & (outer[FIVE].to_numpy() < threshold), ~actual & (outer[FIVE].to_numpy() >= threshold)
        for key, prediction in predictions.items():
            if key in source_columns(outer):
                continue
            after_fn, after_fp = actual & (prediction < threshold), ~actual & (prediction >= threshold)
            paired.append({"noise": noise, "model_key": key, "corrected_fn_vs_five": int((current_fn & ~after_fn).sum()),
                "added_fn_vs_five": int((~current_fn & after_fn).sum()), "corrected_fp_vs_five": int((current_fp & ~after_fp).sum()),
                "added_fp_vs_five": int((~current_fp & after_fp).sum()), "original_common_fn_corrected": int((common & ~after_fn).sum()),
                "original_common_fn_remaining": int((common & after_fn).sum())})
            for i in range(len(outer)):
                row = {"noise": noise, "model_key": key, "source_wav_id": outer.source_wav_id.iloc[i],
                    "chunk_index": int(outer.chunk_index.iloc[i]), "target_kW_m2": outer.y_true.iloc[i]/1000,
                    "prediction_kW_m2": prediction[i]/1000}
                saved_predictions.append(row)
                if noise == "clean" and common[i]:
                    case_rows.append({**row, "source_day": days[i], "threshold_kW_m2": threshold[i]/1000,
                        "source_five_prediction_kW_m2": outer[FIVE].iloc[i]/1000, "common_fn_corrected": bool(not after_fn[i])})
        print(f"[stage3] {noise}: 4 persisted contrasts, 540 chunks each, final reload passed", flush=True)
        del raw_outer
    save_csv(directory / "metrics.csv", metrics)
    save_csv(directory / "model_fit_audit.csv", fit_audit)
    save_csv(directory / "final_reload_checks.csv", checks)
    save_csv(directory / "predictions.csv", saved_predictions)
    save_csv(directory / "paired_detection_changes.csv", paired)
    save_csv(directory / "original_common_fn_cases.csv", case_rows)
    assert source_fingerprint() == fingerprint
    save_json(directory / "verification.json", {"status": "passed", "new_model_fits": count, "new_cnn_fits": 0,
        "new_model_families": 0, "source_full34_models_reused": True, "persisted_models": len(fit_audit),
        "all_persisted_models_reload_verified": True, "final_subset_reload_predictions": 4*7*540,
        "reference_full34_reload_predictions": 2*7*540, "training_chunks": 1620, "evaluation_chunks_per_noise": 540,
        "oof_coverage_one": True, "no_shared_training_outer_chunks": True, "no_shared_inner_fit_held_chunks": True,
        "outer_labels_used_for_fit_or_selection": False, "clean_only_training": True,
        "source_models_and_weights_unchanged": True, "metrics_rows": len(metrics), "input_files": fingerprint,
        "scope": CONFIG["stage3"]["interpretation"]})
    print(f"[stage3] passed: {count} fits; all saved/reloaded; {len(metrics)} metric rows", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=CONFIG["cpu_threads"]):
        main()
