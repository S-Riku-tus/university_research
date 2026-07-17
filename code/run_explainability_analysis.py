"""
Run explainability checks for heat-flux regression models.

This script trains the three current comparison models on one CV fold and saves
input-resolution explanation maps for representative validation samples:

- AlexNet: Integrated Gradients and Grad-CAM
- CNN+Transformer v2 GAP: Integrated Gradients and Grad-CAM
- RF/XGBRF: PCA feature importance plus grouped spectrogram occlusion
- All models: grouped frequency/time occlusion maps

Set EXPLAIN_SMOKE=1 for a short end-to-end check.
"""

import csv
import gc
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend as K

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_ensemble_regression_onb as regression_run
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.explainability.spectrogram_explainers import (
    deletion_curve,
    ensure_dir,
    grad_cam_regression,
    integrated_gradients,
    make_axis_groups,
    normalize_map,
    normalize_magnitude,
    occlusion_importance,
    save_array_and_png,
    save_input_spectrogram_png,
    save_signed_array_and_png,
    summarize_map_by_axis,
    windows_long_path,
    write_csv,
)
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.model_training import ModelTrainer


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


EXPLAIN_CONFIG = {
    "run": {
        "fold": env_int("EXPLAIN_FOLD", 1),
        "epochs": env_int("EXPLAIN_EPOCHS", 300),
        "smoke_test": env_bool("EXPLAIN_SMOKE", False),
        "smoke_epochs": env_int("EXPLAIN_SMOKE_EPOCHS", 2),
        "random_seed": 42,
        "max_samples_per_group": env_int("EXPLAIN_SAMPLES_PER_GROUP", 1),
        "ig_steps": env_int("EXPLAIN_IG_STEPS", 32),
        "pca_components": 100,
    },
    "data": {
        "max_freq_hz": 22000,
        "result_date_dir": datetime.now().strftime("%Y%m%d") + "_xai_fold1",
    },
    "models": {
        "rf": {
            "n_estimators": 300,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bynode": 0.6,
        },
        "alexnet": {
            "lr": 0.005,
            "batch_size": 32,
            "fit_verbose": 1,
        },
        "cnntf_v2_gap": {
            "lr": 0.0005,
            "batch_size": 32,
            "fit_verbose": 1,
            "builder_params": {
                "num_transformer_blocks": 4,
                "head_size": 256,
                "num_heads": 4,
                "ff_dim": 2048,
                "model_dim": 32,
                "dropout": 0.2,
            },
        },
    },
}


def safe_name(text):
    chars = []
    for ch in str(text):
        chars.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(chars).strip("_")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def model_spec(model_key):
    spec_by_key = {spec["key"]: spec for spec in regression_run.MODEL_SPECS}
    spec = dict(spec_by_key[model_key])
    cfg = EXPLAIN_CONFIG["models"].get(model_key, {})
    builder_params = dict(spec.get("builder_params", {}))
    builder_params.update(cfg.get("builder_params", {}))
    spec["builder_params"] = builder_params
    for key in ("lr", "batch_size", "fit_verbose"):
        if key in cfg:
            spec[key] = cfg[key]
    if model_key == "rf":
        spec["builder_params"].update(cfg)
    return spec


def selected_sample_indices(y_val, pred, threshold, max_per_group):
    y_val = np.asarray(y_val)
    pred = np.asarray(pred)
    candidates = [
        ("low_heatflux", int(np.argmin(y_val))),
        ("near_onb", int(np.argmin(np.abs(y_val - threshold)))),
        ("high_heatflux", int(np.argmax(y_val))),
        ("best_prediction", int(np.argmin(np.abs(y_val - pred)))),
        ("worst_prediction", int(np.argmax(np.abs(y_val - pred)))),
    ]
    selected = []
    used = set()
    for label, idx in candidates:
        if idx in used:
            continue
        selected.append((label, idx))
        used.add(idx)
        if len(selected) >= len(candidates) * max(1, int(max_per_group)):
            break
    return selected


def keras_predict_fn(model, scaler):
    def predict(batch):
        pred_scaled = model.predict(batch, verbose=0)
        return scaler.inverse_transform(pred_scaled).ravel()
    return predict


def rf_predict_fn(model, pca, scaler):
    def predict(batch):
        flat = batch.reshape(batch.shape[0], -1)
        pred_scaled = model.predict(pca.transform(flat)).reshape(-1, 1)
        return scaler.inverse_transform(pred_scaled).ravel()
    return predict


def write_model_metrics(path, y_true, y_pred):
    rows = [[
        len(y_true),
        r2_score(y_true, y_pred),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mean_absolute_error(y_true, y_pred),
    ]]
    write_csv(path, ["n", "r2", "rmse", "mae"], rows)


def write_rf_feature_importance(model, out_dir):
    if not hasattr(model, "feature_importances_"):
        return
    rows = [
        [i, float(v)]
        for i, v in enumerate(np.asarray(model.feature_importances_).ravel())
    ]
    rows.sort(key=lambda row: row[1], reverse=True)
    write_csv(
        os.path.join(out_dir, "rf_pca_feature_importance.csv"),
        ["pca_component", "importance"],
        rows,
    )

    try:
        import shap
    except Exception:
        write_csv(
            os.path.join(out_dir, "rf_treeshap_status.csv"),
            ["status", "message"],
            [["skipped", "shap is not installed or could not be imported"]],
        )
        return

    write_csv(
        os.path.join(out_dir, "rf_treeshap_status.csv"),
        ["status", "message"],
        [["available", f"shap {getattr(shap, '__version__', 'unknown')} can be used for PCA-space TreeSHAP"]],
    )


def explain_keras_model(model_key, model, scaler, x_val, y_val, pred, threshold,
                        out_dir, groups, max_freq_hz, ig_steps):
    predict_fn = keras_predict_fn(model, scaler)
    sample_rows = []
    for label, local_idx in selected_sample_indices(y_val, pred, threshold, 1):
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = os.path.join(out_dir, sample_id)
        ensure_dir(sample_dir)

        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])

        save_input_spectrogram_png(
            sample,
            os.path.join(sample_dir, "input_spectrogram.png"),
            f"{model_key} input {sample_id}",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )

        ig_raw = integrated_gradients(model, sample, steps=ig_steps)
        ig = normalize_magnitude(ig_raw)
        save_signed_array_and_png(
            ig_raw,
            os.path.join(sample_dir, "integrated_gradients_signed"),
            f"{model_key} IG signed {sample_id}",
            unit="scaled model output",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )
        save_array_and_png(
            ig,
            os.path.join(sample_dir, "integrated_gradients_magnitude"),
            f"{model_key} IG magnitude {sample_id}",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )
        deletion_rows = deletion_curve(predict_fn, sample, ig)
        write_csv(
            os.path.join(sample_dir, "integrated_gradients_deletion_curve.csv"),
            ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
            deletion_rows,
        )
        freq_rows, time_rows = summarize_map_by_axis(ig, max_freq_hz)
        write_csv(os.path.join(sample_dir, "integrated_gradients_frequency_profile.csv"),
                  ["frequency_bin", "frequency_hz", "importance"], freq_rows)
        write_csv(os.path.join(sample_dir, "integrated_gradients_time_profile.csv"),
                  ["time_frame", "importance"], time_rows)

        try:
            cam = normalize_map(grad_cam_regression(model, sample))
            save_array_and_png(
                cam,
                os.path.join(sample_dir, "grad_cam"),
                f"{model_key} Grad-CAM {sample_id}",
                max_freq_hz=max_freq_hz,
                time_extent_seconds=1.0,
            )
            deletion_rows = deletion_curve(predict_fn, sample, cam)
            write_csv(
                os.path.join(sample_dir, "grad_cam_deletion_curve.csv"),
                ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
                deletion_rows,
            )
        except Exception as exc:
            write_csv(os.path.join(sample_dir, "grad_cam_status.csv"),
                      ["status", "message"], [["failed", repr(exc)]])

        occ_rows, occ_map = occlusion_importance(predict_fn, sample, groups)
        write_csv(
            os.path.join(sample_dir, "group_occlusion.csv"),
            ["group", "axis", "low", "high", "base_pred", "masked_pred", "delta", "abs_delta"],
            occ_rows,
        )
        save_array_and_png(
            normalize_map(occ_map),
            os.path.join(sample_dir, "group_occlusion_map"),
            f"{model_key} grouped occlusion {sample_id}",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )

    write_csv(
        os.path.join(out_dir, "explained_samples.csv"),
        ["sample_id", "val_local_index", "y_true", "y_pred", "abs_error"],
        sample_rows,
    )


def explain_rf_model(model, pca, scaler, x_val, y_val, pred, threshold,
                     out_dir, groups, max_freq_hz):
    predict_fn = rf_predict_fn(model, pca, scaler)
    write_rf_feature_importance(model, out_dir)
    sample_rows = []
    for label, local_idx in selected_sample_indices(y_val, pred, threshold, 1):
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = os.path.join(out_dir, sample_id)
        ensure_dir(sample_dir)

        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])

        save_input_spectrogram_png(
            sample,
            os.path.join(sample_dir, "input_spectrogram.png"),
            f"rf input {sample_id}",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )

        occ_rows, occ_map = occlusion_importance(predict_fn, sample, groups)
        write_csv(
            os.path.join(sample_dir, "group_occlusion.csv"),
            ["group", "axis", "low", "high", "base_pred", "masked_pred", "delta", "abs_delta"],
            occ_rows,
        )
        occ_norm = normalize_map(occ_map)
        save_array_and_png(
            occ_norm,
            os.path.join(sample_dir, "group_occlusion_map"),
            f"rf grouped occlusion {sample_id}",
            max_freq_hz=max_freq_hz,
            time_extent_seconds=1.0,
        )
        deletion_rows = deletion_curve(predict_fn, sample, occ_norm)
        write_csv(
            os.path.join(sample_dir, "occlusion_deletion_curve.csv"),
            ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
            deletion_rows,
        )
        freq_rows, time_rows = summarize_map_by_axis(occ_norm, max_freq_hz)
        write_csv(os.path.join(sample_dir, "occlusion_frequency_profile.csv"),
                  ["frequency_bin", "frequency_hz", "importance"], freq_rows)
        write_csv(os.path.join(sample_dir, "occlusion_time_profile.csv"),
                  ["time_frame", "importance"], time_rows)

    write_csv(
        os.path.join(out_dir, "explained_samples.csv"),
        ["sample_id", "val_local_index", "y_true", "y_pred", "abs_error"],
        sample_rows,
    )


def main():
    set_seed(EXPLAIN_CONFIG["run"]["random_seed"])
    smoke = EXPLAIN_CONFIG["run"]["smoke_test"]
    epochs = EXPLAIN_CONFIG["run"]["smoke_epochs"] if smoke else EXPLAIN_CONFIG["run"]["epochs"]
    fold_target = EXPLAIN_CONFIG["run"]["fold"]
    pca_components = EXPLAIN_CONFIG["run"]["pca_components"]
    ig_steps = 4 if smoke else EXPLAIN_CONFIG["run"]["ig_steps"]

    jobs = regression_run.build_dataset_jobs()
    if not jobs:
        raise FileNotFoundError("No dataset jobs were found from run_ensemble_regression_onb.py config.")
    job = jobs[0]
    threshold = job["threshold"]
    max_freq_hz = EXPLAIN_CONFIG["data"]["max_freq_hz"]

    loader = DataLoadingConversion()
    x, y = loader.load_npy_data(job["data_path"])
    kf = KFold(n_splits=regression_run.DIVISIONS, shuffle=True,
               random_state=regression_run.RANDOM_SEED)
    splits = list(kf.split(x))
    if not 1 <= fold_target <= len(splits):
        raise ValueError(f"fold must be 1..{len(splits)}, got {fold_target}.")
    train_index, val_index = splits[fold_target - 1]
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]

    groups = make_axis_groups(x.shape[1], x.shape[2], max_freq_hz)
    result_date_dir = EXPLAIN_CONFIG["data"]["result_date_dir"]
    if smoke and not result_date_dir.endswith("_smoke"):
        result_date_dir = result_date_dir + "_smoke"
    out_root = Path(job["experiment_root"]) / "explainability_result" / "npy" / result_date_dir
    run_out = out_root / job["experiment_name"] / job["noise_dir_name"] / job["max_freq_hz"] / f"fold{fold_target}"
    ensure_dir(str(run_out))

    write_csv(
        str(run_out / "run_config.csv"),
        ["key", "value"],
        [
            ["experiment_name", job["experiment_name"]],
            ["data_path", job["data_path"]],
            ["threshold", threshold],
            ["fold", fold_target],
            ["epochs", epochs],
            ["smoke_test", int(smoke)],
            ["ig_steps", ig_steps],
        ],
    )

    trainer = ModelTrainer(random_seed=EXPLAIN_CONFIG["run"]["random_seed"])
    mm = RegressionModelMaker((224, 224, regression_run.COLOR_CHANNEL))
    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

    for model_key in ("rf", "alexnet", "cnntf_v2_gap"):
        print(f"\n=== explain {model_key} ===")
        K.clear_session()
        gc.collect()
        spec = model_spec(model_key)
        model_out = run_out / model_key
        ensure_dir(str(model_out))

        if spec["kind"] == "sklearn":
            x_train_flat = x_train.reshape(x_train.shape[0], -1)
            x_val_flat = x_val.reshape(x_val.shape[0], -1)
            pca = PCA(
                n_components=min(pca_components, x_train_flat.shape[0], x_train_flat.shape[1]),
                random_state=EXPLAIN_CONFIG["run"]["random_seed"],
            )
            x_train_pca = pca.fit_transform(x_train_flat)
            x_val_pca = pca.transform(x_val_flat)
            model = spec["builder"](mm, **spec.get("builder_params", {}))
            model.fit(x_train_pca, y_train_scaled.ravel())
            pred_scaled = model.predict(x_val_pca).reshape(-1, 1)
            pred = scaler.inverse_transform(pred_scaled).ravel()
            write_model_metrics(str(model_out / "validation_metrics.csv"), y_val, pred)
            explain_rf_model(model, pca, scaler, x_val, y_val, pred, threshold,
                             str(model_out), groups, max_freq_hz)
        else:
            model, _ = trainer.train_one_model(spec, mm, x_train, y_train_scaled, None, epochs)
            pred = keras_predict_fn(model, scaler)(x_val)
            write_model_metrics(str(model_out / "validation_metrics.csv"), y_val, pred)
            explain_keras_model(model_key, model, scaler, x_val, y_val, pred, threshold,
                                str(model_out), groups, max_freq_hz, ig_steps)

        K.clear_session()
        gc.collect()

    print(f"\nexplainability outputs saved: {run_out}")


if __name__ == "__main__":
    main()
