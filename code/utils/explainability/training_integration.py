import os

import numpy as np

from utils.explainability.spectrogram_explainers import (
    deletion_curve,
    ensure_dir,
    grad_cam_regression,
    integrated_gradients,
    make_axis_groups,
    normalize_map,
    occlusion_importance,
    save_array_and_png,
    summarize_map_by_axis,
    write_csv,
)


def parse_max_freq_hz(max_freq_name):
    text = str(max_freq_name).lower().replace("maxfreq=", "").strip()
    if text.endswith("khz"):
        return float(text[:-3]) * 1000.0
    if text.endswith("hz"):
        return float(text[:-2])
    return float(text)


def selected_sample_indices(y_val, pred, threshold, max_samples=5):
    y_val = np.asarray(y_val)
    pred = np.asarray(pred)
    candidates = [
        ("low_heatflux", int(np.argmin(y_val))),
        ("high_heatflux", int(np.argmax(y_val))),
        ("best_prediction", int(np.argmin(np.abs(y_val - pred)))),
        ("worst_prediction", int(np.argmax(np.abs(y_val - pred)))),
    ]
    if threshold is not None and np.isfinite(float(threshold)):
        candidates.insert(1, ("near_onb", int(np.argmin(np.abs(y_val - threshold)))))

    selected = []
    used = set()
    for label, idx in candidates:
        if idx in used:
            continue
        selected.append((label, idx))
        used.add(idx)
        if len(selected) >= int(max_samples):
            break
    return selected


def keras_predict_fn(model, scaler):
    def predict(batch):
        pred_scaled = model.predict(batch, verbose=0)
        return scaler.inverse_transform(pred_scaled).ravel()
    return predict


def sklearn_predict_fn(model, pca, scaler):
    def predict(batch):
        flat = batch.reshape(batch.shape[0], -1)
        pred_scaled = model.predict(pca.transform(flat)).reshape(-1, 1)
        return scaler.inverse_transform(pred_scaled).ravel()
    return predict


def _write_status(out_dir, status, message):
    write_csv(os.path.join(out_dir, "explainability_status.csv"),
              ["status", "message"], [[status, message]])


def _sample_output_dir(out_dir, sample_id):
    sample_dir = os.path.join(out_dir, sample_id)
    ensure_dir(sample_dir)
    return sample_dir


def _write_sample_index(out_dir, rows):
    write_csv(
        os.path.join(out_dir, "explained_samples.csv"),
        ["sample_id", "val_local_index", "y_true", "y_pred", "abs_error"],
        rows,
    )


def explain_keras_model(model_key, model, scaler, x_val, y_val, pred, threshold,
                        out_dir, max_freq_hz, config):
    methods = set(config.get("methods", ["integrated_gradients", "grad_cam", "occlusion"]))
    ig_steps = int(config.get("ig_steps", 32))
    max_samples = int(config.get("max_samples_per_fold", 5))
    groups = make_axis_groups(x_val.shape[1], x_val.shape[2], max_freq_hz)
    predict_fn = keras_predict_fn(model, scaler)
    sample_rows = []

    for label, local_idx in selected_sample_indices(y_val, pred, threshold, max_samples):
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = _sample_output_dir(out_dir, sample_id)
        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])

        if "integrated_gradients" in methods:
            ig = normalize_map(integrated_gradients(model, sample, steps=ig_steps))
            save_array_and_png(ig, os.path.join(sample_dir, "integrated_gradients"),
                               f"{model_key} IG {sample_id}")
            write_csv(
                os.path.join(sample_dir, "integrated_gradients_deletion_curve.csv"),
                ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
                deletion_curve(predict_fn, sample, ig),
            )
            freq_rows, time_rows = summarize_map_by_axis(ig, max_freq_hz)
            write_csv(os.path.join(sample_dir, "integrated_gradients_frequency_profile.csv"),
                      ["frequency_bin", "frequency_hz", "importance"], freq_rows)
            write_csv(os.path.join(sample_dir, "integrated_gradients_time_profile.csv"),
                      ["time_frame", "importance"], time_rows)

        if "grad_cam" in methods:
            try:
                cam = normalize_map(grad_cam_regression(model, sample))
                save_array_and_png(cam, os.path.join(sample_dir, "grad_cam"),
                                   f"{model_key} Grad-CAM {sample_id}")
                write_csv(
                    os.path.join(sample_dir, "grad_cam_deletion_curve.csv"),
                    ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
                    deletion_curve(predict_fn, sample, cam),
                )
            except Exception as exc:
                write_csv(os.path.join(sample_dir, "grad_cam_status.csv"),
                          ["status", "message"], [["failed", repr(exc)]])

        if "occlusion" in methods:
            occ_rows, occ_map = occlusion_importance(predict_fn, sample, groups)
            write_csv(
                os.path.join(sample_dir, "group_occlusion.csv"),
                ["group", "axis", "low", "high", "base_pred", "masked_pred", "delta", "abs_delta"],
                occ_rows,
            )
            save_array_and_png(normalize_map(occ_map), os.path.join(sample_dir, "group_occlusion_map"),
                               f"{model_key} grouped occlusion {sample_id}")

    _write_sample_index(out_dir, sample_rows)


def explain_sklearn_model(model_key, model, pca, scaler, x_val, y_val, pred, threshold,
                          out_dir, max_freq_hz, config):
    if pca is None:
        _write_status(out_dir, "skipped", "PCA object was not retained for this run.")
        return

    if hasattr(model, "feature_importances_"):
        rows = [[i, float(v)] for i, v in enumerate(np.asarray(model.feature_importances_).ravel())]
        rows.sort(key=lambda row: row[1], reverse=True)
        write_csv(os.path.join(out_dir, "pca_feature_importance.csv"),
                  ["pca_component", "importance"], rows)

    max_samples = int(config.get("max_samples_per_fold", 5))
    groups = make_axis_groups(x_val.shape[1], x_val.shape[2], max_freq_hz)
    predict_fn = sklearn_predict_fn(model, pca, scaler)
    sample_rows = []
    for label, local_idx in selected_sample_indices(y_val, pred, threshold, max_samples):
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = _sample_output_dir(out_dir, sample_id)
        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])

        occ_rows, occ_map = occlusion_importance(predict_fn, sample, groups)
        write_csv(
            os.path.join(sample_dir, "group_occlusion.csv"),
            ["group", "axis", "low", "high", "base_pred", "masked_pred", "delta", "abs_delta"],
            occ_rows,
        )
        occ_norm = normalize_map(occ_map)
        save_array_and_png(occ_norm, os.path.join(sample_dir, "group_occlusion_map"),
                           f"{model_key} grouped occlusion {sample_id}")
        write_csv(
            os.path.join(sample_dir, "occlusion_deletion_curve.csv"),
            ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
            deletion_curve(predict_fn, sample, occ_norm),
        )
        freq_rows, time_rows = summarize_map_by_axis(occ_norm, max_freq_hz)
        write_csv(os.path.join(sample_dir, "occlusion_frequency_profile.csv"),
                  ["frequency_bin", "frequency_hz", "importance"], freq_rows)
        write_csv(os.path.join(sample_dir, "occlusion_time_profile.csv"),
                  ["time_frame", "importance"], time_rows)

    _write_sample_index(out_dir, sample_rows)


def maybe_explain_trained_model(spec, model, scaler, x_val, y_val, pred, threshold,
                                save_path, fold, max_freq_name, config, pca=None):
    if not config.get("enabled", False):
        return

    target_folds = config.get("target_folds")
    if target_folds and int(fold) not in {int(v) for v in target_folds}:
        return

    model_key = spec["key"]
    if config.get("model_keys") and model_key not in set(config["model_keys"]):
        return

    max_freq_hz = parse_max_freq_hz(max_freq_name)
    out_dir = os.path.join(save_path, "explainability", f"fold{fold}", model_key)
    ensure_dir(out_dir)
    write_csv(
        os.path.join(out_dir, "explainability_config.csv"),
        ["key", "value"],
        [
            ["model_key", model_key],
            ["fold", fold],
            ["max_freq_name", max_freq_name],
            ["max_freq_hz", max_freq_hz],
            ["methods", "|".join(config.get("methods", []))],
            ["max_samples_per_fold", config.get("max_samples_per_fold", "")],
            ["ig_steps", config.get("ig_steps", "")],
        ],
    )

    if spec["kind"] == "keras":
        explain_keras_model(model_key, model, scaler, x_val, y_val, pred, threshold,
                            out_dir, max_freq_hz, config)
    elif spec["kind"] == "sklearn":
        explain_sklearn_model(model_key, model, pca, scaler, x_val, y_val, pred,
                              threshold, out_dir, max_freq_hz, config)
    else:
        _write_status(out_dir, "skipped", f"Unknown model kind: {spec['kind']}")
