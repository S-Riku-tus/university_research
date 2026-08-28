import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, r2_score, recall_score

from utils.explainability.spectrogram_explainers import (
    deletion_curve,
    ensure_dir,
    grad_cam_regression,
    insertion_curve,
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


SUMMARY_HEADER = [
    "sample_id",
    "val_local_index",
    "y_true",
    "y_pred",
    "abs_error",
    "method",
    "attribution_units",
    "attribution_sum",
    "output_delta_from_baseline",
    "completeness_error",
    "completeness_relative_error",
    "deletion_prediction_auc",
    "deletion_linear_auc",
    "deletion_area_between_curve",
    "insertion_prediction_auc",
    "insertion_linear_auc",
    "insertion_area_between_curve",
    "top_frequency_bin",
    "top_frequency_hz",
    "top_time_frame",
    "top_time_s",
]

OCCLUSION_HEADER = [
    "sample_id",
    "val_local_index",
    "y_true",
    "y_pred",
    "abs_error",
    "group",
    "axis",
    "low",
    "high",
    "unit",
    "low_index",
    "high_index",
    "base_pred",
    "masked_pred",
    "delta",
    "abs_delta",
]

GROUP_MASK_PERFORMANCE_HEADER = [
    "group",
    "axis",
    "low",
    "high",
    "unit",
    "low_index",
    "high_index",
    "n_samples",
    "n_onb_samples",
    "base_r2",
    "masked_r2",
    "r2_drop",
    "base_rmse_all",
    "masked_rmse_all",
    "rmse_all_increase",
    "base_mae_all",
    "masked_mae_all",
    "mae_all_increase",
    "base_rmse_onb",
    "masked_rmse_onb",
    "rmse_onb_increase",
    "base_recall",
    "masked_recall",
    "recall_drop",
    "base_f1",
    "masked_f1",
    "f1_drop",
    "base_false_negatives",
    "masked_false_negatives",
    "false_negative_increase",
]


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
    candidates = []
    if threshold is not None and np.isfinite(float(threshold)):
        threshold = float(threshold)
        below = np.flatnonzero(y_val < threshold)
        above = np.flatnonzero(y_val >= threshold)
        if len(below):
            idx = int(below[np.argmin(threshold - y_val[below])])
            candidates.append(("near_onb_below", idx))
        if len(above):
            idx = int(above[np.argmin(y_val[above] - threshold)])
            candidates.append(("near_onb_above", idx))

        false_negative = np.flatnonzero((y_val >= threshold) & (pred < threshold))
        if len(false_negative):
            miss = y_val[false_negative] - pred[false_negative]
            candidates.append(("false_negative", int(false_negative[np.argmax(miss)])))

    candidates.extend([
        ("worst_prediction", int(np.argmax(np.abs(y_val - pred)))),
        ("low_heatflux", int(np.argmin(y_val))),
        ("high_heatflux", int(np.argmax(y_val))),
        ("best_prediction", int(np.argmin(np.abs(y_val - pred)))),
    ])

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


def _path_exists(path):
    return os.path.exists(path) or os.path.exists(windows_long_path(path))


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


def _curve_fractions(config):
    raw = config.get("curve_fractions", config.get("deletion_fractions"))
    if raw is None:
        return [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    return [float(v) for v in raw]


def _baseline_value(config):
    return float(config.get("baseline_value", 0.0))


def _frequency_bands(config):
    bands = config.get("frequency_bands_hz")
    if not bands:
        return None
    return [(float(low), float(high)) for low, high in bands]


def _axis_groups(x_val, max_freq_hz, config):
    return make_axis_groups(
        x_val.shape[1],
        x_val.shape[2],
        max_freq_hz,
        frequency_bands_hz=_frequency_bands(config),
        time_groups=int(config.get("time_groups", 4)),
        time_extent_seconds=float(config.get("time_extent_seconds", 1.0)),
    )


def _methods_for_model(config, model_key):
    by_model = config.get("methods_by_model") or {}
    methods = by_model.get(model_key, config.get("methods", []))
    return {str(method).lower() for method in methods}


def _curve_metrics(rows, prediction_index):
    """Return prediction AUC and regression area-above-straight-line.

    Hama, Mase, and Owen (JMLR 2023) show that a straight line between
    the two regression endpoints is the meaningful reference.  The returned
    signed difference retains the heat-flux output unit.
    """
    if len(rows) < 2:
        return [float("nan")] * 3
    fraction = np.asarray([float(row[0]) for row in rows], dtype=float)
    prediction = np.asarray(
        [float(row[prediction_index]) for row in rows], dtype=float)
    order = np.argsort(fraction)
    fraction = fraction[order]
    prediction = prediction[order]
    prediction_auc = float(np.trapz(prediction, fraction))
    reference = np.interp(
        fraction,
        [float(fraction[0]), float(fraction[-1])],
        [float(prediction[0]), float(prediction[-1])],
    )
    linear_auc = float(np.trapz(reference, fraction))
    return [prediction_auc, linear_auc, prediction_auc - linear_auc]


def _inverse_output_scale(scaler):
    scale = float(np.asarray(scaler.scale_).ravel()[0])
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Invalid MinMaxScaler scale for attribution: {scale}.")
    return 1.0 / scale


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _relative_l1(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b)) / (np.sum(np.abs(a)) + 1e-12))


def _regression_metrics(y_true, prediction, threshold, onb_band_frac):
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = prediction - y_true
    metrics = {
        "r2": float(r2_score(y_true, prediction)),
        "rmse_all": float(np.sqrt(np.mean(residual ** 2))),
        "mae_all": float(np.mean(np.abs(residual))),
        "rmse_onb": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "false_negatives": float("nan"),
        "n_onb": 0,
    }
    if threshold is None or not np.isfinite(float(threshold)):
        return metrics

    threshold = float(threshold)
    onb_mask = np.abs(y_true - threshold) <= abs(threshold) * float(onb_band_frac)
    metrics["n_onb"] = int(np.sum(onb_mask))
    if np.any(onb_mask):
        metrics["rmse_onb"] = float(
            np.sqrt(np.mean((prediction[onb_mask] - y_true[onb_mask]) ** 2))
        )

    true_binary = y_true >= threshold
    pred_binary = prediction >= threshold
    metrics["recall"] = float(recall_score(
        true_binary, pred_binary, zero_division=0))
    metrics["f1"] = float(f1_score(
        true_binary, pred_binary, zero_division=0))
    metrics["false_negatives"] = int(np.sum(true_binary & ~pred_binary))
    return metrics


def _group_mask_performance(predict_fn, x_val, y_val, base_pred, groups,
                            threshold, config):
    base = _regression_metrics(
        y_val, base_pred, threshold, config.get("onb_band_frac", 0.10))
    rows = []
    baseline_value = _baseline_value(config)
    for group in groups:
        masked = np.array(x_val, copy=True)
        masked[:, group["mask"], :] = baseline_value
        masked_pred = np.asarray(predict_fn(masked), dtype=float)
        current = _regression_metrics(
            y_val, masked_pred, threshold, config.get("onb_band_frac", 0.10))
        rows.append([
            group["group"], group["axis"], group["low"], group["high"],
            group.get("unit", ""), group.get("low_index", ""),
            group.get("high_index", ""), len(y_val), base["n_onb"],
            base["r2"], current["r2"], base["r2"] - current["r2"],
            base["rmse_all"], current["rmse_all"],
            current["rmse_all"] - base["rmse_all"],
            base["mae_all"], current["mae_all"],
            current["mae_all"] - base["mae_all"],
            base["rmse_onb"], current["rmse_onb"],
            current["rmse_onb"] - base["rmse_onb"],
            base["recall"], current["recall"],
            base["recall"] - current["recall"],
            base["f1"], current["f1"], base["f1"] - current["f1"],
            base["false_negatives"], current["false_negatives"],
            current["false_negatives"] - base["false_negatives"],
        ])
    return rows


def _top_axis_bins(values, max_freq_hz, time_extent_seconds):
    arr = np.asarray(values, dtype=np.float32)
    freq_profile = arr.mean(axis=0)
    time_profile = arr.mean(axis=1)
    top_freq_bin = int(np.argmax(freq_profile))
    top_time_frame = int(np.argmax(time_profile))
    top_freq_hz = float(max_freq_hz * (top_freq_bin + 0.5) / len(freq_profile))
    top_time_s = float(
        time_extent_seconds * (top_time_frame + 0.5) / len(time_profile)
    )
    return top_freq_bin, top_freq_hz, top_time_frame, top_time_s


def _write_attribution_outputs(
    method,
    model_key,
    sample_id,
    local_idx,
    y_true,
    y_pred,
    values,
    predict_fn,
    sample,
    sample_dir,
    max_freq_hz,
    config,
    signed=False,
    units="relative importance",
    compute_curves=True,
):
    raw_values = np.nan_to_num(np.asarray(values, dtype=np.float32))
    importance = (
        normalize_magnitude(raw_values)
        if signed
        else normalize_map(raw_values)
    )
    baseline_value = _baseline_value(config)
    baseline = np.full_like(sample, baseline_value, dtype=np.float32)
    baseline_pred = float(np.ravel(predict_fn(baseline[None, ...]))[0])
    output_delta = float(y_pred) - baseline_pred
    time_extent_seconds = float(config.get("time_extent_seconds", 1.0))
    plot_title = (
        f"{model_key} {method} {sample_id}\n"
        f"prediction - baseline = {output_delta:,.3g} heat-flux units"
    )
    if config.get("save_maps", True):
        if signed:
            save_signed_array_and_png(
                raw_values,
                os.path.join(sample_dir, f"{method}_signed"),
                plot_title + " (signed)",
                unit=units,
                max_freq_hz=max_freq_hz,
                time_extent_seconds=time_extent_seconds,
            )
            save_array_and_png(
                importance,
                os.path.join(sample_dir, f"{method}_magnitude"),
                plot_title + " (magnitude)",
                max_freq_hz=max_freq_hz,
                time_extent_seconds=time_extent_seconds,
            )
        else:
            save_array_and_png(
                importance,
                os.path.join(sample_dir, method),
                plot_title,
                max_freq_hz=max_freq_hz,
                time_extent_seconds=time_extent_seconds,
            )

    if compute_curves:
        fractions = _curve_fractions(config)
        deletion_rows = deletion_curve(
            predict_fn, sample, importance,
            fractions=fractions, baseline_value=baseline_value,
        )
        write_csv(
            os.path.join(sample_dir, f"{method}_deletion_curve.csv"),
            ["masked_fraction", "masked_pixels", "base_pred", "masked_pred", "delta", "abs_delta"],
            deletion_rows,
        )

        insertion_rows = insertion_curve(
            predict_fn, sample, importance,
            fractions=fractions, baseline_value=baseline_value,
        )
        write_csv(
            os.path.join(sample_dir, f"{method}_insertion_curve.csv"),
            [
                "inserted_fraction", "inserted_pixels", "original_pred",
                "baseline_pred", "inserted_pred", "delta_from_baseline",
                "abs_delta_from_baseline", "remaining_delta_to_original",
                "abs_remaining_delta",
            ],
            insertion_rows,
        )
        deletion_metrics = _curve_metrics(deletion_rows, 3)
        insertion_metrics = _curve_metrics(insertion_rows, 4)
    else:
        deletion_metrics = [float("nan")] * 3
        insertion_metrics = [float("nan")] * 3

    freq_rows, time_rows = summarize_map_by_axis(importance, max_freq_hz)
    write_csv(os.path.join(sample_dir, f"{method}_frequency_profile.csv"),
              ["frequency_bin", "frequency_hz", "importance"], freq_rows)
    time_frame_count = max(1, len(time_rows))
    write_csv(
        os.path.join(sample_dir, f"{method}_time_profile.csv"),
        ["time_frame", "time_center_s", "importance"],
        [
            [
                int(row[0]),
                time_extent_seconds * (int(row[0]) + 0.5) / time_frame_count,
                float(row[1]),
            ]
            for row in time_rows
        ],
    )

    if signed:
        signed_frequency = np.sum(raw_values, axis=0)
        signed_time = np.sum(raw_values, axis=1)
        write_csv(
            os.path.join(sample_dir, f"{method}_signed_frequency_profile.csv"),
            ["frequency_bin", "frequency_hz", f"contribution_{units}"],
            [
                [i, max_freq_hz * (i + 0.5) / len(signed_frequency), float(value)]
                for i, value in enumerate(signed_frequency)
            ],
        )
        write_csv(
            os.path.join(sample_dir, f"{method}_signed_time_profile.csv"),
            ["time_frame", "time_center_s", f"contribution_{units}"],
            [
                [
                    i,
                    time_extent_seconds * (i + 0.5) / len(signed_time),
                    float(value),
                ]
                for i, value in enumerate(signed_time)
            ],
        )

    attribution_sum = float(np.sum(raw_values)) if signed else float("nan")
    completeness_error = (
        attribution_sum - output_delta if signed else float("nan"))
    completeness_relative_error = (
        abs(completeness_error) / (abs(output_delta) + 1e-12)
        if signed else float("nan")
    )
    top_freq_bin, top_freq_hz, top_time_frame, top_time_s = _top_axis_bins(
        importance, max_freq_hz, time_extent_seconds)
    return [
        sample_id,
        int(local_idx),
        float(y_true),
        float(y_pred),
        abs(float(y_true) - float(y_pred)),
        method,
        units,
        attribution_sum,
        output_delta,
        completeness_error,
        completeness_relative_error,
        *deletion_metrics,
        *insertion_metrics,
        top_freq_bin,
        top_freq_hz,
        top_time_frame,
        top_time_s,
    ]


def _append_group_rows(target_rows, sample_id, local_idx, y_true, y_pred, occ_rows):
    abs_error = abs(float(y_true) - float(y_pred))
    for row in occ_rows:
        target_rows.append([
            sample_id,
            int(local_idx),
            float(y_true),
            float(y_pred),
            abs_error,
            *row,
        ])


def _write_tree_shap_pca(model, pca, scaler, x_val, selected_samples, out_dir):
    """Write native XGBoost TreeSHAP values for the RF's PCA features.

    These values faithfully explain the fitted XGBRFRegressor, but their
    features are PCA components rather than physical time/frequency regions.
    They are therefore a model audit, not the physical explanation used for
    the boiling-acoustics claim.
    """
    try:
        from xgboost import DMatrix

        local_indices = [int(local_idx) for _, local_idx in selected_samples]
        flat = x_val[local_indices].reshape(len(local_indices), -1)
        x_pca = pca.transform(flat)
        contributions = np.asarray(
            model.get_booster().predict(
                DMatrix(x_pca), pred_contribs=True, approx_contribs=False),
            dtype=float,
        )
        if contributions.ndim != 2 or contributions.shape[1] != x_pca.shape[1] + 1:
            raise ValueError(
                "Unexpected pred_contribs shape: "
                f"{contributions.shape}; expected (*, {x_pca.shape[1] + 1})."
            )

        inverse_scale = _inverse_output_scale(scaler)
        model_pred_scaled = np.asarray(model.predict(x_pca), dtype=float).ravel()
        model_pred_heat_flux = scaler.inverse_transform(
            model_pred_scaled.reshape(-1, 1)).ravel()
        detail_rows = []
        summary_rows = []
        for sample_i, ((label, local_idx), row) in enumerate(
                zip(selected_samples, contributions)):
            sample_id = f"{label}_val{int(local_idx):04d}"
            shap_scaled = row[:-1]
            bias_scaled = float(row[-1])
            shap_heat_flux = shap_scaled * inverse_scale
            expected_heat_flux = float(scaler.inverse_transform(
                np.asarray([[bias_scaled]], dtype=float))[0, 0])
            reconstructed = expected_heat_flux + float(np.sum(shap_heat_flux))
            completeness_error = reconstructed - float(model_pred_heat_flux[sample_i])
            order = np.argsort(np.abs(shap_heat_flux))[::-1]
            rank_by_component = np.empty_like(order)
            rank_by_component[order] = np.arange(1, len(order) + 1)
            for component, (value_scaled, value_heat_flux) in enumerate(
                    zip(shap_scaled, shap_heat_flux)):
                detail_rows.append([
                    sample_id,
                    int(local_idx),
                    component,
                    int(rank_by_component[component]),
                    float(x_pca[sample_i, component]),
                    float(value_scaled),
                    float(value_heat_flux),
                    abs(float(value_heat_flux)),
                ])
            summary_rows.append([
                sample_id,
                int(local_idx),
                float(model_pred_heat_flux[sample_i]),
                expected_heat_flux,
                float(np.sum(shap_heat_flux)),
                reconstructed,
                completeness_error,
            ])

        write_csv(
            os.path.join(out_dir, "treeshap_pca_values.csv"),
            [
                "sample_id", "val_local_index", "pca_component", "abs_rank",
                "pca_value", "shap_scaled_output", "shap_heat_flux",
                "abs_shap_heat_flux",
            ],
            detail_rows,
        )
        write_csv(
            os.path.join(out_dir, "treeshap_pca_summary.csv"),
            [
                "sample_id", "val_local_index", "model_prediction_heat_flux",
                "expected_value_heat_flux", "sum_shap_heat_flux",
                "reconstructed_prediction_heat_flux", "completeness_error",
            ],
            summary_rows,
        )
        write_csv(
            os.path.join(out_dir, "treeshap_status.csv"),
            ["status", "method", "physical_interpretation", "message"],
            [[
                "complete",
                "xgboost.Booster.predict(pred_contribs=True)",
                "not_direct",
                (
                    "Exact TreeSHAP contributions were saved in PCA space. "
                    "Do not label PCA components as physical time/frequency bands; "
                    "use group_mask_performance.csv for the physical comparison."
                ),
            ]],
        )
    except Exception as exc:
        write_csv(
            os.path.join(out_dir, "treeshap_status.csv"),
            ["status", "method", "physical_interpretation", "message"],
            [["failed", "xgboost pred_contribs", "not_direct", repr(exc)]],
        )


def _keras_attribution(method, model, sample, scaler, config):
    if method == "integrated_gradients":
        baseline = np.full_like(
            sample, _baseline_value(config), dtype=np.float32)
        values_scaled = integrated_gradients(
            model, sample, baseline=baseline,
            steps=int(config.get("ig_steps", 32)),
        )
        return values_scaled * _inverse_output_scale(scaler), True, "heat_flux"
    if method == "grad_cam":
        return grad_cam_regression(model, sample), False, "relative_importance"
    raise ValueError(f"Unsupported Keras attribution method: {method}")


def _input_stability_rows(model, scaler, sample_records, base_attributions,
                          config):
    stability = config.get("stability") or {}
    if not stability.get("enabled", False):
        return []
    repeats = int(stability.get("repeats", 2))
    noise_fraction = float(stability.get("noise_fraction", 0.01))
    seed = int(stability.get("random_seed", 42))
    methods = {
        str(v).lower()
        for v in stability.get("methods", ["integrated_gradients"])
    }
    rows = []
    for record_i, record in enumerate(sample_records):
        sample_id = record["sample_id"]
        sample = record["sample"]
        sample_scale = max(float(np.std(sample)), 1e-12)
        for method in methods:
            base_values = base_attributions.get((sample_id, method))
            if base_values is None:
                continue
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + 1009 * record_i + repeat)
                noisy = sample + rng.normal(
                    0.0, noise_fraction * sample_scale, size=sample.shape)
                if stability.get("clip_nonnegative", True):
                    noisy = np.maximum(noisy, 0.0)
                noisy = noisy.astype(np.float32)
                noisy_values, _, _ = _keras_attribution(
                    method, model, noisy, scaler, config)
                base_magnitude = np.abs(base_values)
                noisy_magnitude = np.abs(noisy_values)
                rows.append([
                    sample_id,
                    method,
                    repeat + 1,
                    noise_fraction,
                    _safe_corr(base_magnitude, noisy_magnitude),
                    _cosine_similarity(base_magnitude, noisy_magnitude),
                    _relative_l1(base_magnitude, noisy_magnitude),
                ])
    return rows


def _randomize_top_layer(model, seed):
    rng = np.random.default_rng(int(seed))
    for layer in reversed(model.layers):
        original = layer.get_weights()
        if not layer.trainable or not original:
            continue
        randomized = []
        for values in original:
            values = np.asarray(values)
            scale = float(np.std(values))
            if not np.isfinite(scale) or scale <= 1e-12:
                fan_in = values.shape[0] if values.ndim else 1
                scale = 1.0 / np.sqrt(max(int(fan_in), 1))
            randomized.append(
                rng.normal(0.0, scale, size=values.shape).astype(values.dtype))
        layer.set_weights(randomized)
        return layer, original
    return None, None


def _top_layer_sanity_rows(model, scaler, sample_records, base_attributions,
                           config):
    sanity = config.get("sanity_check") or {}
    if not sanity.get("enabled", False):
        return []
    methods = {
        str(v).lower()
        for v in sanity.get("methods", ["integrated_gradients"])
    }
    layer, original_weights = _randomize_top_layer(
        model, sanity.get("random_seed", 42))
    if layer is None:
        return []

    randomized_predict = keras_predict_fn(model, scaler)
    rows = []
    try:
        for record in sample_records:
            sample_id = record["sample_id"]
            sample = record["sample"]
            for method in methods:
                base_values = base_attributions.get((sample_id, method))
                if base_values is None:
                    continue
                randomized_values, _, _ = _keras_attribution(
                    method, model, sample, scaler, config)
                base_magnitude = np.abs(base_values)
                randomized_magnitude = np.abs(randomized_values)
                rows.append([
                    sample_id,
                    method,
                    layer.name,
                    float(record["y_pred"]),
                    float(np.ravel(randomized_predict(sample[None, ...]))[0]),
                    _safe_corr(base_magnitude, randomized_magnitude),
                    _cosine_similarity(base_magnitude, randomized_magnitude),
                    _relative_l1(base_magnitude, randomized_magnitude),
                ])
    finally:
        layer.set_weights(original_weights)
    return rows


def explain_keras_model(model_key, model, scaler, x_val, y_val, pred, threshold,
                        out_dir, max_freq_hz, config):
    methods = _methods_for_model(config, model_key)
    max_samples = int(config.get("max_samples_per_fold", 5))
    groups = _axis_groups(x_val, max_freq_hz, config)
    predict_fn = keras_predict_fn(model, scaler)
    selected_samples = selected_sample_indices(
        y_val, pred, threshold, max_samples)
    sample_rows = []
    summary_rows = []
    group_rows = []
    sample_records = []
    base_attributions = {}

    if "group_occlusion" in methods or "occlusion" in methods:
        performance_rows = _group_mask_performance(
            predict_fn, x_val, y_val, pred, groups, threshold, config)
        write_csv(
            os.path.join(out_dir, "group_mask_performance.csv"),
            GROUP_MASK_PERFORMANCE_HEADER,
            performance_rows,
        )

    for label, local_idx in selected_samples:
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = _sample_output_dir(out_dir, sample_id)
        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])
        sample_records.append({
            "sample_id": sample_id,
            "local_idx": int(local_idx),
            "sample": sample,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if config.get("save_maps", True):
            save_input_spectrogram_png(
                sample,
                os.path.join(sample_dir, "input_spectrogram.png"),
                f"{model_key} input {sample_id}",
                max_freq_hz=max_freq_hz,
                time_extent_seconds=float(
                    config.get("time_extent_seconds", 1.0)
                ),
            )

        if "integrated_gradients" in methods:
            ig, signed, units = _keras_attribution(
                "integrated_gradients", model, sample, scaler, config)
            base_attributions[(sample_id, "integrated_gradients")] = ig
            summary_rows.append(_write_attribution_outputs(
                "integrated_gradients", model_key, sample_id, local_idx,
                y_true, y_pred, ig, predict_fn, sample, sample_dir,
                max_freq_hz, config, signed=signed, units=units,
            ))

        if "grad_cam" in methods:
            try:
                cam, signed, units = _keras_attribution(
                    "grad_cam", model, sample, scaler, config)
                base_attributions[(sample_id, "grad_cam")] = cam
                summary_rows.append(_write_attribution_outputs(
                    "grad_cam", model_key, sample_id, local_idx,
                    y_true, y_pred, cam, predict_fn, sample, sample_dir,
                    max_freq_hz, config, signed=signed, units=units,
                ))
            except Exception as exc:
                write_csv(os.path.join(sample_dir, "grad_cam_status.csv"),
                          ["status", "message"], [["failed", repr(exc)]])

        if "occlusion" in methods or "group_occlusion" in methods:
            occ_rows, occ_map, signed_occ_map = occlusion_importance(
                predict_fn,
                sample,
                groups,
                baseline_value=_baseline_value(config),
                return_signed_map=True,
            )
            write_csv(
                os.path.join(sample_dir, "group_occlusion.csv"),
                [
                    "group", "axis", "low", "high", "unit", "low_index",
                    "high_index", "base_pred", "masked_pred", "delta", "abs_delta",
                ],
                occ_rows,
            )
            if config.get("save_maps", True):
                save_signed_array_and_png(
                    signed_occ_map,
                    os.path.join(sample_dir, "group_occlusion_signed"),
                    f"{model_key} signed grouped occlusion\n{sample_id}",
                    unit="heat_flux_change",
                    max_freq_hz=max_freq_hz,
                    time_extent_seconds=float(
                        config.get("time_extent_seconds", 1.0)
                    ),
                )
            _append_group_rows(group_rows, sample_id, local_idx, y_true, y_pred, occ_rows)
            summary_rows.append(_write_attribution_outputs(
                "group_occlusion", model_key, sample_id, local_idx,
                y_true, y_pred, occ_map, predict_fn, sample, sample_dir,
                max_freq_hz, config, units="heat_flux_change",
                compute_curves=False,
            ))

    stability_rows = _input_stability_rows(
        model, scaler, sample_records, base_attributions, config)
    if stability_rows:
        write_csv(
            os.path.join(out_dir, "input_stability.csv"),
            [
                "sample_id", "method", "repeat", "noise_fraction_of_sample_std",
                "pearson_abs_map", "cosine_abs_map", "relative_l1_abs_map",
            ],
            stability_rows,
        )

    sanity_rows = _top_layer_sanity_rows(
        model, scaler, sample_records, base_attributions, config)
    if sanity_rows:
        write_csv(
            os.path.join(out_dir, "top_layer_randomization_sanity.csv"),
            [
                "sample_id", "method", "randomized_layer", "original_prediction",
                "randomized_prediction", "pearson_abs_map", "cosine_abs_map",
                "relative_l1_abs_map",
            ],
            sanity_rows,
        )

    _write_sample_index(out_dir, sample_rows)
    write_csv(os.path.join(out_dir, "explainability_summary.csv"), SUMMARY_HEADER, summary_rows)
    if group_rows:
        write_csv(os.path.join(out_dir, "group_occlusion_summary.csv"),
                  OCCLUSION_HEADER, group_rows)


def explain_sklearn_model(model_key, model, pca, scaler, x_val, y_val, pred, threshold,
                          out_dir, max_freq_hz, config):
    if pca is None:
        _write_status(out_dir, "skipped", "PCA object was not retained for this run.")
        return

    methods = _methods_for_model(config, model_key)
    if hasattr(model, "feature_importances_"):
        rows = [[i, float(v)] for i, v in enumerate(np.asarray(model.feature_importances_).ravel())]
        rows.sort(key=lambda row: row[1], reverse=True)
        write_csv(os.path.join(out_dir, "pca_feature_importance.csv"),
                  ["pca_component", "importance"], rows)

    max_samples = int(config.get("max_samples_per_fold", 5))
    groups = _axis_groups(x_val, max_freq_hz, config)
    predict_fn = sklearn_predict_fn(model, pca, scaler)
    selected_samples = selected_sample_indices(
        y_val, pred, threshold, max_samples)

    if "tree_shap_pca" in methods or "treeshap" in methods:
        _write_tree_shap_pca(
            model, pca, scaler, x_val, selected_samples, out_dir)

    if "group_occlusion" in methods or "occlusion" in methods:
        performance_rows = _group_mask_performance(
            predict_fn, x_val, y_val, pred, groups, threshold, config)
        write_csv(
            os.path.join(out_dir, "group_mask_performance.csv"),
            GROUP_MASK_PERFORMANCE_HEADER,
            performance_rows,
        )

    sample_rows = []
    summary_rows = []
    group_rows = []

    for label, local_idx in selected_samples:
        sample = x_val[local_idx]
        sample_id = f"{label}_val{local_idx:04d}"
        sample_dir = _sample_output_dir(out_dir, sample_id)
        y_true = float(y_val[local_idx])
        y_pred = float(pred[local_idx])
        sample_rows.append([sample_id, int(local_idx), y_true, y_pred, abs(y_true - y_pred)])
        if config.get("save_maps", True):
            save_input_spectrogram_png(
                sample,
                os.path.join(sample_dir, "input_spectrogram.png"),
                f"{model_key} input {sample_id}",
                max_freq_hz=max_freq_hz,
                time_extent_seconds=float(
                    config.get("time_extent_seconds", 1.0)
                ),
            )

        if "group_occlusion" in methods or "occlusion" in methods:
            occ_rows, occ_map, signed_occ_map = occlusion_importance(
                predict_fn,
                sample,
                groups,
                baseline_value=_baseline_value(config),
                return_signed_map=True,
            )
            write_csv(
                os.path.join(sample_dir, "group_occlusion.csv"),
                [
                    "group", "axis", "low", "high", "unit", "low_index",
                    "high_index", "base_pred", "masked_pred", "delta", "abs_delta",
                ],
                occ_rows,
            )
            if config.get("save_maps", True):
                save_signed_array_and_png(
                    signed_occ_map,
                    os.path.join(sample_dir, "group_occlusion_signed"),
                    f"{model_key} signed grouped occlusion\n{sample_id}",
                    unit="heat_flux_change",
                    max_freq_hz=max_freq_hz,
                    time_extent_seconds=float(
                        config.get("time_extent_seconds", 1.0)
                    ),
                )
            _append_group_rows(group_rows, sample_id, local_idx, y_true, y_pred, occ_rows)
            summary_rows.append(_write_attribution_outputs(
                "group_occlusion", model_key, sample_id, local_idx,
                y_true, y_pred, occ_map, predict_fn, sample, sample_dir,
                max_freq_hz, config, units="heat_flux_change",
                compute_curves=False,
            ))

    _write_sample_index(out_dir, sample_rows)
    write_csv(os.path.join(out_dir, "explainability_summary.csv"), SUMMARY_HEADER, summary_rows)
    if group_rows:
        write_csv(os.path.join(out_dir, "group_occlusion_summary.csv"),
                  OCCLUSION_HEADER, group_rows)


def explainability_condition_selected(config, experiment_name, max_freq_name,
                                      noise_dir_name):
    filters = config.get("condition_filter") or {}
    checks = [
        ("experiment_names", experiment_name),
        ("max_freq_hz_list", max_freq_name),
        ("noise_dir_names", noise_dir_name),
    ]
    for key, actual in checks:
        allowed = filters.get(key)
        if allowed and actual not in set(allowed):
            return False
    return True


def resolve_explainability_scope(
    config,
    experiment_names,
    max_freq_hz_list,
    noise_dir_names,
    model_keys,
    fold_count,
):
    """Derive the XAI execution scope from the parent validation settings.

    The run configuration should have one source of truth for datasets, models,
    and folds.  The resolved fields are retained in manifests and per-model XAI
    configuration CSV files so a completed run remains auditable.
    """
    resolved = dict(config or {})
    enabled = bool(resolved.get("enabled", False))
    resolved["condition_filter"] = {
        "experiment_names": list(experiment_names),
        "max_freq_hz_list": list(max_freq_hz_list),
        "noise_dir_names": list(noise_dir_names),
    }
    resolved["model_keys"] = list(model_keys)
    resolved["target_folds"] = list(range(1, int(fold_count) + 1))
    resolved["save_maps"] = enabled
    return resolved


def aggregate_group_mask_comparison(save_path, config, model_keys, fold_count):
    """Collect model-wise physical mask effects into presentation-ready CSVs."""
    if not config.get("enabled", False):
        return False
    target_folds = config.get("target_folds")
    folds = (
        [int(value) for value in target_folds]
        if target_folds else list(range(1, int(fold_count) + 1))
    )
    collected = []
    for fold in folds:
        for model_key in model_keys:
            path = os.path.join(
                save_path,
                "explainability",
                f"fold{fold}",
                model_key,
                "group_mask_performance.csv",
            )
            if not _path_exists(path):
                continue
            with open(windows_long_path(path), "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                row["fold"] = fold
                row["model_key"] = model_key
                collected.append(row)

    if not collected:
        return False

    rank_metrics = ["r2_drop", "rmse_onb_increase", "recall_drop"]
    for fold in folds:
        for model_key in model_keys:
            for axis in ("frequency", "time"):
                subset = [
                    row for row in collected
                    if int(row["fold"]) == fold
                    and row["model_key"] == model_key
                    and row["axis"] == axis
                ]
                for metric_name in rank_metrics:
                    def score(row):
                        try:
                            value = float(row[metric_name])
                        except (TypeError, ValueError):
                            return float("-inf")
                        return value if np.isfinite(value) else float("-inf")

                    ordered = sorted(
                        [row for row in subset if score(row) > float("-inf")],
                        key=score,
                        reverse=True,
                    )
                    for rank, row in enumerate(ordered, start=1):
                        row[f"{metric_name}_rank"] = rank

    original_header = list(collected[0].keys())
    prefix = ["fold", "model_key"]
    rank_headers = [f"{name}_rank" for name in rank_metrics]
    value_headers = [
        name for name in original_header
        if name not in set(prefix + rank_headers)
    ]
    comparison_header = prefix + value_headers + rank_headers
    comparison_rows = [
        [row.get(name, "") for name in comparison_header]
        for row in collected
    ]
    root = os.path.join(save_path, "explainability")
    write_csv(
        os.path.join(root, "model_group_mask_comparison.csv"),
        comparison_header,
        comparison_rows,
    )

    top_rows = []
    for fold in folds:
        for model_key in model_keys:
            for axis in ("frequency", "time"):
                subset = [
                    row for row in collected
                    if int(row["fold"]) == fold
                    and row["model_key"] == model_key
                    and row["axis"] == axis
                ]
                for metric_name in rank_metrics:
                    ranked = [
                        row for row in subset
                        if row.get(f"{metric_name}_rank") == 1
                    ]
                    if not ranked:
                        continue
                    row = ranked[0]
                    top_rows.append([
                        fold,
                        model_key,
                        axis,
                        metric_name,
                        row.get("group", ""),
                        row.get("low", ""),
                        row.get("high", ""),
                        row.get("unit", ""),
                        row.get(metric_name, ""),
                    ])
    write_csv(
        os.path.join(root, "top_groups_by_model.csv"),
        [
            "fold", "model_key", "axis", "ranking_metric", "group",
            "low", "high", "unit", "metric_value",
        ],
        top_rows,
    )
    _plot_group_mask_comparisons(root, collected, model_keys, rank_metrics)
    return True


def _plot_group_mask_comparisons(root, rows, model_keys, metric_names):
    """Save large-font cross-model masking figures for presentation use."""
    model_aliases = {
        "rf": "RF",
        "cnntf_v2_gap": "CNN+Tf v2 GAP",
        "alexnet": "AlexNet",
    }
    metric_labels = {
        "r2_drop": "Decrease in R² after masking",
        "rmse_onb_increase": "Increase in ONB RMSE after masking",
        "recall_drop": "Decrease in Recall after masking",
    }

    for axis in ("frequency", "time"):
        axis_rows = [row for row in rows if row.get("axis") == axis]
        group_meta = {}
        for row in axis_rows:
            group_meta[row["group"]] = (
                float(row["low"]),
                float(row["high"]),
                row.get("unit", ""),
            )
        groups = sorted(group_meta, key=lambda name: group_meta[name][0])
        if not groups:
            continue

        group_labels = []
        for group in groups:
            low, high, unit = group_meta[group]
            if axis == "frequency":
                group_labels.append(f"{low:g}–{high:g}\n{unit}")
            else:
                group_labels.append(f"{low:g}–{high:g} {unit}")

        for metric_name in metric_names:
            matrix = np.full((len(model_keys), len(groups)), np.nan, dtype=float)
            for model_index, model_key in enumerate(model_keys):
                for group_index, group in enumerate(groups):
                    values = []
                    for row in axis_rows:
                        if row.get("model_key") != model_key or row.get("group") != group:
                            continue
                        try:
                            value = float(row[metric_name])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if np.isfinite(value):
                            values.append(value)
                    if values:
                        matrix[model_index, group_index] = float(np.mean(values))

            finite = np.abs(matrix[np.isfinite(matrix)])
            if not finite.size:
                continue
            limit = max(float(np.max(finite)), 1e-12)
            cmap = plt.get_cmap("coolwarm").copy()
            cmap.set_bad("#e8e8e8")
            figure_width = max(11.0, 1.35 * len(groups) + 4.0)
            figure_height = max(5.5, 1.15 * len(model_keys) + 2.5)
            with plt.rc_context({"font.family": "Times New Roman"}):
                fig, ax = plt.subplots(figsize=(figure_width, figure_height))
                image = ax.imshow(
                    np.ma.masked_invalid(matrix),
                    aspect="auto",
                    cmap=cmap,
                    vmin=-limit,
                    vmax=limit,
                )
                ax.set_xticks(
                    np.arange(len(groups)), labels=group_labels,
                    fontsize=15, rotation=35, ha="right",
                )
                ax.set_yticks(
                    np.arange(len(model_keys)),
                    labels=[model_aliases.get(key, key) for key in model_keys],
                    fontsize=17,
                )
                ax.set_xlabel(
                    "Frequency band" if axis == "frequency" else "Time interval",
                    fontsize=20,
                )
                ax.set_ylabel("Model", fontsize=20)
                ax.set_title(
                    f"{axis.capitalize()}-group masking: "
                    f"{metric_labels[metric_name]}",
                    fontsize=21,
                    pad=12,
                )
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        value = matrix[row_index, column_index]
                        if not np.isfinite(value):
                            continue
                        color = "white" if abs(value) >= limit * 0.55 else "black"
                        ax.text(
                            column_index,
                            row_index,
                            f"{value:.3g}",
                            ha="center",
                            va="center",
                            fontsize=13,
                            color=color,
                        )
                colorbar = fig.colorbar(image, ax=ax, pad=0.025)
                colorbar.set_label(metric_labels[metric_name], fontsize=17)
                colorbar.ax.tick_params(labelsize=14, direction="in")
                fig.tight_layout()
                fig.savefig(
                    windows_long_path(os.path.join(
                        root,
                        f"group_mask_comparison_{axis}_{metric_name}.png",
                    )),
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
                plt.close(fig)


def explainability_outputs_complete(save_path, config, model_keys, fold_count,
                                    experiment_name=None, max_freq_name=None,
                                    noise_dir_name=None):
    if not config.get("enabled", False):
        return True
    if not explainability_condition_selected(
            config, experiment_name, max_freq_name, noise_dir_name):
        return True

    target_folds = config.get("target_folds")
    folds = [int(v) for v in target_folds] if target_folds else list(range(1, int(fold_count) + 1))
    requested_model_keys = config.get("model_keys") or model_keys
    requested_model_keys = [key for key in requested_model_keys if key in set(model_keys)]
    if not requested_model_keys:
        return True

    for fold in folds:
        for model_key in requested_model_keys:
            out_dir = os.path.join(
                save_path, "explainability", f"fold{fold}", model_key)
            expected_files = ["explainability_summary.csv"]
            methods = _methods_for_model(config, model_key)
            if "group_occlusion" in methods or "occlusion" in methods:
                expected_files.append("group_mask_performance.csv")
            if "tree_shap_pca" in methods or "treeshap" in methods:
                expected_files.append("treeshap_status.csv")
            stability = config.get("stability") or {}
            stability_methods = {
                str(value).lower()
                for value in stability.get("methods", ["integrated_gradients"])
            }
            if (
                model_key != "rf"
                and stability.get("enabled", False)
                and bool(methods & stability_methods)
            ):
                expected_files.append("input_stability.csv")
            sanity = config.get("sanity_check") or {}
            sanity_methods = {
                str(value).lower()
                for value in sanity.get("methods", ["integrated_gradients"])
            }
            if (
                model_key != "rf"
                and sanity.get("enabled", False)
                and bool(methods & sanity_methods)
            ):
                expected_files.append("top_layer_randomization_sanity.csv")
            for filename in expected_files:
                if not _path_exists(os.path.join(out_dir, filename)):
                    return False
    return True


def maybe_explain_trained_model(spec, model, scaler, x_val, y_val, pred, threshold,
                                save_path, fold, max_freq_name, config, pca=None,
                                experiment_name=None, noise_dir_name=None):
    if not config.get("enabled", False):
        return

    if not explainability_condition_selected(
            config, experiment_name, max_freq_name, noise_dir_name):
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
            ["experiment_name", experiment_name],
            ["noise_dir_name", noise_dir_name],
            ["max_freq_name", max_freq_name],
            ["max_freq_hz", max_freq_hz],
            ["methods", "|".join(sorted(_methods_for_model(config, model_key)))],
            ["max_samples_per_fold", config.get("max_samples_per_fold", "")],
            ["ig_steps", config.get("ig_steps", "")],
            ["baseline_value", config.get("baseline_value", 0.0)],
            ["time_groups", config.get("time_groups", "")],
            ["time_extent_seconds", config.get("time_extent_seconds", "")],
            ["frequency_bands_hz", config.get("frequency_bands_hz", "")],
            ["curve_fractions", config.get("curve_fractions", "")],
            ["onb_band_frac", config.get("onb_band_frac", "")],
            ["stability", config.get("stability", "")],
            ["sanity_check", config.get("sanity_check", "")],
            ["save_maps", config.get("save_maps", True)],
            ["plot_axis_order", "x=time_s,y=frequency_khz"],
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
