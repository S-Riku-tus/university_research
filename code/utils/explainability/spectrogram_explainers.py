import csv
import os
import warnings
from contextlib import contextmanager, nullcontext

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model


PLOT_FIGSIZE = (9, 8)
PLOT_LABEL_FONTSIZE = 25
PLOT_TICK_FONTSIZE = 20
PLOT_TITLE_FONTSIZE = 20
PLOT_COLORBAR_LABEL_FONTSIZE = 20
PLOT_COLORBAR_TICK_FONTSIZE = 16


class BatchNormalizationEndpointMismatchError(RuntimeError):
    """Signal that the temporary non-fused BN kernel changed predictions."""

    def __init__(self, max_abs_difference):
        self.max_abs_difference = float(max_abs_difference)
        super().__init__(
            "Non-fused BatchNormalization changed IG endpoint predictions; "
            f"maximum absolute difference={self.max_abs_difference:.6g}."
        )


def _canonical_tf_device(device):
    """Return an explicit TensorFlow device or ``None`` for normal placement."""
    if device is None:
        return None
    value = str(device).strip()
    if not value or value.lower() in {"auto", "default", "none"}:
        return None
    if value.lower() == "cpu":
        return "/CPU:0"
    if value.lower() == "gpu":
        return "/GPU:0"
    return value


def _tf_device_scope(device):
    resolved = _canonical_tf_device(device)
    return nullcontext() if resolved is None else tf.device(resolved)


@contextmanager
def _temporary_nonfused_batchnorm(model, enabled=False):
    """Use deterministic BN gradients without changing fitted model state.

    TensorFlow 2.9 does not provide a deterministic GPU gradient for fused
    inference-mode BatchNormalization. The non-fused graph is mathematically
    equivalent in inference mode and uses the same moving statistics and
    learned weights. Only the layer execution flag is changed temporarily;
    it is restored even when attribution raises an exception.
    """
    changed = []
    if enabled:
        layers = getattr(model, "submodules", model.layers)
        for layer in layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                continue
            fused = getattr(layer, "fused", None)
            if fused is True:
                changed.append((layer, fused))
                layer.fused = False
    try:
        yield len(changed)
    finally:
        for layer, fused in changed:
            layer.fused = fused


def windows_long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        os.makedirs(windows_long_path(path), exist_ok=True)


def normalize_map(values, eps=1e-12):
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr)
    arr = arr - np.min(arr)
    denom = np.max(arr) + eps
    return arr / denom


def normalize_magnitude(values, eps=1e-12):
    """Normalize attribution magnitude without shifting signed values."""
    arr = np.abs(np.nan_to_num(np.asarray(values, dtype=np.float32)))
    return arr / (np.max(arr) + eps)


def _time_frequency_display(values):
    """Return a conventional display matrix for an internal (time, frequency) map.

    The model and all attribution calculations retain the repository's
    ``(time_frame, frequency_bin)`` order.  Matplotlib interprets the first
    matrix dimension as vertical, so plotting requires a transpose to show
    time on x and frequency on y without changing the saved numeric array.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"time-frequency values must be 2-D, got shape {arr.shape}."
        )
    return arr.T


def _apply_time_frequency_axes(ax, arr, max_freq_hz, time_extent_seconds):
    if max_freq_hz is None:
        ax.set_xlabel("Time frame", fontsize=PLOT_LABEL_FONTSIZE, labelpad=6)
        ax.set_ylabel("Frequency bin", fontsize=PLOT_LABEL_FONTSIZE, labelpad=6)
    else:
        max_freq_hz = float(max_freq_hz)
        time_extent_seconds = float(time_extent_seconds)
        ax.set_xlabel("Time s", fontsize=PLOT_LABEL_FONTSIZE, labelpad=6)
        ax.set_ylabel("Frequency kHz", fontsize=PLOT_LABEL_FONTSIZE, labelpad=6)
        ax.set_xlim(0.0, time_extent_seconds)
        ax.set_ylim(0.0, max_freq_hz / 1000.0)
        ax.set_xticks([0.0, time_extent_seconds / 2.0, time_extent_seconds])
        ax.set_xticklabels(
            [f"{value:g}" for value in ax.get_xticks()],
            fontsize=PLOT_TICK_FONTSIZE,
        )

        max_freq_khz = max_freq_hz / 1000.0
        frequency_step_khz = 1 if max_freq_khz <= 10 else int(
            np.ceil((max_freq_khz + 1) / 5)
        )
        frequency_ticks = np.arange(
            0.0, max_freq_khz + 1e-9, frequency_step_khz
        )
        ax.set_yticks(frequency_ticks)
        ax.set_yticklabels(
            [f"{value:g}" for value in frequency_ticks],
            fontsize=PLOT_TICK_FONTSIZE,
        )

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=PLOT_TICK_FONTSIZE,
    )


def _save_time_frequency_png(
    values,
    png_path,
    title,
    cmap,
    colorbar_label,
    max_freq_hz=None,
    time_extent_seconds=1.0,
    vmin=None,
    vmax=None,
):
    arr = np.asarray(values, dtype=np.float32)
    shown = _time_frequency_display(arr)
    extent = None
    if max_freq_hz is not None:
        extent = [
            0.0,
            float(time_extent_seconds),
            0.0,
            float(max_freq_hz) / 1000.0,
        ]

    with plt.rc_context({"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        image = ax.imshow(
            shown,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        _apply_time_frequency_axes(
            ax, arr, max_freq_hz, time_extent_seconds
        )
        if title:
            ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE, pad=10)
        colorbar = fig.colorbar(image, ax=ax, pad=0.03)
        colorbar.set_label(
            colorbar_label,
            fontsize=PLOT_COLORBAR_LABEL_FONTSIZE,
            labelpad=10,
        )
        colorbar.ax.tick_params(
            direction="in", labelsize=PLOT_COLORBAR_TICK_FONTSIZE
        )
        fig.tight_layout()
        fig.savefig(
            windows_long_path(png_path),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)


def save_array_and_png(values, out_base, title="", cmap="magma",
                       colorbar_label="normalized importance",
                       max_freq_hz=None, time_extent_seconds=1.0):
    """Save the internal (time, frequency) array and a conventional plot."""
    arr = np.asarray(values, dtype=np.float32)
    ensure_dir(os.path.dirname(out_base))
    np.save(windows_long_path(out_base + ".npy"), arr)

    _save_time_frequency_png(
        arr,
        out_base + ".png",
        title,
        cmap,
        colorbar_label,
        max_freq_hz=max_freq_hz,
        time_extent_seconds=time_extent_seconds,
    )


def save_signed_array_and_png(
    values,
    out_base,
    title="",
    unit="model output",
    max_freq_hz=None,
    time_extent_seconds=1.0,
):
    """Save a signed attribution map with a zero-centred colour scale."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise FloatingPointError("Cannot save a nonfinite signed attribution map.")
    ensure_dir(os.path.dirname(out_base))
    np.save(windows_long_path(out_base + ".npy"), arr)

    limit = float(np.max(np.abs(arr)))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    _save_time_frequency_png(
        arr,
        out_base + ".png",
        title,
        "coolwarm",
        f"signed attribution ({unit})",
        max_freq_hz=max_freq_hz,
        time_extent_seconds=time_extent_seconds,
        vmin=-limit,
        vmax=limit,
    )


def save_input_spectrogram_png(
    sample,
    png_path,
    title="",
    max_freq_hz=None,
    time_extent_seconds=1.0,
):
    """Save the explained input beside its maps using the same physical axes."""
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.mean(arr, axis=-1)
    ensure_dir(os.path.dirname(png_path))
    _save_time_frequency_png(
        arr,
        png_path,
        title,
        "jet",
        "spectrogram power (a.u.)",
        max_freq_hz=max_freq_hz,
        time_extent_seconds=time_extent_seconds,
    )


def write_csv(path, header, rows):
    ensure_dir(os.path.dirname(path))
    with open(windows_long_path(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def last_conv2d_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            nested = last_conv2d_layer_name(layer)
            if nested:
                return nested
    return None


def integrated_gradients(model, sample, baseline=None, steps=64, *,
                         max_steps=4096, batch_size=8, rtol=1e-3,
                         atol=1e-6, map_rtol=1e-2, return_diagnostics=False,
                         device=None, nonfused_batchnorm=False):
    """Return signed input-resolution attributions for a scalar regression model.

    Raw-input straight-line IG with adaptive Gauss-Legendre quadrature.
    For a leading LogPowerCompression layer, concentrate nodes near BOTH
    endpoints using a scalar substitution and its Jacobian. This preserves
    the raw-input path and baseline (not IG in log coordinates).
    Check completeness AND successive signed-map agreement; never rescale
    the answer to force completeness. Tolerances use model-output units.
    """
    x = np.asarray(sample, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"sample must have shape (H, W, C), got {x.shape}")
    if int(steps) < 2 or int(max_steps) < 2 * int(steps) or int(batch_size) <= 0:
        raise ValueError("Require steps >= 2, max_steps >= 2*steps, batch_size > 0.")
    if not all(np.isfinite(v) and v >= 0 for v in (rtol, atol, map_rtol)):
        raise ValueError("IG tolerances must be finite and nonnegative.")
    baseline = (
        np.zeros_like(x, dtype=np.float32)
        if baseline is None
        else np.asarray(baseline, dtype=np.float32)
    )
    if baseline.shape != x.shape:
        raise ValueError(
            f"baseline shape {baseline.shape} does not match sample shape {x.shape}."
        )

    if not np.isfinite(x).all() or not np.isfinite(baseline).all():
        raise ValueError("IG input and baseline must be finite.")
    difference = x.astype(np.float64) - baseline.astype(np.float64)
    endpoint_inputs = np.stack([baseline, x])
    with _tf_device_scope(device):
        reference_endpoints = np.asarray(
            model(endpoint_inputs, training=False)
        )
    if (reference_endpoints.size != 2
            or not np.isfinite(reference_endpoints).all()):
        raise ValueError("IG requires one finite scalar output per sample.")
    delta = (float(reference_endpoints.reshape(-1)[1])
             - float(reference_endpoints.reshape(-1)[0]))
    ratio = 0.0
    layers = [layer for layer in model.layers
              if not isinstance(layer, tf.keras.layers.InputLayer)]
    if (layers and type(layers[0]).__name__ == 'LogPowerCompression'
            and np.all(x >= 0) and np.all(baseline >= 0)):
        ratio = float(np.max(np.abs(difference) /
                     (float(layers[0].scale) + np.minimum(x, baseline).astype(np.float64))))
    history, previous = [], None
    n = int(steps)
    gradient_device = None
    with _temporary_nonfused_batchnorm(
            model, enabled=bool(nonfused_batchnorm)) as changed_bn_count:
        with _tf_device_scope(device):
            attribution_endpoints = np.asarray(
                model(endpoint_inputs, training=False)
            )
        endpoint_kernel_max_abs_difference = float(np.max(np.abs(
            attribution_endpoints.astype(np.float64)
            - reference_endpoints.astype(np.float64)
        )))
        if not np.allclose(
                attribution_endpoints, reference_endpoints,
                rtol=1e-5, atol=1e-6):
            raise BatchNormalizationEndpointMismatchError(
                endpoint_kernel_max_abs_difference
            )
        while True:
            if ratio > 1:
                # Integrate each half separately. Mirroring handles decreasing
                # features/nonzero baselines too, without changing the path.
                nodes, w = np.polynomial.legendre.leggauss(n//2)
                t, w = (nodes+1)/2, w/2
                log_ratio = np.log1p(ratio)
                left = 0.5*np.expm1(t*log_ratio)/ratio
                jacobian = 0.5*log_ratio*np.exp(t*log_ratio)/ratio
                alpha = np.concatenate([left, 1-left])
                weights = np.concatenate([w*jacobian, w*jacobian])
            else:
                nodes, w = np.polynomial.legendre.leggauss(n)
                alpha, weights = (nodes+1)/2, w/2
            integral = np.zeros_like(difference, dtype=np.float64)
            for offset in range(0, len(alpha), int(batch_size)):
                sl = slice(offset, offset + int(batch_size))
                points = (baseline[None, ...]
                          + alpha[sl, None, None, None]*difference[None, ...])
                with _tf_device_scope(device):
                    inputs = tf.convert_to_tensor(points, dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        tape.watch(inputs)
                        target = model(inputs, training=False)
                    grads = tape.gradient(target, inputs)
                if grads is None:
                    raise RuntimeError(
                        "Integrated Gradients could not compute input gradients."
                    )
                if gradient_device is None:
                    gradient_device = str(grads.device)
                values = np.asarray(grads, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError(
                        "Nonfinite IG gradients; attribution is invalid."
                    )
                integral += np.sum(
                    values*weights[sl, None, None, None], axis=0)
            attrs = difference*integral
            error = float(np.sum(attrs, dtype=np.float64)-delta)
            map_change = (float(np.sum(np.abs(attrs-previous)) /
                                max(np.sum(np.abs(attrs)), atol, 1e-30))
                          if previous is not None else None)
            complete = abs(error) <= atol + rtol*abs(delta)
            converged = bool(
                complete and map_change is not None and map_change <= map_rtol
            )
            history.append({
                'nodes': len(alpha),
                'completeness_error': error,
                'map_relative_l1_change': map_change,
            })
            if converged or n >= int(max_steps):
                break
            previous = attrs
            n = min(2*n, int(max_steps))
    diagnostics = {
        'algorithm': 'raw_straight_line_ig_gauss_legendre_v2',
        'sampling': 'symmetric_log_alpha' if ratio > 1 else 'uniform_alpha',
        'alpha_ratio': ratio, 'nodes': len(alpha),
        'gradient_batch_size': int(batch_size),
        'gradient_device_requested': (
            _canonical_tf_device(device) or 'automatic'
        ),
        'gradient_device_actual': gradient_device,
        'nonfused_batchnorm_for_gradients': bool(nonfused_batchnorm),
        'nonfused_batchnorm_layer_count': int(changed_bn_count),
        'endpoint_kernel_max_abs_difference': endpoint_kernel_max_abs_difference,
        'rtol': rtol, 'atol_model_output': atol, 'map_rtol': map_rtol,
        'output_delta_model_units': delta,
        'attribution_sum_model_units': float(np.sum(attrs, dtype=np.float64)),
        'completeness_error_model_units': error,
        'completeness_passed': bool(complete), 'converged': converged,
        'history': history,
    }
    if not converged:
        warnings.warn(f"IG did not converge at {len(alpha)} nodes: completeness error={error:.6g}, "
                      f"map change={map_change}. Do not treat this map as validated.",
                      RuntimeWarning, stacklevel=2)
    result = np.sum(attrs, axis=-1, dtype=np.float64)
    return (result, diagnostics) if return_diagnostics else result


def integrated_gradients_log_power(model, sample, baseline=None, steps=64, *,
                                   max_steps=4096, batch_size=8, rtol=1e-3,
                                   atol=1e-6, map_rtol=1e-2,
                                   return_diagnostics=False, device=None,
                                   nonfused_batchnorm=False):
    """Compute IG on the model's actual log-power feature input.

    The selected neural regressors begin with ``LogPowerCompression``.  A
    straight line from zero in raw-power space passes through its extremely
    steep region near zero and can require thousands of quadrature nodes.
    This variant transforms both endpoints once, then integrates through the
    remaining network along a straight line in log-power space.  The returned
    map still has the input spectrogram resolution, but its path and feature
    meaning are log-power rather than raw-power.
    """
    x = np.asarray(sample, dtype=np.float32)
    baseline = (
        np.zeros_like(x, dtype=np.float32)
        if baseline is None
        else np.asarray(baseline, dtype=np.float32)
    )
    if baseline.shape != x.shape:
        raise ValueError(
            f"baseline shape {baseline.shape} does not match sample shape {x.shape}."
        )
    layers = [
        layer for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    if not layers or type(layers[0]).__name__ != "LogPowerCompression":
        raise ValueError(
            "log-power IG requires LogPowerCompression as the first model layer."
        )
    transform = layers[0]
    with _tf_device_scope(device):
        transformed = np.asarray(
            transform(
                tf.convert_to_tensor(np.stack([baseline, x])),
                training=False,
            )
        )
    if transformed.shape[0] != 2 or not np.isfinite(transformed).all():
        raise ValueError("Log-power IG produced nonfinite transformed endpoints.")
    tail_model = tf.keras.Model(
        inputs=transform.output,
        outputs=model.output,
        name=f"{model.name}_after_{transform.name}",
    )
    result, diagnostics = integrated_gradients(
        tail_model,
        transformed[1],
        baseline=transformed[0],
        steps=steps,
        max_steps=max_steps,
        batch_size=batch_size,
        rtol=rtol,
        atol=atol,
        map_rtol=map_rtol,
        return_diagnostics=True,
        device=device,
        nonfused_batchnorm=nonfused_batchnorm,
    )
    diagnostics.update({
        "algorithm": "log_power_straight_line_ig_gauss_legendre_v1",
        "path_space": "log_power",
        "transform_layer": transform.name,
        "transform_scale": float(transform.scale),
        "raw_baseline_min": float(np.min(baseline)),
        "raw_baseline_max": float(np.max(baseline)),
    })
    return (result, diagnostics) if return_diagnostics else result


def grad_cam_regression(model, sample, conv_layer_name=None):
    """Grad-CAM for a scalar regression output."""
    x = np.asarray(sample, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"sample must have shape (H, W, C), got {x.shape}")
    conv_layer_name = conv_layer_name or last_conv2d_layer_name(model)
    if conv_layer_name is None:
        raise ValueError(f"No Conv2D layer was found in model {model.name}.")

    conv_layer = model.get_layer(conv_layer_name)
    grad_model = Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x[None, ...], training=False)
        target = tf.reshape(predictions, (-1,))[0]
    grads = tape.gradient(target, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = normalize_map(cam)
    cam = tf.image.resize(cam[..., None], x.shape[:2], method="bilinear").numpy()[..., 0]
    return cam


def make_frequency_groups(height, width, max_freq_hz, frequency_bands_hz=None):
    if frequency_bands_hz is None:
        frequency_bands_hz = [
            (0, 256),
            (256, 512),
            (512, 1000),
            (1000, 2000),
            (2000, 5000),
            (5000, 10000),
            (10000, 15000),
            (15000, max_freq_hz),
        ]

    groups = []
    for low, high in frequency_bands_hz:
        low = float(low)
        high = float(high)
        if high <= 0 or low >= max_freq_hz:
            continue
        low = max(0.0, low)
        high = min(float(max_freq_hz), high)
        if high <= low:
            continue
        low_bin = int(round(width * low / max_freq_hz))
        high_bin = int(round(width * high / max_freq_hz))
        low_bin = max(0, min(width, low_bin))
        high_bin = max(low_bin + 1, min(width, high_bin))
        mask = np.zeros((height, width), dtype=bool)
        mask[:, low_bin:high_bin] = True
        groups.append({
            "group": f"freq_{low:g}_{high:g}Hz",
            "axis": "frequency",
            "low": low,
            "high": high,
            "unit": "Hz",
            "low_index": low_bin,
            "high_index": high_bin,
            "mask": mask,
        })

    return groups


def occlusion_importance(
    predict_fn,
    sample,
    groups,
    baseline_value=0.0,
    return_signed_map=False,
):
    x = np.asarray(sample, dtype=np.float32)
    base_pred = float(np.ravel(predict_fn(x[None, ...]))[0])
    rows = []
    score_map = np.zeros(x.shape[:2], dtype=np.float32)
    signed_score_map = np.zeros(x.shape[:2], dtype=np.float32)

    for group in groups:
        masked = np.array(x, copy=True)
        mask = group["mask"]
        masked[mask, :] = baseline_value
        pred = float(np.ravel(predict_fn(masked[None, ...]))[0])
        delta = base_pred - pred
        abs_delta = abs(delta)
        score_map[mask] += abs_delta
        signed_score_map[mask] += delta
        rows.append([
            group["group"],
            group["axis"],
            group["low"],
            group["high"],
            group.get("unit", ""),
            group.get("low_index", ""),
            group.get("high_index", ""),
            base_pred,
            pred,
            delta,
            abs_delta,
        ])
    if return_signed_map:
        return rows, score_map, signed_score_map
    return rows, score_map


def deletion_curve(predict_fn, sample, importance, fractions=None, baseline_value=0.0):
    """Mask the most important pixels first and record prediction changes."""
    if fractions is None:
        fractions = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    x = np.asarray(sample, dtype=np.float32)
    imp = np.asarray(importance, dtype=np.float32)
    if imp.shape != x.shape[:2]:
        raise ValueError(f"importance shape {imp.shape} does not match sample {x.shape[:2]}.")

    flat_order = np.argsort(imp.ravel())[::-1]
    base_pred = float(np.ravel(predict_fn(x[None, ...]))[0])
    rows = []
    n_pixels = imp.size
    for frac in fractions:
        masked = np.array(x, copy=True)
        n_mask = int(round(n_pixels * float(frac)))
        if n_mask > 0:
            selected = flat_order[:n_mask]
            mask = np.zeros(n_pixels, dtype=bool)
            mask[selected] = True
            mask = mask.reshape(imp.shape)
            masked[mask, :] = baseline_value
        pred = float(np.ravel(predict_fn(masked[None, ...]))[0])
        rows.append([frac, n_mask, base_pred, pred, base_pred - pred, abs(base_pred - pred)])
    return rows


def insertion_curve(predict_fn, sample, importance, fractions=None, baseline_value=0.0):
    """Start from a baseline image, insert important pixels first, and record recovery."""
    if fractions is None:
        fractions = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    x = np.asarray(sample, dtype=np.float32)
    imp = np.asarray(importance, dtype=np.float32)
    if imp.shape != x.shape[:2]:
        raise ValueError(f"importance shape {imp.shape} does not match sample {x.shape[:2]}.")

    flat_order = np.argsort(imp.ravel())[::-1]
    original_pred = float(np.ravel(predict_fn(x[None, ...]))[0])
    baseline = np.full_like(x, baseline_value, dtype=np.float32)
    baseline_pred = float(np.ravel(predict_fn(baseline[None, ...]))[0])
    rows = []
    n_pixels = imp.size
    flat_x = x.reshape(n_pixels, x.shape[-1])
    for frac in fractions:
        inserted = np.array(baseline, copy=True)
        n_insert = int(round(n_pixels * float(frac)))
        if n_insert > 0:
            selected = flat_order[:n_insert]
            flat_inserted = inserted.reshape(n_pixels, x.shape[-1])
            flat_inserted[selected, :] = flat_x[selected, :]
        pred = float(np.ravel(predict_fn(inserted[None, ...]))[0])
        delta_from_baseline = pred - baseline_pred
        remaining_delta = original_pred - pred
        rows.append([
            frac,
            n_insert,
            original_pred,
            baseline_pred,
            pred,
            delta_from_baseline,
            abs(delta_from_baseline),
            remaining_delta,
            abs(remaining_delta),
        ])
    return rows


def summarize_map_by_frequency(values, max_freq_hz):
    arr = np.asarray(values, dtype=np.float32)
    freq_profile = arr.mean(axis=0)
    freq_rows = []
    for i, value in enumerate(freq_profile):
        freq_hz = max_freq_hz * (i + 0.5) / len(freq_profile)
        freq_rows.append([i, freq_hz, float(value)])
    return freq_rows
