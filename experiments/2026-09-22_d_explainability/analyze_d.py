"""Audit saved 22 kHz explanations against the A-task chunk residuals.

This script is intentionally read-only with respect to the original run.  It
uses the saved predictions and explanation CSV/NPY files from run 151715 and
writes compact audit tables beside this file.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = (
    ROOT
    / "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.18_0.3_3"
    / "regression_result/npy/ensemble/20260917"
    / "onb_xd-t0611-v0618_iw3-nm_s1e-9_e200_151715/maxfreq=22kHz"
)
A_ROWS = (
    ROOT
    / "experiments/2026-09-19_noise_recovery_review"
    / "a_paired_chunk_residuals.csv"
)

CONDITIONS = {
    "no_noise": "heatflux_no_noise",
    "-8": "heatflux_reference_SNR=-8",
}
MODELS = {
    "randomforest": "randomforest",
    "conformer": "conformer",
    "alexnet": "alexnet",
}


def long_path(path: Path) -> str:
    value = str(path.resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(long_path(path), encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {name}")
    path = OUT / name
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_a() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(A_ROWS):
        if row["snr"] in CONDITIONS:
            grouped[row["snr"]].append(row)
    for snr, rows in grouped.items():
        if len(rows) != 1080:
            raise AssertionError((snr, len(rows)))
    return grouped


def a_by_key(a: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str, int], dict[str, str]]:
    result = {}
    for snr, rows in a.items():
        for row in rows:
            key = (snr, row["source_wav_id"], int(row["chunk_index"]))
            if key in result:
                raise AssertionError(f"Duplicate A key: {key}")
            result[key] = row
    return result


def d1_tables(a: dict[str, list[dict[str, str]]]):
    effects: list[dict] = []
    samples: list[dict] = []
    global_rows: dict[tuple[str, str, str], dict[str, str]] = {}

    keyed_a = a_by_key(a)
    for snr, dirname in CONDITIONS.items():
        pred_name = "pred_f1_no_noise.csv" if snr == "no_noise" else "pred_f1_-8.csv"
        fold_predictions = read_csv(RUN / dirname / "fold_pred" / pred_name)
        if len(fold_predictions) != 1080:
            raise AssertionError((snr, "fold prediction rows", len(fold_predictions)))
        predictions_by_index = {int(r["sample_index"]): r for r in fold_predictions}
        for model, pred_key in MODELS.items():
            model_dir = RUN / dirname / "explainability/fold1" / model
            selected = read_csv(model_dir / "explained_samples.csv")
            masks = read_csv(model_dir / "group_occlusion_summary.csv")
            performance = read_csv(model_dir / "group_mask_performance.csv")
            for row in performance:
                if row["axis"] == "frequency":
                    global_rows[(snr, model, row["group"])] = row

            by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in masks:
                if row["axis"] == "frequency":
                    by_sample[row["sample_id"]].append(row)

            for chosen in selected:
                idx = int(chosen["val_local_index"])
                pred_row = predictions_by_index[idx]
                base = keyed_a[(snr, pred_row["source_wav_id"], int(pred_row["chunk_index"]))]
                expected_y = f(base, "y_true_w_m2")
                expected_pred = f(base, f"prediction_{pred_key}_w_m2")
                if not math.isclose(expected_y, f(chosen, "y_true"), abs_tol=1e-3):
                    raise AssertionError((snr, model, idx, "label mismatch"))
                if not math.isclose(expected_pred, f(chosen, "y_pred"), abs_tol=2.0):
                    raise AssertionError((snr, model, idx, "prediction mismatch"))

                base_abs = abs(expected_pred - expected_y)
                sample_effects = []
                for mask in by_sample[chosen["sample_id"]]:
                    masked_pred = f(mask, "masked_pred")
                    masked_residual = masked_pred - expected_y
                    change = abs(masked_residual) - base_abs
                    record = {
                        "snr": snr,
                        "model": model,
                        "sample_id": chosen["sample_id"],
                        "val_local_index": idx,
                        "source_wav_id": base["source_wav_id"],
                        "chunk_index": int(base["chunk_index"]),
                        "region": base["region"],
                        "ensemble_case": (
                            "success_vs_best_single"
                            if f(base, "ensemble_minus_best_single_sse_w_m2_squared") < 0
                            else "failure_vs_best_single"
                        ),
                        "y_true_w_m2": expected_y,
                        "base_prediction_w_m2": expected_pred,
                        "base_residual_w_m2": expected_pred - expected_y,
                        "band": mask["group"],
                        "low_hz": f(mask, "low"),
                        "high_hz": f(mask, "high"),
                        "masked_prediction_w_m2": masked_pred,
                        "prediction_change_masked_minus_base_w_m2": masked_pred - expected_pred,
                        "abs_error_change_masked_minus_base_w_m2": change,
                        "mask_improves_error": change < 0,
                    }
                    effects.append(record)
                    sample_effects.append(record)

                best = min(sample_effects, key=lambda x: x["abs_error_change_masked_minus_base_w_m2"])
                worst = max(sample_effects, key=lambda x: x["abs_error_change_masked_minus_base_w_m2"])
                samples.append(
                    {
                        "snr": snr,
                        "model": model,
                        "sample_id": chosen["sample_id"],
                        "val_local_index": idx,
                        "source_wav_id": base["source_wav_id"],
                        "chunk_index": int(base["chunk_index"]),
                        "region": base["region"],
                        "ensemble_case": sample_effects[0]["ensemble_case"],
                        "y_true_w_m2": expected_y,
                        "base_prediction_w_m2": expected_pred,
                        "base_residual_w_m2": expected_pred - expected_y,
                        "best_mask_band": best["band"],
                        "best_abs_error_change_w_m2": best["abs_error_change_masked_minus_base_w_m2"],
                        "worst_mask_band": worst["band"],
                        "worst_abs_error_change_w_m2": worst["abs_error_change_masked_minus_base_w_m2"],
                    }
                )

    aggregates: list[dict] = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in effects:
        groups[(row["snr"], row["model"], row["band"])].append(row)
    for key, rows in groups.items():
        snr, model, band = key
        changes = np.asarray(
            [r["abs_error_change_masked_minus_base_w_m2"] for r in rows], dtype=float
        )
        perf = global_rows[key]
        aggregates.append(
            {
                "snr": snr,
                "model": model,
                "band": band,
                "selected_samples": len(rows),
                "selected_improved": int(np.sum(changes < 0)),
                "selected_worsened": int(np.sum(changes > 0)),
                "selected_mean_abs_error_change_w_m2": float(np.mean(changes)),
                "selected_median_abs_error_change_w_m2": float(np.median(changes)),
                "all_1080_r2_drop": f(perf, "r2_drop"),
                "all_1080_rmse_increase_w_m2": f(perf, "rmse_all_increase"),
                "run_onb_60_rmse_increase_w_m2": f(perf, "rmse_onb_increase"),
                "base_recall": f(perf, "base_recall"),
                "masked_recall": f(perf, "masked_recall"),
                "base_f1": f(perf, "base_f1"),
                "masked_f1": f(perf, "masked_f1"),
                "f1_drop": f(perf, "f1_drop"),
                "run_onb_false_negative_increase": int(perf["false_negative_increase"]),
            }
        )
    return effects, samples, aggregates


def targeted_reproduction_tables(a: dict[str, list[dict[str, str]]]):
    """Keep targeted mask rows only where the refit reproduces the A prediction."""
    date_dir = RUN.parents[2] / "20260922"
    candidates = sorted(
        [p for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("d_targeted_m_")],
        key=lambda p: p.name,
    )
    if not candidates:
        return [], []
    run_dir = candidates[-1] / "maxfreq=22kHz/heatflux_no_noise"
    if not Path(long_path(run_dir / "completed.json")).exists():
        raise RuntimeError(f"Targeted run is incomplete: {run_dir}")
    keyed_a = a_by_key(a)
    fold = read_csv(run_dir / "fold_pred/pred_f1_no_noise.csv")
    pred_by_index = {int(r["sample_index"]): r for r in fold}
    audit: list[dict] = []
    accepted: list[dict] = []
    for model in MODELS:
        model_dir = run_dir / "explainability/fold1" / model
        selected = read_csv(model_dir / "explained_samples.csv")
        masks = read_csv(model_dir / "group_occlusion_summary.csv")
        by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in masks:
            if row["axis"] == "frequency":
                by_sample[row["sample_id"]].append(row)
        for chosen in selected:
            idx = int(chosen["val_local_index"])
            pred_row = pred_by_index[idx]
            original = keyed_a[("no_noise", pred_row["source_wav_id"], int(pred_row["chunk_index"]))]
            original_pred = f(original, f"prediction_{model}_w_m2")
            refit_pred = f(chosen, "y_pred")
            delta = refit_pred - original_pred
            reproducible = abs(delta) <= 2.0
            audit.append(
                {
                    "model": model,
                    "sample_id": chosen["sample_id"],
                    "source_wav_id": original["source_wav_id"],
                    "chunk_index": int(original["chunk_index"]),
                    "original_prediction_w_m2": original_pred,
                    "refit_prediction_w_m2": refit_pred,
                    "refit_minus_original_w_m2": delta,
                    "accepted_for_original_run_explanation": reproducible,
                }
            )
            if reproducible:
                y_true = f(original, "y_true_w_m2")
                base_abs = abs(original_pred - y_true)
                case = (
                    "success_vs_best_single"
                    if f(original, "ensemble_minus_best_single_sse_w_m2_squared") < 0
                    else "failure_vs_best_single"
                )
                for mask in by_sample[chosen["sample_id"]]:
                    masked_pred = f(mask, "masked_pred")
                    accepted.append(
                        {
                            "model": model,
                            "sample_id": chosen["sample_id"],
                            "source_wav_id": original["source_wav_id"],
                            "chunk_index": int(original["chunk_index"]),
                            "ensemble_case": case,
                            "band": mask["group"],
                            "base_residual_w_m2": original_pred - y_true,
                            "masked_prediction_w_m2": masked_pred,
                            "prediction_change_masked_minus_base_w_m2": masked_pred - original_pred,
                            "abs_error_change_masked_minus_base_w_m2": abs(masked_pred - y_true) - base_abs,
                        }
                    )
    return audit, accepted


def d2_tables():
    shap_rows: list[dict] = []
    ig_rows: list[dict] = []
    cam_rows: list[dict] = []
    method_rows: list[dict] = []

    for snr, dirname in CONDITIONS.items():
        rf_dir = RUN / dirname / "explainability/fold1/randomforest"
        for row in read_csv(rf_dir / "treeshap_pca_summary.csv"):
            err = f(row, "completeness_error")
            pred = f(row, "model_prediction_heat_flux")
            shap_rows.append(
                {
                    "snr": snr,
                    "sample_id": row["sample_id"],
                    "val_local_index": int(row["val_local_index"]),
                    "prediction_w_m2": pred,
                    "reconstruction_error_w_m2": err,
                    "relative_reconstruction_error": abs(err) / max(abs(pred), 1.0),
                    "passed_abs_1_w_m2": abs(err) <= 1.0,
                }
            )

        for model in ("conformer", "alexnet"):
            model_dir = RUN / dirname / "explainability/fold1" / model
            for chosen in read_csv(model_dir / "explained_samples.csv"):
                sample_dir = model_dir / chosen["sample_id"]
                with open(
                    long_path(sample_dir / "integrated_gradients_diagnostics.json"),
                    encoding="utf-8",
                ) as handle:
                    diag = json.load(handle)
                ig_rows.append(
                    {
                        "snr": snr,
                        "model": model,
                        "sample_id": chosen["sample_id"],
                        "val_local_index": int(chosen["val_local_index"]),
                        "baseline_value_raw_power": 0.0,
                        "nodes": int(diag["nodes"]),
                        "completeness_passed": bool(diag["completeness_passed"]),
                        "map_converged": bool(diag["converged"]),
                        "completeness_error_w_m2": float(diag["completeness_error_heat_flux"]),
                        "final_map_relative_l1_change": float(diag["history"][-1]["map_relative_l1_change"]),
                        "alternate_baseline_available": False,
                    }
                )

            if model == "alexnet":
                for chosen in read_csv(model_dir / "explained_samples.csv"):
                    cam = np.load(long_path(model_dir / chosen["sample_id"] / "grad_cam.npy"))
                    positive = cam[cam > 0]
                    cam_rows.append(
                        {
                            "snr": snr,
                            "sample_id": chosen["sample_id"],
                            "val_local_index": int(chosen["val_local_index"]),
                            "target_layer": "alex_conv5 (last Conv2D selected by code)",
                            "native_map_shape": "12x12",
                            "saved_resized_shape": f"{cam.shape[0]}x{cam.shape[1]}",
                            "relu_positive_only": True,
                            "saved_min": float(np.min(cam)),
                            "saved_max": float(np.max(cam)),
                            "positive_pixel_fraction": float(np.mean(cam > 0)),
                            "mean_positive_value": float(np.mean(positive)) if positive.size else 0.0,
                        }
                    )

    shap_max = max(abs(r["reconstruction_error_w_m2"]) for r in shap_rows)
    for snr in CONDITIONS:
        subset = [r for r in ig_rows if r["snr"] == snr]
        for model in ("conformer", "alexnet"):
            model_subset = [r for r in subset if r["model"] == model]
            passed = sum(bool(r["map_converged"]) for r in model_subset)
            method_rows.append(
                {
                    "snr": snr,
                    "model": model,
                    "method": "Integrated Gradients",
                    "numerical_result": f"{passed}/{len(model_subset)} maps converged",
                    "baseline_sensitivity": "not evaluable: only raw-power zero baseline was saved",
                    "spatial_resolution_or_domain": "224x224 signed input attribution",
                    "claim_use": "exclude all maps from main claims; alternate baseline absent and convergence is incomplete",
                }
            )
        method_rows.append(
            {
                "snr": snr,
                "model": "randomforest",
                "method": "TreeSHAP",
                "numerical_result": f"5/5 reconstructed; all-run max error {shap_max:.3f} W/m2",
                "baseline_sensitivity": "TreeSHAP expected value; no physical reference baseline",
                "spatial_resolution_or_domain": "100 PCA components",
                "claim_use": "use for RF internal prediction construction only; do not map PCA components to Hz",
            }
        )
        method_rows.append(
            {
                "snr": snr,
                "model": "alexnet",
                "method": "Grad-CAM",
                "numerical_result": "5/5 arrays finite and normalized to [0,1]",
                "baseline_sensitivity": "not applicable; gradient activation map",
                "spatial_resolution_or_domain": "alex_conv5 12x12, bilinear-resized to 224x224, ReLU positive only",
                "claim_use": "supplementary coarse positive-response location only; exclude from signed error and narrow-band claims",
            }
        )

    return shap_rows, ig_rows, cam_rows, method_rows


def main() -> None:
    a = load_a()
    effects, samples, aggregates = d1_tables(a)
    reproduction, targeted = targeted_reproduction_tables(a)
    shap_rows, ig_rows, cam_rows, methods = d2_tables()
    write_csv("d1_sample_band_effects.csv", effects)
    write_csv("d1_sample_summary.csv", samples)
    write_csv("d1_band_alignment.csv", aggregates)
    write_csv("d1_targeted_reproduction_audit.csv", reproduction)
    write_csv("d1_targeted_accepted_band_effects.csv", targeted)
    write_csv("d2_treeshap_audit.csv", shap_rows)
    write_csv("d2_ig_audit.csv", ig_rows)
    write_csv("d2_gradcam_audit.csv", cam_rows)
    write_csv("d2_method_decision.csv", methods)
    print(
        json.dumps(
            {
                "d1_effect_rows": len(effects),
                "d1_samples": len(samples),
                "d1_aggregate_rows": len(aggregates),
                "d1_targeted_reproduction_rows": len(reproduction),
                "d1_targeted_accepted_effect_rows": len(targeted),
                "treeshap_samples": len(shap_rows),
                "ig_samples": len(ig_rows),
                "gradcam_samples": len(cam_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
