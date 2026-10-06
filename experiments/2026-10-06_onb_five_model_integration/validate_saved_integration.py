"""Validate production integration and summary pipelines on saved paired data."""
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_ensemble_regression_onb as onb
from run_hgb_complementarity_validation import read_json, read_csv, save_json, save_csv
from utils.calculation.source_day_metrics import source_day_metric_rows, metric_delta_rows
from utils.dataloading.acoustic_summary_features import AcousticFrequency34
from utils.ensemble.fixed_core_stacking import mse_simplex
from utils.models.regression.base_regression import RegressionModelMaker
from utils.training.fitted_artifacts import begin_fitted_state, save_and_verify_model, finish_fitted_state
from utils.training.model_training import ModelTrainer

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/2026-10-06_fixed_three_additions"
PAIRED = ROOT / "experiments/2026-10-06_hgb_followup_validation"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
MATCH = {"original3_mse": "existing3_clean_mse", "original3_performance": "existing_performance",
         "original3_hgb": "existing3_clean_mse__hgb4_anchor",
         "original3_extra_trees": "existing3_clean_mse__extra_trees4_anchor",
         "original3_hgb_extra_trees": "existing3_clean_mse__selected5"}


def audit_from_saved(manifest, oof, train_indices):
    positions = {int(i): j for j, i in enumerate(train_indices)}
    return {"test_used": False, "method": "chunk_kfold", "folds": [
        {"fold": split["fold"], "fit_indices": [positions[i] for i in split["fit_indices"]],
         "validation_indices": [positions[i] for i in split["held_indices"]]}
        for split in manifest["folds"]],
        "samples": [{"heat_flux": float(manifest["samples"][int(idx)]["sample_filename"].split("_")[0]),
            "source_wav_id": manifest["samples"][int(idx)]["source_wav_id"],
            **{key: float(oof[j, k]*1000) for k, key in enumerate(onb.MODEL_KEYS)}} for j, idx in enumerate(train_indices)]}


def removal_weights(matrix, y, removed):
    core_idx = [j for j in range(3) if onb.MODEL_KEYS[j] != removed]
    added_idx = [j for j in [3, 4] if onb.MODEL_KEYS[j] != removed]
    core = mse_simplex(matrix[:, core_idx], y, [1e-6]*len(core_idx))
    blocks = np.column_stack([matrix[:, core_idx]@core, *[matrix[:, j] for j in added_idx]])
    mix = mse_simplex(blocks, y, [.25]+[.05]*len(added_idx))
    weights = np.zeros(5)
    weights[core_idx] = core*mix[0]
    weights[added_idx] = mix[1:]
    return weights


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    onb.validate_validation_config(onb.MODEL_SPECS)
    rows, changes, ablation, probes, weight_rows = [], [], [], [], []
    maximum_prediction_difference = 0.
    maximum_component_difference = 0.
    metric_lookup = {(int(r["seed"]), r["method"], r["noise"], r["day"]): r for r in read_csv(SOURCE / "metrics.csv")}
    plan = onb.parameter_plan_for_max_freq("maxfreq=3kHz")["parameter_sets"][0]
    specs = onb.resolve_parameter_set(onb.MODEL_SPECS, plan)
    for seed in [43, 44]:
        manifest = read_json(PAIRED / f"seed{seed}/manifest.json")
        with np.load(SOURCE / f"seed{seed}/predictions.npz") as saved:
            selected = [saved["keys"].tolist().index(k) for k in onb.MODEL_KEYS]
            oof, outer = saved["oof"][:, :, selected].copy(), saved["outer"][:, :, selected].copy()
            train, test = saved["train_indices"].copy(), saved["test_indices"].copy()
        metadata = [manifest["samples"][int(i)] for i in test]
        y = np.asarray([float(r["sample_filename"].split("_")[0]) for r in metadata])
        audit = audit_from_saved(manifest, oof[0], train)
        run = onb.ENSEMBLE_MANAGER.create_run(specs)
        fit = run.fit_weights_from_internal_audit(audit, 1)
        (OUT / f"seed{seed}").mkdir(exist_ok=True)
        run.save_crossfit_fit(OUT / f"seed{seed}", 1, fit)
        with np.load(SOURCE / f"seed{seed}/outer_integrated.npz") as saved:
            prior_predictions, prior_names = saved["predictions"].copy(), saved["methods"].tolist()
        removed = {key: removal_weights(oof[0]*1000, fit["targets"], key) for key in onb.MODEL_KEYS}
        state = OUT / f"seed{seed}/summary_pipeline_probe"
        scaler = joblib.load(PAIRED / f"seed{seed}/final/target_scaler.joblib")
        target_scaler = scaler
        trainer, maker = ModelTrainer(seed), RegressionModelMaker((224, 224, 1))
        artifact = begin_fitted_state(state, f"compatibility-{seed}", 1, seed+1,
            [r["source_wav_id"] for r in metadata], {}, None, target_scaler)
        models = {
            "hgb": joblib.load(PAIRED / f"seed{seed}/final/hgb.joblib"),
            "extra_trees": joblib.load(SOURCE / f"seed{seed}/final/extra_trees.joblib").regressor_,
        }
        pipelines = {key: Pipeline([("features", AcousticFrequency34()), ("regressor", model)]) for key, model in models.items()}
        for n, noise in enumerate(NOISES):
            outputs = run.combine_predictions({key: outer[n, :, j]*1000 for j, key in enumerate(onb.MODEL_KEYS)}, {}, 1, fit)
            predictions = {key: item["prediction"] for key, item in outputs.items()}
            for production, prior in MATCH.items():
                p = predictions[f"ensemble__{production}"]/1000
                expected = prior_predictions[n, :, prior_names.index(prior)]
                difference = float(np.max(abs(p-expected)))
                maximum_prediction_difference = max(maximum_prediction_difference, difference)
                np.testing.assert_allclose(p, expected, rtol=1e-7, atol=1e-4)
            current = source_day_metric_rows(y, predictions, metadata, onb.THRESHOLD_BY_EXPERIMENT, 1, run.labels)
            for result in current:
                key = result["model_key"].replace("ensemble__", "")
                if key not in MATCH:
                    continue
                day = "two_day" if result["source_day"] == "pooled_source_day_thresholds" else result["source_day"][:10].replace(".", "")
                expected = metric_lookup[(seed, MATCH[key], noise, day)]
                for new_key, old_key, scale in [("rmse_all", "rmse", 1000), ("mae_all", "mae", 1000),
                    ("rmse_onb", "rmse_onb", 1000), ("r2", "r2", 1), ("recall", "recall", 1),
                    ("precision", "precision", 1), ("f1", "f1", 1), ("fp", "fp", 1), ("fn", "fn", 1),
                    ("roc_auc_cont", "roc_auc", 1), ("pr_auc_cont", "pr_auc", 1), ("q100", "q100", 1000)]:
                    np.testing.assert_allclose(result[new_key], float(expected[old_key])*scale,
                        rtol=1e-6, atol=.2 if scale==1000 else 1e-6, equal_nan=True)
            rows.extend({"seed": seed, "noise": noise, **r} for r in current)
            changes.extend({"seed": seed, "noise": noise, **r} for r in metric_delta_rows(current))
            for key, weight in removed.items():
                ablated = source_day_metric_rows(y, {f"remove_{key}": outer[n]@weight*1000}, metadata,
                    onb.THRESHOLD_BY_EXPERIMENT, 1)[0]
                original = next(r for r in current if r["source_day"]=="pooled_source_day_thresholds" and r["model_key"]=="ensemble__original3_hgb_extra_trees")
                ablation.append({"seed": seed, "noise": noise, "removed_model": key,
                    "full5_rmse_kW_m2": original["rmse_all"]/1000, "ablated_rmse_kW_m2": ablated["rmse_all"]/1000,
                    "delta_rmse_removal_kW_m2": (ablated["rmse_all"]-original["rmse_all"])/1000,
                    "full5_fn": original["fn"], "ablated_fn": ablated["fn"], "full5_fp": original["fp"], "ablated_fp": ablated["fp"],
                    "weights_fit": "clean OOF only; retained core reweighted; descriptive developmental ablation"})
            raw = np.asarray(np.load(PAIRED / f"cache/raw_{noise}.npy", mmap_mode="r")[test])
            for key, pipeline in pipelines.items():
                spec = next(s for s in specs if s["key"] == key)
                p = trainer.predict_one_model(spec, pipeline, raw, None, target_scaler)
                difference = float(np.max(abs(p-outer[n, :, onb.MODEL_KEYS.index(key)]*1000)))
                maximum_component_difference = max(maximum_component_difference, difference)
                np.testing.assert_allclose(p, outer[n, :, onb.MODEL_KEYS.index(key)]*1000, rtol=1e-10, atol=1e-6)
                probes.append({"seed": seed, "noise": noise, "model": key, "samples": len(test), "max_abs_difference_W_m2": difference})
                if noise == "clean":
                    save_and_verify_model(state, artifact, spec, pipeline, maker, trainer, raw[:8], target_scaler, None)
            if noise == "clean":
                # This directory verifies only the two summary components;
                # it is not a deployable complete five-model fitted state.
                finish_fitted_state(state, artifact, {})
            del raw
        for name, weights in fit["weights"].items():
            weight_rows.extend({"seed": seed, "strategy": name, "model": key, "weight": value} for key, value in weights.items())
    save_csv(OUT / "metrics_recomputed.csv", rows)
    save_csv(OUT / "metric_changes.csv", changes)
    save_csv(OUT / "model_removal_ablation.csv", ablation)
    save_csv(OUT / "summary_pipeline_compatibility.csv", probes)
    save_csv(OUT / "production_weights.csv", weight_rows)
    save_json(OUT / "saved_integration_verification.json", {"status": "passed", "seeds": [43, 44],
        "production_strategies_verified": list(MATCH), "noise_conditions": NOISES,
        "maximum_prediction_difference_kW_m2": maximum_prediction_difference,
        "maximum_component_difference_W_m2": maximum_component_difference,
        "component_comparisons": len(probes), "component_predictions_checked": sum(p["samples"] for p in probes),
        "source_metric_rows": len(rows), "metric_changes": len(changes), "removal_ablation_conditions": len(ablation),
        "original_predictions_changed": False, "new_150_epoch_main_run_executed": False,
        "core_member_floor": 1e-6, "note": "Numerical membership floor is the only added guard versus the prior pilot; these paired cores were already positive"})
    print(f"[saved validation] {len(rows)} metrics; max integration difference {maximum_prediction_difference:.9g} kW/m2; {len(probes)} pipeline conditions")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
