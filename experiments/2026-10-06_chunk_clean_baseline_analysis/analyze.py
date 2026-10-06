"""Analyze the completed chunk-clean run against the fixed old clean run."""

import csv
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reference_analysis", ROOT / "experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/analyze.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
sys.path.insert(0, str(ROOT / "code"))
from utils.training.internal_validation import internal_splits

KEYS = ("randomforest", "conformer", "alexnet")
PERFORMANCE = "ensemble__performance_kfold"


def save_csv(name, rows):
    assert rows, name
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(ref.long_path(path).read_bytes()).hexdigest()


def endpoint(y, p, threshold):
    q, gap = ref.q100(y, p, threshold)
    failed = y[p < threshold]
    last = float(failed.max()) if len(failed) else float("nan")
    return {
        "q100_kW_m2": q / 1000, "g100_kW_m2": gap / 1000,
        "last_failed_stage_kW_m2": last / 1000,
        "last_stage_negative_chunks": int(((y == last) & (p < threshold)).sum()),
        "suffix_chunks": int((y >= q).sum()) if np.isfinite(q) else 0,
        "suffix_stages": int((np.unique(y) >= q).sum()) if np.isfinite(q) else 0,
    }


def values(rows, oof=False):
    y = np.asarray([float(row["heat_flux" if oof else "y_true"]) for row in rows])
    t = ref.source_thresholds(rows)
    matrix = np.column_stack([[float(row[key]) for row in rows] for key in KEYS])
    return y, t, matrix


def record_metrics(out, ep, tag, unit, noise, rows, predictions, oof=False):
    y, t, _ = values(rows, oof)
    scopes = {"two_day": np.ones(len(y), dtype=bool), **{day: t == threshold for day, threshold in ref.THRESHOLDS.items()}}
    for model, pred in predictions.items():
        for day, mask in scopes.items():
            prefix = {"run": tag, "unit": unit, "noise": noise, "scope": day, "model": model}
            out.append({**prefix, **ref.metrics(y[mask], pred[mask], t[mask])})
            if day != "two_day":
                ep.append({**prefix, **endpoint(y[mask], pred[mask], ref.THRESHOLDS[day])})


def record_failures(stages, failures, feasibility, tag, unit, noise, rows, pred, oof=False):
    y, t, matrix = values(rows, oof)
    post, negative = y >= t, pred < t
    common = matrix.max(axis=1) < t
    for scope in ["two_day", *ref.THRESHOLDS]:
        mask = np.ones(len(y), dtype=bool) if scope == "two_day" else t == ref.THRESHOLDS[scope]
        feasibility.append({"run": tag, "unit": unit, "noise": noise, "scope": scope,
            "post_chunks": int((mask & post).sum()), "ensemble_fn": int((mask & post & negative).sum()),
            "all_three_fn": int((mask & post & common).sum()),
            "ensemble_fn_with_positive_individual": int((mask & post & negative & ~common).sum())})
        if scope == "two_day":
            continue
        failed_y = y[mask & negative]
        last = float(failed_y.max()) if len(failed_y) else float("nan")
        for level in np.unique(y[mask]):
            selected = mask & (y == level)
            stages.append({"run": tag, "unit": unit, "noise": noise, "day": scope,
                "heat_flux_kW_m2": level / 1000, "onb_kW_m2": ref.THRESHOLDS[scope] / 1000,
                "n": int(selected.sum()), "positive": int((selected & ~negative).sum()),
                "negative": int((selected & negative).sum()), "all_three_negative": int((selected & common).sum()),
                "prediction_min_kW_m2": float(pred[selected].min()) / 1000,
                "minimum_margin_kW_m2": float((pred[selected] - t[selected]).min()) / 1000,
                "last_negative_stage": bool(level == last)})
        for i in np.flatnonzero(mask & post & negative):
            failures.append({"run": tag, "unit": unit, "noise": noise, "day": scope,
                "source_wav_id": rows[i]["source_wav_id"], "chunk_index": int(rows[i]["chunk_index"]),
                "heat_flux_kW_m2": y[i] / 1000, "onb_kW_m2": t[i] / 1000,
                "ensemble_margin_kW_m2": (pred[i] - t[i]) / 1000,
                **{key + "_margin_kW_m2": (matrix[i, k] - t[i]) / 1000 for k, key in enumerate(KEYS)},
                "all_three_negative": bool(common[i]), "last_negative_stage": bool(y[i] == last)})


def main():
    scope = ref.read_json(OUTPUT / "run_scope.json")
    metrics, endpoints, stages, failures, feasibility, weight_rows, audit_rows = [], [], [], [], [], [], []
    outer, internal, weights, hashes, fit_ids = {}, {}, {}, {}, {}
    prediction_changes, decomposition, corrections, subset_metrics, subset_endpoints = [], [], [], [], []
    fold_support, bottleneck_bounds, actual_training = [], [], []
    reference_ids = reference_y = reference_training_ids = first_manifest = first_split = None
    for tag, relative in scope["runs"].items():
        run = ROOT / relative
        fits = set()
        hashes[tag] = {}
        for nd, suffix, noise, _ in ref.NOISES:
            path = run / "maxfreq=3kHz" / nd
            manifest = ref.read_json(path / "run_manifest.json")
            completed = ref.read_json(path / "completed.json")
            split = ref.read_json(path / "split_manifest.json")["folds"][0]
            selection = ref.read_json(path / "training_selection_fold1.json")
            rows = sorted(ref.read_rows(path / "fold_pred" / f"pred_f1_{suffix}.csv"), key=ref.identity)
            y, t, matrix = values(rows)
            ids = [ref.identity(row) for row in rows]
            if reference_ids is None:
                reference_ids, reference_y, first_manifest, first_split = ids, y, manifest, split
            assert len(rows) == len(set(ids)) == 540 and ids == reference_ids and np.array_equal(y, reference_y)
            assert np.isfinite(matrix).all()
            assert completed["run_hash"] == manifest["run_hash"] and completed["fit_ids"] == [split["fit_id"]]
            assert split["n_training_chunks"] == 1620 and split["n_evaluation_chunks"] == 540
            assert split["training_wav_groups"] == first_split["training_wav_groups"]
            assert split["evaluation_sample_indices"] == first_split["evaluation_sample_indices"]
            assert manifest["model_params"] == first_manifest["model_params"] and manifest["run_specs"] == first_manifest["run_specs"]
            for section in ("thresholds", "acoustic_selection", "ensemble"):
                assert manifest["validation_config"][section] == first_manifest["validation_config"][section]
            for key in ("epochs", "folds", "random_seed", "color_channel", "smoke_test"):
                assert manifest["validation_config"]["run"][key] == first_manifest["validation_config"]["run"][key]
            assert manifest["learning_context"]["training_noise"] == "clean_only"
            assert manifest["learning_context"]["training_noise_dir"] == "heatflux_no_noise"
            assert not selection["enabled"] and selection["n_after"] == 1620
            fits.add(split["fit_id"])
            wrows = ref.read_rows(path / f"ensemble_weights_{suffix}.csv")
            wrow = next(row for row in wrows if row["strategy_name"] == "performance_kfold")
            w = np.asarray([float(wrow[key]) for key in KEYS])
            assert np.all(w >= 0) and np.isclose(w.sum(), 1)
            if tag in weights:
                np.testing.assert_array_equal(w, weights[tag])
            else:
                weights[tag] = w
            for key, value in zip(KEYS, w):
                weight_rows.append({"run": tag, "noise": noise, "model": key, "weight": value})
            predictions = {key: ref.arrays(rows, key)[1] for key in ref.MODELS}
            np.testing.assert_allclose(predictions[PERFORMANCE], matrix @ w, atol=1e-8, rtol=1e-12)
            np.testing.assert_allclose(predictions["ensemble__simple_equal"], matrix.mean(axis=1), atol=1e-8, rtol=1e-12)
            outer[tag, noise] = rows
            record_metrics(metrics, endpoints, tag, "outer", noise, rows, predictions)
            record_failures(stages, failures, feasibility, tag, "outer", noise, rows, predictions[PERFORMANCE])
            hashes[tag][noise] = {name: digest(path / name) for name in ("run_manifest.json", "completed.json", "internal_validation_fold1.json")}
            hashes[tag][noise]["predictions"] = digest(path / "fold_pred" / f"pred_f1_{suffix}.csv")
            audit_rows.append({"run": tag, "noise": noise, "completed": True, "outer_ids_and_targets_match": True,
                "training_wav_groups_match": True, "models_and_parameters_match": True, "fit_id": split["fit_id"],
                "n_training": 1620, "n_evaluation": 540, "outer_wavs": len(split["evaluation_wav_groups"])})
            if tag == "new_chunk_clean":
                state_ref = ref.read_json(path / "fitted_state_reference_f1.json")
                assert state_ref["fit_id"] == split["fit_id"]
                if noise == "clean":
                    artifact = ref.read_json(Path(state_ref["canonical_directory"]) / "artifact_manifest.json")
                    assert artifact["pca_transform_version"] == 2 and artifact["pca_transform_component_order"] == "C"
                    assert set(artifact["models"]) == set(KEYS)
                    for key, info in artifact["models"].items():
                        assert info["verification"]["performed"] and info["verification"]["passed"]
                        artifact_path = ref.long_path(Path(state_ref["canonical_directory"]) / info["artifact"]["path"])
                        assert artifact_path.stat().st_size == info["artifact"]["bytes"]
                else:
                    assert state_ref["canonical_directory"] == canonical_directory
                canonical_directory = state_ref["canonical_directory"]
        assert len(fits) == 1
        fit_ids[tag] = list(fits)
        oof = ref.read_json(run / "maxfreq=3kHz/heatflux_no_noise/internal_validation_fold1.json")
        os = sorted(oof["samples"], key=ref.identity)
        oof_ids = [ref.identity(row) for row in os]
        assert len(os) == len(set(oof_ids)) == 1620 and not set(oof_ids).intersection(reference_ids)
        if reference_training_ids is None:
            reference_training_ids = oof_ids
        assert oof_ids == reference_training_ids and not oof["test_used"]
        oy, ot, om = values(os, True)
        internal[tag] = os
        op = om @ weights[tag]
        predictions = {**{key: om[:, i] for i, key in enumerate(KEYS)}, PERFORMANCE: op, "ensemble__simple_equal": om.mean(axis=1)}
        record_metrics(metrics, endpoints, tag, "training_oof", "clean", os, predictions, True)
        record_failures(stages, failures, feasibility, tag, "training_oof", "clean", os, op, True)
        for key, error in oof["individual_errors"].items():
            assert np.isclose(1 - ref.metrics(oy, predictions[key], ot)["r2"], error)
        inverse = np.array([1 / max(oof["individual_errors"][key], 1e-6) for key in KEYS])
        np.testing.assert_allclose(weights[tag], inverse / inverse.sum())
        if tag == "new_chunk_clean":
            assert oof["method"] == "chunk_kfold" and oof["shuffle"] and oof["random_state"] == 42
            coverage = np.zeros(len(os), dtype=int)
            source_samples = oof["samples"]
            expected = internal_splits(source_samples, 3, 42, "chunk_kfold")
            for stored, (fit, held) in zip(oof["folds"], expected):
                assert stored["fit_indices"] == fit.tolist() and stored["validation_indices"] == held.tolist()
                coverage[held] += 1
                fit_keys = {ref.identity(source_samples[int(i)]) for i in fit}
                held_keys = {ref.identity(source_samples[int(i)]) for i in held}
                assert not fit_keys.intersection(held_keys) and not (fit_keys | held_keys).intersection(reference_ids)
                assert len({source_samples[int(i)]["source_wav_id"] for i in fit}) == 36
                fold_support.append({"fold": stored["fold"], "fit_chunks": len(fit), "validation_chunks": len(held),
                    "fit_wavs": 36, "shared_chunks": len(fit_keys & held_keys), "outer_test_excluded": True})
            assert np.all(coverage == 1)
            for size in range(1, 4):
                for indexes in combinations(range(3), size):
                    subset = "+".join(KEYS[i] for i in indexes)
                    record_metrics(subset_metrics, subset_endpoints, "new_chunk_clean", "training_oof", "clean", os,
                                   {subset: om[:, indexes].mean(axis=1)}, True)
            for day, threshold in ref.THRESHOLDS.items():
                mask = ot == threshold
                for level in np.unique(oy[mask & (oy >= threshold)]):
                    suffix = mask & (oy >= level)
                    bottleneck_bounds.append({"day": day, "candidate_start_stage_kW_m2": level / 1000,
                        "suffix_chunks": int(suffix.sum()), "common_negative_chunks_in_suffix": int((suffix & (om.max(axis=1) < ot)).sum()),
                        "necessary_condition_for_convex_average_met": bool(np.all(om[suffix].max(axis=1) >= ot[suffix]))})
        tuning = ref.read_rows(run / "tuning_summary.csv")
        for row in tuning:
            if row["model_key"] not in ("conformer", "alexnet"):
                continue
            assert row["epochs_completed"] == "150" and row["stopped_by_memory_error"] == "0"
            assert row["actual_batch_sizes"] == ("12" if row["model_key"] == "conformer" else "8")
            actual_training.append({"run": tag, "noise": row["snr_value"], "model": row["model_key"],
                "epochs_completed": row["epochs_completed"], "actual_batch_sizes": row["actual_batch_sizes"],
                "stopped_by_memory_error": row["stopped_by_memory_error"]})
    for nd, suffix, noise, _ in ref.NOISES:
        old, new = outer["old_wav_clean", noise], outer["new_chunk_clean", noise]
        y, t, old_matrix = values(old)
        _, _, new_matrix = values(new)
        for i, key in enumerate(KEYS):
            difference = new_matrix[:, i] - old_matrix[:, i]
            prediction_changes.append({"noise": noise, "model": key,
                "changed_chunks": int(np.count_nonzero(difference)), "max_abs_delta_W_m2": float(np.abs(difference).max()),
                "mean_abs_delta_W_m2": float(np.abs(difference).mean())})
            if key in ("conformer", "alexnet"):
                assert not np.count_nonzero(difference)
        for prediction_source, matrix in (("old", old_matrix), ("new", new_matrix)):
            for weight_source, weight in (("old", weights["old_wav_clean"]), ("new", weights["new_chunk_clean"])):
                record_metrics(decomposition, [], prediction_source + "_predictions_" + weight_source + "_weights", "outer_descriptive", noise,
                               new, {"performance": matrix @ weight})
        new_p = ref.arrays(new, PERFORMANCE)[1]
        for baseline in (PERFORMANCE, "conformer", "ensemble__simple_equal"):
            rows = old if baseline == PERFORMANCE else new
            baseline_p = ref.arrays(rows, baseline)[1]
            before, after = (baseline_p >= t) != (y >= t), (new_p >= t) != (y >= t)
            corrections.append({"noise": noise, "baseline": "old_performance" if baseline == PERFORMANCE else baseline,
                "errors_corrected": int((before & ~after).sum()), "errors_added": int((~before & after).sum())})
        for indexes in ((1, 2),):
            record_metrics(subset_metrics, subset_endpoints, "new_chunk_clean", "outer_descriptive", noise, new,
                           {"conformer+alexnet": new_matrix[:, indexes].mean(axis=1)})
    for name, rows in (("full_metrics.csv", metrics), ("endpoints.csv", endpoints), ("stage_profiles.csv", stages),
                       ("false_negative_chunks.csv", failures), ("complementarity.csv", feasibility), ("weights.csv", weight_rows),
                       ("run_audit.csv", audit_rows), ("prediction_changes.csv", prediction_changes),
                       ("prediction_weight_decomposition.csv", decomposition), ("corrections.csv", corrections),
                       ("fixed_subset_metrics.csv", subset_metrics), ("fixed_subset_endpoints.csv", subset_endpoints),
                       ("internal_fold_support.csv", fold_support), ("convex_average_necessary_conditions.csv", bottleneck_bounds),
                       ("actual_training.csv", actual_training)):
        save_csv(name, rows)
    aggregate = []
    for tag in scope["runs"]:
        for noise_scope, included in (("all_6_noise", [str(n) for n in (0, -4, -8, -12, -16, -20)]), ("strong_12_16_20", ["-12", "-16", "-20"])):
            selected = [row for row in metrics if row["run"] == tag and row["unit"] == "outer" and row["scope"] == "two_day" and row["model"] == PERFORMANCE and row["noise"] in included]
            aggregate.append({"run": tag, "noise_scope": noise_scope, "mean_rmse_kW_m2": float(np.mean([row["rmse"] for row in selected])) / 1000,
                "mean_onb_rmse_kW_m2": float(np.mean([row["rmse_onb"] for row in selected])) / 1000,
                "mean_fn": float(np.mean([row["fn"] for row in selected])), "mean_fpr": float(np.mean([row["fpr"] for row in selected]))})
    save_csv("noise_aggregates.csv", aggregate)
    # Reuse only the existing training-feature cache. Its old fold and
    # failure labels are discarded; the new OOF labels drive this contrast.
    feature_path = ROOT / "experiments/2026-10-04_training_oof_diversity_diagnosis/training_input_features.csv"
    feature_rows = [row for row in ref.read_rows(feature_path) if row["noise"] == "clean"]
    feature_lookup = {(row["source_wav_id"], int(row["chunk_index"])): row for row in feature_rows}
    assert len(feature_lookup) == 1620
    assert set(feature_lookup) == {(row["source_wav_id"], int(row["chunk_index"])) for row in internal["new_chunk_clean"]}
    for row in internal["new_chunk_clean"]:
        feature = feature_lookup[row["source_wav_id"], int(row["chunk_index"])]
        assert np.isclose(float(feature["heat_flux_kW_m2"]) * 1000, row["heat_flux"])
    feature_contrasts = []
    for day in ref.THRESHOLDS:
        failed = [row for row in failures if row["run"] == "new_chunk_clean" and row["unit"] == "training_oof"
                  and row["day"] == day and row["last_negative_stage"]]
        bad = {(row["source_wav_id"], row["chunk_index"]) for row in failed}
        stage = failed[0]["heat_flux_kW_m2"]
        selected = [row for row in feature_rows if ref.source_day(row) == day and np.isclose(float(row["heat_flux_kW_m2"]), stage)]
        for column in ("log_power_2100_2500", "log_power_total", "temporal_power_cv", "spectral_centroid_hz", "normalized_spectral_entropy"):
            low = [float(row[column]) for row in selected if (row["source_wav_id"], int(row["chunk_index"])) in bad]
            high = [float(row[column]) for row in selected if (row["source_wav_id"], int(row["chunk_index"])) not in bad]
            feature_contrasts.append({"day": day, "last_failed_stage_kW_m2": stage, "feature": column,
                "failed_chunks": len(low), "passed_chunks": len(high),
                "failed_median": float(np.median(low)), "passed_median": float(np.median(high)),
                "comparison": "same-source-WAV training chunks; descriptive, small failed count"})
    save_csv("bottleneck_feature_contrasts.csv", feature_contrasts)
    audit = {"scope": scope, "input_sha256": hashes, "fit_ids": fit_ids, "completed_conditions_per_run": 7,
        "same_540_outer_ids_and_targets": True, "same_1620_training_ids": True, "fixed_parameters_and_no_selection": True,
        "new_internal_fold_support": fold_support, "all_new_final_models_reloaded_without_prediction_difference": True,
        "conformer_and_alexnet_outer_predictions_identical_in_all_7_conditions": True,
        "reused_training_feature_cache_sha256": digest(feature_path),
        "new_training_executed_for_analysis": False, "xai_executed_for_analysis": False}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"performance_outer": [row for row in metrics if row["unit"] == "outer" and row["model"] == PERFORMANCE and row["scope"] == "two_day"],
        "performance_clean_oof": [row for row in metrics if row["unit"] == "training_oof" and row["model"] == PERFORMANCE and row["scope"] == "two_day"],
        "performance_endpoints": [row for row in endpoints if row["model"] == PERFORMANCE], "noise_aggregates": aggregate}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs_verified": 2, "conditions_each": 7, "outer_n": 540, "oof_n": 1620, "new_weights": weights["new_chunk_clean"].tolist(), "new_training": False}))


if __name__ == "__main__":
    main()
