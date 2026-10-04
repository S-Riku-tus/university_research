"""Paired analysis of the adopted clean-only and matched 3 kHz saved runs."""

import importlib.util
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "reference_analysis", ROOT / "experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/analyze.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
RUNS = {
    "clean_only": ref.RUNS["3kHz"],
    "matched": ref.RESULT_ROOT / "202610/03/onb_wc-t0611-v0611_iw3-nm_c1s_s0_e150_153735",
}
REGIONS = ("below_60kW", "60kW_to_ONB", "ONB_to_1.5ONB", "at_least_1.5ONB")


def save(name, rows):
    # Reuse the existing CSV helper without touching its historical output files.
    original = ref.OUTPUT
    try:
        ref.OUTPUT = OUTPUT
        ref.write_rows(name, rows)
    finally:
        ref.OUTPUT = original


def region_masks(y, t):
    return {
        "below_60kW": y < 60000,
        "60kW_to_ONB": (y >= 60000) & (y < t),
        "ONB_to_1.5ONB": (y >= t) & (y < 1.5 * t),
        "at_least_1.5ONB": y >= 1.5 * t,
    }


def region_metrics(y, prediction, thresholds):
    # A pre-ONB/post-ONB region contains one class by definition. Its AUCs
    # are not interpretable; suppress only these expected metric warnings.
    if np.unique(y >= thresholds).size > 1:
        return ref.metrics(y, prediction, thresholds)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only one class is present in y_true.*")
        warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
        result = ref.metrics(y, prediction, thresholds)
    result["roc_auc"] = result["pr_auc"] = float("nan")
    return result


def bootstrap(rows, y, first, second, rng, repeats=5000):
    clusters, indices = ref.cluster_indices(rows)
    draws = rng.integers(0, len(clusters), size=(repeats, len(clusters)))
    first_sse = np.asarray([np.sum((first - y)[indices[c]] ** 2) for c in clusters])
    second_sse = np.asarray([np.sum((second - y)[indices[c]] ** 2) for c in clusters])
    sizes = np.asarray([len(indices[c]) for c in clusters])
    sampled_n = sizes[draws].sum(axis=1)
    differences = (np.sqrt(first_sse[draws].sum(axis=1) / sampled_n)
                   - np.sqrt(second_sse[draws].sum(axis=1) / sampled_n)) / 1000
    return {
        "delta_rmse_kW_m2": (ref.rmse(y, first) - ref.rmse(y, second)) / 1000,
        "ci95_low_kW_m2": float(np.percentile(differences, 2.5)),
        "ci95_high_kW_m2": float(np.percentile(differences, 97.5)),
        "n_wav_clusters": len(clusters), "repeats": repeats,
    }


def main():
    audit_rows, metric_rows, regions, stages, q100s, weights = [], [], [], [], [], []
    feasibility, residual_pairs, oof_rows, corrections, oof_feasibility = [], [], [], [], []
    loaded, seen_fits = {}, {policy: set() for policy in RUNS}
    clean_identity, clean_y, reference_manifest, reference_training_ids = None, None, None, None
    for policy, run in RUNS.items():
        for nd, suffix, noise, order in ref.NOISES:
            path = run / "maxfreq=3kHz" / nd
            completed = ref.read_json(path / "completed.json")
            manifest = ref.read_json(path / "run_manifest.json")
            split = ref.read_json(path / "split_manifest.json")["folds"][0]
            selection = ref.read_json(path / "training_selection_fold1.json")
            rows = ref.read_rows(path / "fold_pred" / f"pred_f1_{suffix}.csv")
            rows = sorted(rows, key=ref.identity)
            loaded[(policy, noise)] = rows
            ids = [ref.identity(row) for row in rows]
            y, p = ref.arrays(rows, "ensemble__performance_kfold")
            t = ref.source_thresholds(rows)
            if clean_identity is None:
                clean_identity, clean_y, reference_manifest = ids, y, manifest
            assert len(rows) == len(set(ids)) == 540 and ids == clean_identity
            assert np.array_equal(y, clean_y)
            assert split["n_training_chunks"] == 1620 and split["n_evaluation_chunks"] == 540
            assert len(split["training_wav_groups"]) == len(split["evaluation_wav_groups"]) == 36
            assert completed["run_hash"] == manifest["run_hash"]
            assert completed["fit_ids"] == [split["fit_id"]]
            assert not selection["enabled"] and selection["n_after"] == 1620
            assert manifest["model_params"] == reference_manifest["model_params"]
            assert manifest["run_specs"] == reference_manifest["run_specs"]
            for section in ("run", "features", "ensemble", "thresholds", "acoustic_selection"):
                assert manifest["validation_config"][section] == reference_manifest["validation_config"][section]
            assert manifest["learning_context"]["training_noise"] == policy
            assert manifest["learning_context"]["evaluation_mode"] == "within_wav_chunk"
            assert manifest["learning_context"]["training_noise_dir"] == (
                "heatflux_no_noise" if policy == "clean_only" else nd)
            seen_fits[policy].add(split["fit_id"])
            audit_rows.append({"policy": policy, "noise": noise, "completed": True,
                               "n": len(rows), "same_ids_and_targets": True,
                               "n_training": 1620, "fit_id": split["fit_id"],
                               "parameters_and_method_match": True})
            for w in ref.read_rows(path / f"ensemble_weights_{suffix}.csv"):
                weights.append({"policy": policy, "noise": noise, **w})

            scopes = {"two_day": rows, **{day: [r for r in rows if ref.source_day(r) == day]
                                           for day in ref.THRESHOLDS}}
            for scope, selected in scopes.items():
                ys, ps = ref.arrays(selected, "ensemble__performance_kfold")
                ts = ref.source_thresholds(selected)
                total_sse = np.sum((ps - ys) ** 2)
                for key, model in ref.MODELS.items():
                    _, pred = ref.arrays(selected, key)
                    metric_rows.append({"policy": policy, "noise": noise, "scope": scope,
                                        "model": model, **ref.metrics(ys, pred, ts)})
                    if scope != "two_day":
                        q, gap = ref.q100(ys, pred, ref.THRESHOLDS[scope])
                        q100s.append({"policy": policy, "noise": noise, "day": scope,
                                      "model": model, "q100": q, "g100": gap})
                model_predictions = np.column_stack([ref.arrays(selected, key)[1]
                    for key in ("randomforest", "conformer", "alexnet")])
                for region, mask in region_masks(ys, ts).items():
                    vals = region_metrics(ys[mask], ps[mask], ts[mask])
                    regions.append({"policy": policy, "noise": noise, "scope": scope,
                                    "region": region, "sse_share_pct": np.sum((ps[mask]-ys[mask])**2)/total_sse*100,
                                    **vals})
                    preds = model_predictions[mask]
                    actual_y, thresholds = ys[mask], ts[mask]
                    lo, hi = preds.min(axis=1), preds.max(axis=1)
                    fn = (actual_y >= thresholds) & (ps[mask] < thresholds)
                    feasibility.append({"policy": policy, "noise": noise, "scope": scope,
                        "region": region, "n": int(mask.sum()),
                        "all_over": int((lo > actual_y).sum()), "all_under": int((hi < actual_y).sum()),
                        "inside_range": int(((lo <= actual_y) & (actual_y <= hi)).sum()),
                        "fn": int(fn.sum()), "fn_some_model_positive": int((fn & (hi >= thresholds)).sum()),
                        "fn_all_models_negative": int((fn & (hi < thresholds)).sum())})
                for heat in sorted(set(ys)):
                    mask = ys == heat
                    if scope == "two_day":
                        continue
                    stages.append({"policy": policy, "noise": noise, "day": scope,
                        "heatflux": heat, "n": int(mask.sum()), "onb": ref.THRESHOLDS[scope],
                        "prediction_mean": float(np.mean(ps[mask])),
                        "prediction_std": float(np.std(ps[mask])),
                        "prediction_min": float(np.min(ps[mask])), "prediction_max": float(np.max(ps[mask])),
                        "rmse": ref.rmse(ys, ps, mask), "bias": float(np.mean((ps-ys)[mask])),
                        "positive_count": int((ps[mask] >= ts[mask]).sum())})
            for baseline in ("conformer", "ensemble__simple_equal"):
                _, bp = ref.arrays(rows, baseline)
                truth = y >= t
                wrong_baseline, wrong_perf = (bp >= t) != truth, (p >= t) != truth
                corrections.append({"policy": policy, "noise": noise, "baseline": ref.MODELS[baseline],
                    "errors_corrected": int((wrong_baseline & ~wrong_perf).sum()),
                    "errors_added": int((~wrong_baseline & wrong_perf).sum()),
                    "perf_lower_squared_error_chunks": int(((p-y)**2 < (bp-y)**2).sum())})
            for region, mask in {"all": np.ones(len(y), dtype=bool), **region_masks(y, t)}.items():
                for a, b in combinations(("randomforest", "conformer", "alexnet"), 2):
                    ea = ref.arrays(rows, a)[1][mask] - y[mask]
                    eb = ref.arrays(rows, b)[1][mask] - y[mask]
                    residual_pairs.append({"policy": policy, "noise": noise, "region": region,
                        "model_a": ref.MODELS[a], "model_b": ref.MODELS[b], "n": int(mask.sum()),
                        "bias_a": float(np.mean(ea)), "bias_b": float(np.mean(eb)),
                        "correlation": float(np.corrcoef(ea, eb)[0, 1]),
                        "mean_error_product": float(np.mean(ea*eb)),
                        "opposite_sign_count": int((ea*eb < 0).sum())})
            oof = ref.read_json(path / "internal_validation_fold1.json")
            assert not oof["test_used"] and all(f["shared_source_wavs"] == 0 for f in oof["folds"])
            samples = oof["samples"]
            ids_train = {(ref.source_day(s), s["source_wav_id"], int(s["chunk_index"])) for s in samples}
            assert len(samples) == len(ids_train) == 1620 and not ids_train.intersection(ids)
            if reference_training_ids is None:
                reference_training_ids = ids_train
            assert ids_train == reference_training_ids
            oy = np.asarray([s["heat_flux"] for s in samples])
            ot = ref.source_thresholds(samples)
            for key, model in ref.MODELS.items():
                if key.startswith("ensemble"):
                    continue
                op = np.asarray([s[key] for s in samples])
                oof_rows.append({"policy": policy, "noise": noise, "model": model,
                                 **ref.metrics(oy, op, ot)})
            op_matrix = np.column_stack([np.asarray([s[key] for s in samples])
                for key in ("randomforest", "conformer", "alexnet")])
            for region, mask in region_masks(oy, ot).items():
                lo, hi = op_matrix[mask].min(axis=1), op_matrix[mask].max(axis=1)
                oof_feasibility.append({"policy": policy, "noise": noise, "region": region,
                    "n": int(mask.sum()), "all_over": int((lo > oy[mask]).sum()),
                    "all_under": int((hi < oy[mask]).sum()),
                    "inside_range": int(((lo <= oy[mask]) & (oy[mask] <= hi)).sum()),
                    "all_models_false_negative": int(((oy[mask] >= ot[mask]) & (hi < ot[mask])).sum())})
    assert len(seen_fits["clean_only"]) == 1 and len(seen_fits["matched"]) == 7
    clean_differences = []
    for key, model in ref.MODELS.items():
        diff = float(np.max(np.abs(ref.arrays(loaded[("clean_only", "clean")], key)[1]
                                  - ref.arrays(loaded[("matched", "clean")], key)[1])))
        assert diff == 0.0
        clean_differences.append({"model": model, "max_clean_prediction_difference": diff})
    save("completion_audit.csv", audit_rows)
    save("clean_identity.csv", clean_differences)
    save("metrics_source_day_thresholds.csv", metric_rows)
    save("region_metrics.csv", regions)
    save("stage_profiles.csv", stages)
    save("q100_source_day_thresholds.csv", q100s)
    save("weights.csv", weights)
    save("complementarity.csv", feasibility)
    save("residual_pairs.csv", residual_pairs)
    save("ensemble_corrections.csv", corrections)
    save("training_oof_metrics.csv", oof_rows)
    save("training_oof_complementarity.csv", oof_feasibility)

    aggregates = []
    for policy in RUNS:
        for model in ref.MODELS.values():
            for name, noises in (("clean", ["clean"]), ("noise_mean", ["0", "-4", "-8", "-12", "-16", "-20"]),
                                 ("strong_mean", ["-12", "-16", "-20"]), ("minus20", ["-20"])):
                selected = [r for r in metric_rows if r["policy"] == policy and r["scope"] == "two_day"
                            and r["model"] == model and r["noise"] in noises]
                aggregates.append({"policy": policy, "model": model, "aggregation": name,
                    **{key: float(np.mean([r[key] for r in selected]))
                       for key in ("rmse", "rmse_pre", "rmse_post", "rmse_onb", "fpr", "recall", "mae")}})
    save("aggregate_two_day.csv", aggregates)
    bootstraps = []
    rng = np.random.default_rng(20261003)
    for _, _, noise, _ in ref.NOISES:
        matched_rows = loaded[("matched", noise)]
        ys, matched_p = ref.arrays(matched_rows, "ensemble__performance_kfold")
        comparisons = {"matched_minus_clean_only": ref.arrays(loaded[("clean_only", noise)], "ensemble__performance_kfold")[1],
                       "matched_performance_minus_equal": ref.arrays(matched_rows, "ensemble__simple_equal")[1],
                       "matched_performance_minus_RF": ref.arrays(matched_rows, "randomforest")[1]}
        for comparison, baseline in comparisons.items():
            bootstraps.append({"noise": noise, "comparison": comparison,
                               **bootstrap(matched_rows, ys, matched_p, baseline, rng)})
    save("cluster_bootstrap_rmse_deltas.csv", bootstraps)
    (OUTPUT / "run_scope.json").write_text(json.dumps({
        "runs": {k: v.relative_to(ROOT).as_posix() for k, v in RUNS.items()},
        "thresholds": ref.THRESHOLDS, "n_per_condition": 540, "training_n": 1620,
        "evaluation": "known-source-WAV unused chunks; source-day thresholds",
        "bootstrap": "paired source WAVs; not training-seed uncertainty",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("14 conditions audited; identical 540 test IDs; clean predictions identical for all 5 methods.")
    print("clean_only has one fit; matched has seven fits; 1620 OOF IDs disjoint from test.")
    for row in aggregates:
        if row["model"] == "performance":
            print(row["policy"], row["aggregation"], "RMSE", round(row["rmse"]/1000,2),
                  "ONB", round(row["rmse_onb"]/1000,2), "FPR", round(row["fpr"],4),
                  "Recall", round(row["recall"],4))


if __name__ == "__main__":
    main()
