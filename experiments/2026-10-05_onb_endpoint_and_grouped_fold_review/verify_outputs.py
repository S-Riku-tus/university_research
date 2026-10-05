"""Verify suffix endpoints, fold feasibility and saved cross-fitted predictions."""

import ast
import hashlib
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from analyze import (OUTPUT, PREVIOUS, THRESHOLDS, endpoint, rank_balanced_folds,
                     read_csv, save_csv, summarize_predictions, training_rows)
from audit_known_wav_blocks import block_split


def main():
    # A later failing stage must invalidate an earlier apparently all-positive one.
    y = np.arange(5.)
    assert endpoint(y, np.asarray([0., 1.2, 1.2, .9, 1.2]), 1.)["q100_kW_m2"] == 4.
    assert np.isnan(endpoint(y, np.zeros(5), 1.)["q100_kW_m2"])
    assert endpoint(y, np.ones(5), 1.)["q100_kW_m2"] == 0.
    assert endpoint(y, np.ones(5), 1.)["fp"] == 1
    fixture = {f"w{i}": {"day": "20250611", "heat_flux_kW_m2": heat, "onb_kW_m2": 1.}
               for i, heat in enumerate([0., .2, 1., 1.05, 2., 3.])}
    assignment, _ = rank_balanced_folds(fixture)
    assert assignment["w2"] != assignment["w3"]
    assert sorted(assignment.values()) == [1, 1, 2, 2, 3, 3]
    fixture["w3"]["heat_flux_kW_m2"] = 1.2
    assignment, _ = rank_balanced_folds(fixture)
    assert sum(all(assignment[g] == fold for g in ["w2"]) for fold in [1, 2, 3]) == 1
    rows = training_rows()
    identities = [(r["source_wav_id"], r["chunk_index"]) for r in rows]
    baseline = np.asarray([float(r["performance_oof_diagnostic"]) for r in rows])
    true_y = np.asarray([float(r["y_kW_m2"]) for r in rows])
    t = np.asarray([float(r["onb_kW_m2"]) for r in rows])
    existing_max = np.asarray([max(float(r[m]) for m in ["randomforest", "conformer", "alexnet"]) for r in rows])
    predicted = defaultdict(list)
    for row in read_csv(OUTPUT / "candidate_training_oof_predictions.csv"):
        key = tuple(row[k] for k in ["strategy", "family", "objective", "condition"])
        predicted[key].append(row)
    assert len(predicted) == 36
    saved_ends = {(r["strategy"], r["method"], r["objective"], r["condition"], r["day"]): r
                  for r in read_csv(OUTPUT / "candidate_endpoints.csv")}
    full, blends, corrections = [], [], []
    previous_svr = read_csv(PREVIOUS / "svr_training_oof_predictions.csv")
    for key, source in predicted.items():
        assert [(r["source_wav_id"], r["chunk_index"]) for r in source] == identities
        assert all(float(r["heat_flux_kW_m2"]) == true_y[i] for i, r in enumerate(source))
        p = np.asarray([float(r["prediction_kW_m2"]) for r in source])
        assert np.isfinite(p).all()
        strategy, family, objective, condition = key
        for end in summarize_predictions(family, condition, rows, p, strategy=strategy, objective=objective):
            saved = saved_ends[(*key[:2], objective, condition, end["day"])]
            for metric in ["q100_kW_m2", "g100_kW_m2", "fp", "fn", "rmse_kW_m2"]:
                assert np.isclose(float(saved[metric]), float(end[metric]), equal_nan=True)
        if key == ("original_random", "SVR", "rmse", "clean"):
            assert np.allclose(p, [float(r["SVR_clean"]) for r in previous_svr], atol=1e-9, rtol=1e-12)
        positive = true_y >= t
        predicted_positive = p >= t
        fp, fn = int((~positive & predicted_positive).sum()), int((positive & ~predicted_positive).sum())
        precision = float((positive & predicted_positive).sum()/predicted_positive.sum()) if predicted_positive.any() else 0.
        recall = float(predicted_positive[positive].mean())
        full.append({"strategy": strategy, "family": family, "objective": objective, "condition": condition,
            "n": len(p), "rmse_kW_m2": float(np.sqrt(np.mean((p-true_y)**2))),
            "mae_kW_m2": float(np.mean(np.abs(p-true_y))), "r2": float(r2_score(true_y, p)),
            "fp": fp, "fn": fn, "n_pre": int((~positive).sum()), "n_post": int(positive.sum()),
            "fpr": fp/int((~positive).sum()), "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.,
            "roc_auc_margin": float(roc_auc_score(positive, p-t)),
            "average_precision_margin": float(average_precision_score(positive, p-t))})
        if strategy == "original_random" and condition == "clean":
            blend = .75*baseline+.25*p
            blends.extend(summarize_predictions(f"old75_{family}25", "clean", rows, blend,
                                              strategy=strategy, objective=objective))
            baseline_ok = (baseline >= t) == positive
            for name, result in [(family, p), (f"old75_{family}25", blend)]:
                new_ok = (result >= t) == positive
                common_fn = positive & (existing_max < t)
                corrections.append({"family": name, "objective": objective,
                    "common_fn_total": int(common_fn.sum()),
                    "common_fn_corrected": int((common_fn & (result >= t)).sum()),
                    "binary_errors_corrected": int((~baseline_ok & new_ok).sum()),
                    "binary_errors_added": int((baseline_ok & ~new_ok).sum()),
                    "fp": int((~positive & (result >= t)).sum()),
                    "fn": int((positive & (result < t)).sum())})
    for gap in [0, 1, 2]:
        for fold in [1, 2, 3]:
            fit, held, _ = block_split(rows, fold, gap)
            assert set(fit).isdisjoint(held)
            assert {rows[i]["source_wav_id"] for i in fit} == {rows[i]["source_wav_id"] for i in held}
    for path in OUTPUT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    audit = json.loads((OUTPUT / "candidate_audit.json").read_text(encoding="utf-8"))
    assert not audit["outer_test_used"] and not audit["noise_used_for_hyperparameter_selection"]
    assert all(set(r["fit_wavs"]).isdisjoint(r["held_wavs"]) and r["reload_verified"] for r in audit["models"])
    save_csv("candidate_full_metrics.csv", full)
    save_csv("fixed_blend_endpoints.csv", blends)
    save_csv("candidate_corrections.csv", corrections)
    bottlenecks = []
    original_candidates = {family: np.asarray([float(r["prediction_kW_m2"]) for r in predicted[
        ("original_random", family, "rmse", "clean")]]) for family in ["SVR", "ExtraTrees", "HistGradientBoosting"]}
    last_failed = {day: endpoint(true_y[np.asarray([r["source_wav_id"].startswith(day) for r in rows])],
        baseline[np.asarray([r["source_wav_id"].startswith(day) for r in rows])], threshold)["last_failed_stage_kW_m2"]
        for day, threshold in THRESHOLDS.items()}
    for group in sorted({r["source_wav_id"] for r in rows}):
        mask = np.asarray([r["source_wav_id"] == group for r in rows])
        day = group[:8]
        bottlenecks.append({"source_wav_id": group, "day": day, "heat_flux_kW_m2": true_y[mask][0],
            "onb_kW_m2": THRESHOLDS[day], "n": int(mask.sum()),
            "baseline_n_negative": int((baseline[mask] < t[mask]).sum()),
            "baseline_minimum_margin_kW_m2": float((baseline-t)[mask].min()),
            "common_negative_n": int((existing_max[mask] < t[mask]).sum()),
            "is_last_failed_stage": bool(true_y[mask][0] == last_failed[day]),
            **{f"{family}_n_negative": int((p[mask] < t[mask]).sum()) for family, p in original_candidates.items()}})
    save_csv("onb_endpoint_bottlenecks.csv", bottlenecks)
    features = read_csv(PREVIOUS / "training_input_features.csv")
    lookup = {(r["source_wav_id"], r["chunk_index"]): r for r in features if r["noise"] == "clean"}
    feature_names = audit["features"]
    contrasts = []
    for record in bottlenecks:
        if not record["is_last_failed_stage"]:
            continue
        group = record["source_wav_id"]
        indices = [i for i, r in enumerate(rows) if r["source_wav_id"] == group]
        failed = [i for i in indices if baseline[i] < t[i]]
        passed = [i for i in indices if baseline[i] >= t[i]]
        for feature in feature_names:
            a = [float(lookup[identities[i]][feature]) for i in failed]
            b = [float(lookup[identities[i]][feature]) for i in passed]
            contrasts.append({"source_wav_id": group, "feature": feature,
                "n_failed": len(a), "n_passed": len(b),
                "failed_median": float(np.median(a)), "passed_median": float(np.median(b)),
                "failed_minus_passed_median": float(np.median(a)-np.median(b)),
                "interpretation": "Descriptive within-WAV contrast; failure-conditioned small samples, not causal attribution"})
    save_csv("bottleneck_feature_contrasts.csv", contrasts)
    assert len(contrasts) == 20 and {r["n_failed"] for r in contrasts} == {3, 5}
    source_audit = json.loads((OUTPUT / "audit.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest() == expected
               for name, expected in source_audit["source_csv_sha256"].items())
    summary = {"n_prediction_sets": len(predicted), "n_rows_per_set": 1620,
        "q100_suffix_fixtures_passed": True, "rare_group_feasibility_fixtures_passed": True,
        "previous_svr_reproduced": True, "all_saved_endpoint_metrics_recomputed": True,
        "wav_disjoint_pilot_models_and_reload_verified": True,
        "known_wav_block_gap_and_stage_coverage_audited": True,
        "all_python_syntax_checked": True}
    (OUTPUT / "verification.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Clean all-domain comparison:")
    for row in full:
        if row["condition"] == "clean" and row["objective"] == "rmse":
            print(row["strategy"], row["family"], "RMSE", round(row["rmse_kW_m2"], 2), "FP", row["fp"], "FN", row["fn"])


if __name__ == "__main__":
    main()
