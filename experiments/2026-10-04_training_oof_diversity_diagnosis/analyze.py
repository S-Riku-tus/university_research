"""Training-side residual and input diagnostics. Does not fit or select weights."""

import csv
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/experiments/2026-10-04_training_oof_diversity_diagnosis.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SCOPE = json.loads((ROOT / CONFIG["source_run_scope"]).read_text(encoding="utf-8"))
MODELS = ("randomforest", "conformer", "alexnet")
NOISE_DIRS = {"clean": "heatflux_no_noise", **{
    n: f"heatflux_reference_SNR={n}" for n in ("0", "-4", "-8", "-12", "-16", "-20")}}
THRESHOLDS = {"20250611": 221505.1102, "20250618": 271677.6816}
FEATURES = ["log_power_512_1000", "log_power_1000_2000", "log_power_2000_3000",
            "log_power_2100_2500", "log_power_total", "log_ratio_2000_3000_to_1000_2000",
            "log_ratio_2100_2500_to_1000_2000", "spectral_centroid_hz",
            "normalized_spectral_entropy", "temporal_power_cv"]


def source_path(path):
    return Path("\\\\?\\" + str(path.resolve())) if os.name == "nt" else path


def read_json(path):
    return json.loads(source_path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with source_path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def save_csv(name, rows):
    assert rows, name
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return str(row["source_wav_id"]), int(row["chunk_index"])


def region(y, t):
    return np.select([y < 60, y < t, y < 1.5 * t],
                     ["below_60", "60_to_ONB", "ONB_to_1.5ONB"],
                     default="above_1.5ONB")


def metrics(y, p, t):
    e = p - y
    positive, predicted_positive = y >= t, p >= t
    pre, post = ~positive, positive
    return {"n": len(y), "rmse_kW_m2": float(np.sqrt(np.mean(e**2))),
            "mae_kW_m2": float(np.mean(np.abs(e))), "bias_kW_m2": float(np.mean(e)),
            "fp": int((pre & predicted_positive).sum()),
            "fn": int((post & ~predicted_positive).sum()),
            "n_pre": int(pre.sum()), "n_post": int(post.sum()),
            "fpr": float(predicted_positive[pre].mean()) if pre.any() else "",
            "recall": float(predicted_positive[post].mean()) if post.any() else ""}


def input_features(array):
    x = np.asarray(array, dtype=np.float64)
    assert x.shape == (224, 224) and np.isfinite(x).all() and np.min(x) >= 0
    floor = CONFIG["log_floor"]
    result = {}
    powers = {}
    for low, high in CONFIG["power_bands_hz"]:
        a, b = round(224 * low / 3000), round(224 * high / 3000)
        power = float(x[:, a:b].mean())
        powers[(low, high)] = power
        result[f"log_power_{low}_{high}"] = float(np.log10(max(power, floor)))
    result["log_power_total"] = float(np.log10(max(float(x.mean()), floor)))
    for a in [(2000, 3000), (2100, 2500)]:
        result[f"log_ratio_{a[0]}_{a[1]}_to_1000_2000"] = float(
            np.log10(max(powers[a], floor) / max(powers[(1000, 2000)], floor)))
    spectral = x.mean(axis=0)
    weights = spectral / max(float(spectral.sum()), floor)
    result["spectral_centroid_hz"] = float(np.dot(weights, np.linspace(0, 3000, 224)))
    nonzero = weights > 0
    result["normalized_spectral_entropy"] = float(
        -np.sum(weights[nonzero] * np.log(weights[nonzero])) / np.log(224))
    temporal = x.mean(axis=1)
    result["temporal_power_cv"] = float(temporal.std() / max(float(temporal.mean()), floor))
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    loaded = {}
    audits, metric_rows, feasible_rows, pair_rows, wav_rows, sample_rows, support_rows = [], [], [], [], [], [], []
    ref_ids, ref_y, ref_folds, training_ids, data_root = None, None, None, None, None
    for noise in CONFIG["oof_conditions"]:
        policy = "clean_only" if noise == "clean" else "matched"
        run = ROOT / SCOPE["runs"][policy] / "maxfreq=3kHz" / NOISE_DIRS[noise]
        manifest = read_json(run / "run_manifest.json")
        audit_path = run / "internal_validation_fold1.json"
        audit = read_json(audit_path)
        split = read_json(run / "split_manifest.json")["folds"][0]
        samples = audit["samples"]
        ids = [identity(s) for s in samples]
        y = np.asarray([s["heat_flux"] / 1000 for s in samples])
        t = np.asarray([THRESHOLDS[s["source_wav_id"].split("::")[0]] / 1000 for s in samples])
        p = np.column_stack([[s[m] / 1000 for s in samples] for m in MODELS])
        fold_ids, coverage = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)
        for fold in audit["folds"]:
            fit, held = np.asarray(fold["fit_indices"]), np.asarray(fold["validation_indices"])
            assert set(fit).isdisjoint(held) and len(fit) == 1080 and len(held) == 540
            assert {ids[i][0] for i in fit}.isdisjoint({ids[i][0] for i in held})
            assert fold["shared_source_wavs"] == 0
            fold_ids[held] = fold["fold"]
            coverage[held] += 1
        assert len(samples) == len(set(ids)) == 1620 and np.all(coverage == 1)
        assert not audit["test_used"] and audit["method"] == "wav_kfold"
        assert manifest["validation_config"]["output"]["save_fitted_artifacts"] is False
        assert manifest["validation_config"]["acoustic_selection"]["peak_height_threshold"] is None
        if ref_ids is None:
            ref_ids, ref_y, ref_folds = ids, y.copy(), fold_ids.copy()
            training_ids = set(ids)
            data_root = Path(manifest["dataset"]["source_dir"]) / "maxfreq=3kHz"
            clean_metadata = read_csv(data_root / NOISE_DIRS["clean"] / "chunk_manifest.csv")
            assert len(clean_metadata) == 2160
            sorted_metadata = sorted(clean_metadata, key=lambda r: r["sample_filename"])
            eval_ids = {identity(sorted_metadata[i]) for i in split["evaluation_sample_indices"]}
            assert len(eval_ids) == 540 and training_ids.isdisjoint(eval_ids)
            assert training_ids | eval_ids == {identity(r) for r in clean_metadata}
        assert ids == ref_ids and np.array_equal(y, ref_y) and np.array_equal(fold_ids, ref_folds)
        errors = np.asarray([audit["individual_errors"][m] for m in MODELS])
        weights = 1 / np.maximum(errors, 1e-6)
        weights /= weights.sum()
        ensemble = p @ weights
        equal = p.mean(axis=1)
        e = p - y[:, None]
        regions = region(y, t)
        low, high = p.min(axis=1), p.max(axis=1)
        common_fn = (y >= t) & (high < t)
        ensemble_fn = (y >= t) & (ensemble < t)
        loaded[noise] = {"samples": samples, "y": y, "t": t, "p": p, "fold": fold_ids,
                         "weights": weights, "ensemble": ensemble, "regions": regions,
                         "common_fn": common_fn, "ensemble_fn": ensemble_fn}
        masks = {"all": np.ones(len(y), bool), **{r: regions == r for r in np.unique(regions)},
                 **{f"day_{d}": np.asarray([s["source_wav_id"].startswith(d) for s in samples]) for d in THRESHOLDS},
                 **{f"fold_{f}": fold_ids == f for f in [1, 2, 3]}}
        for scope_name, mask in masks.items():
            ym, tm, pm, em = y[mask], t[mask], p[mask], e[mask]
            for name, pred in [(m, p[:, i]) for i, m in enumerate(MODELS)] + [("performance_oof_diagnostic", ensemble), ("equal", equal)]:
                metric_rows.append({"policy": policy, "noise": noise, "scope": scope_name,
                                    "model": name, **metrics(ym, pred[mask], tm)})
            distance = np.maximum.reduce([low[mask] - ym, ym - high[mask], np.zeros(len(ym))])
            feasible_rows.append({"policy": policy, "noise": noise, "scope": scope_name,
                "n": int(mask.sum()), "n_wavs": len({samples[i]["source_wav_id"] for i in np.flatnonzero(mask)}),
                "all_under_y": int((high[mask] < ym).sum()), "all_over_y": int((low[mask] > ym).sum()),
                "inside_prediction_range": int(((low[mask] <= ym) & (ym <= high[mask])).sum()),
                "common_fn": int(common_fn[mask].sum()), "ensemble_fn": int(ensemble_fn[mask].sum()),
                "ensemble_fn_with_any_positive": int((ensemble_fn[mask] & (high[mask] >= tm)).sum()),
                "common_fp": int(((ym < tm) & (low[mask] >= tm)).sum()),
                "convex_oracle_rmse_lower_bound": float(np.sqrt(np.mean(distance**2)))})
            for a, b in combinations(range(3), 2):
                ea, eb = em[:, a], em[:, b]
                pair_rows.append({"policy": policy, "noise": noise, "scope": scope_name,
                    "model_a": MODELS[a], "model_b": MODELS[b], "n": len(ea),
                    "bias_a": float(ea.mean()), "bias_b": float(eb.mean()),
                    "residual_product_mean": float(np.mean(ea * eb)),
                    "bias_product": float(ea.mean() * eb.mean()),
                    "residual_covariance": float(np.mean((ea - ea.mean()) * (eb - eb.mean()))),
                    "residual_correlation": float(np.corrcoef(ea, eb)[0, 1]),
                    "opposite_sign_n": int((ea * eb < 0).sum())})
        for wav in sorted({s["source_wav_id"] for s in samples}):
            mask = np.asarray([s["source_wav_id"] == wav for s in samples])
            idx = np.flatnonzero(mask)
            f = next(f for f in audit["folds"] if f["fold"] == int(fold_ids[idx[0]]))
            fit_y = y[f["fit_indices"]]
            target, threshold = float(y[idx[0]]), float(t[idx[0]])
            below, above = fit_y[fit_y < target], fit_y[fit_y > target]
            support = {"source_wav_id": wav, "day": wav.split("::")[0], "fold": int(fold_ids[idx[0]]),
                "heat_flux_kW_m2": target, "onb_kW_m2": threshold,
                "nearest_fit_label_distance": float(np.min(np.abs(fit_y - target))),
                "fit_label_below": float(below.max()) if len(below) else "",
                "fit_label_above": float(above.min()) if len(above) else "",
                "outside_fit_target_range": bool(target < fit_y.min() or target > fit_y.max()),
                "fit_onb_neighborhood_wavs": len({samples[i]["source_wav_id"] for i in f["fit_indices"]
                                                  if abs(y[i] - t[i]) <= .1 * t[i]})}
            if noise == "clean":
                support_rows.append(support)
            wav_rows.append({"policy": policy, "noise": noise, **support, "n": len(idx),
                "region": str(regions[idx[0]]), "common_fn": int(common_fn[mask].sum()),
                "ensemble_fn": int(ensemble_fn[mask].sum()),
                **{f"{m}_mean_prediction": float(p[mask, j].mean()) for j, m in enumerate(MODELS)},
                **{f"{m}_bias": float(e[mask, j].mean()) for j, m in enumerate(MODELS)},
                "ensemble_mean_prediction": float(ensemble[mask].mean()),
                "ensemble_rmse": float(np.sqrt(np.mean((ensemble[mask] - y[mask])**2)))})
        for i, s in enumerate(samples):
            sample_rows.append({"policy": policy, "noise": noise, "source_wav_id": s["source_wav_id"],
                "chunk_index": int(s["chunk_index"]), "fold": int(fold_ids[i]), "y_kW_m2": float(y[i]),
                "onb_kW_m2": float(t[i]), "region": str(regions[i]),
                **{m: float(p[i, j]) for j, m in enumerate(MODELS)},
                "performance_oof_diagnostic": float(ensemble[i]), "common_fn": bool(common_fn[i]),
                "ensemble_fn": bool(ensemble_fn[i]), "max_prediction_margin_to_onb": float(high[i] - t[i])})
        audits.append({"policy": policy, "noise": noise, "n_oof": len(y), "n_wavs": 36,
            "source_run_hash": manifest["run_hash"], "fit_id": split["fit_id"],
            "n_inner_folds": 3, "shared_fit_held_wavs": 0, "test_used": False,
            "same_training_ids_and_targets": True, "same_inner_folds": True,
            "oof_sha256": hashlib.sha256(source_path(audit_path).read_bytes()).hexdigest(),
            "weights": dict(zip(MODELS, weights.tolist())), "saved_fitted_models": False})
    matched_clean = read_json(ROOT / SCOPE["runs"]["matched"] / "maxfreq=3kHz" /
                              NOISE_DIRS["clean"] / "internal_validation_fold1.json")
    assert matched_clean["samples"] == loaded["clean"]["samples"]
    for name, rows in [("oof_metrics.csv", metric_rows), ("oof_common_failures.csv", feasible_rows),
                       ("oof_residual_pairs.csv", pair_rows), ("oof_wav_profiles.csv", wav_rows),
                       ("oof_sample_diagnostics.csv", sample_rows), ("inner_target_support.csv", support_rows)]:
        save_csv(name, rows)
    print("OOF residual diagnostics complete: 7 distinct fits, 1620 aligned training chunks each.", flush=True)

    features_rows, per_wav_rows, within_rows = [], [], []
    clean = loaded["clean"]
    for noise in CONFIG["input_diagnostic_conditions"]:
        folder = data_root / NOISE_DIRS[noise]
        metadata = {identity(r): r for r in read_csv(folder / "chunk_manifest.csv")}
        assert training_ids <= set(metadata)
        for i, s in enumerate(clean["samples"]):
            r = metadata[identity(s)]
            assert abs(float(r["heat_flux"]) / 1000 - clean["y"][i]) < 1e-6
            array = np.load(source_path(folder / r["sample_filename"]), mmap_mode="r", allow_pickle=False)
            f = input_features(array)
            features_rows.append({"noise": noise, "source_wav_id": s["source_wav_id"],
                "chunk_index": int(s["chunk_index"]), "fold": int(clean["fold"][i]),
                "heat_flux_kW_m2": float(clean["y"][i]), "region": str(clean["regions"][i]),
                "clean_oof_common_fn": bool(clean["common_fn"][i]),
                "realized_snr_db": r["realized_snr_db"], **f})
        selected = [r for r in features_rows if r["noise"] == noise]
        for wav in sorted({r["source_wav_id"] for r in selected}):
            rows = [r for r in selected if r["source_wav_id"] == wav]
            per_wav_rows.append({"noise": noise, "source_wav_id": wav, "day": wav.split("::")[0],
                "heat_flux_kW_m2": rows[0]["heat_flux_kW_m2"], "region": rows[0]["region"],
                "n": len(rows), "clean_common_fn_n": sum(r["clean_oof_common_fn"] for r in rows),
                **{f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in FEATURES},
                **{f"std_{k}": float(np.std([r[k] for r in rows])) for k in FEATURES}})
            bad = [r for r in rows if r["clean_oof_common_fn"]]
            good = [r for r in rows if not r["clean_oof_common_fn"]]
            if rows[0]["region"] == "ONB_to_1.5ONB" and bad and good:
                for k in FEATURES:
                    b, g = np.asarray([r[k] for r in bad]), np.asarray([r[k] for r in good])
                    pooled_sd = float(np.std([r[k] for r in rows]))
                    within_rows.append({"noise": noise, "source_wav_id": wav, "feature": k,
                        "n_common_fn": len(bad), "n_other": len(good), "mean_common_fn": float(b.mean()),
                        "mean_other": float(g.mean()), "within_wav_standardized_difference":
                        float((b.mean() - g.mean()) / pooled_sd) if pooled_sd > 0 else ""})
        print(f"Training-only input features: {noise}, 1620 chunks streamed.", flush=True)
    save_csv("training_input_features.csv", features_rows)
    save_csv("training_wav_features.csv", per_wav_rows)
    if within_rows:
        save_csv("within_wav_common_fn_features.csv", within_rows)

    correlation_rows, paired_rows, contrast_rows = [], [], []
    for noise in CONFIG["input_diagnostic_conditions"]:
        for day in ["all", *THRESHOLDS]:
            selected = [r for r in per_wav_rows if r["noise"] == noise and (day == "all" or r["day"] == day)]
            y = np.asarray([r["heat_flux_kW_m2"] for r in selected])
            for k in FEATURES:
                vals = np.asarray([r[f"mean_{k}"] for r in selected])
                correlation_rows.append({"noise": noise, "day": day, "feature": k,
                    "n_wavs": len(selected), "spearman_rho": float(spearmanr(y, vals).correlation)})
        for day, threshold_wm in THRESHOLDS.items():
            candidates = [r for r in per_wav_rows if r["noise"] == noise and r["day"] == day]
            threshold = threshold_wm / 1000
            pre = max((r for r in candidates if r["heat_flux_kW_m2"] < threshold), key=lambda r: r["heat_flux_kW_m2"])
            onb = min(candidates, key=lambda r: abs(r["heat_flux_kW_m2"] - threshold))
            sample_pre = [r for r in features_rows if r["noise"] == noise and r["source_wav_id"] == pre["source_wav_id"]]
            sample_onb = [r for r in features_rows if r["noise"] == noise and r["source_wav_id"] == onb["source_wav_id"]]
            for k in FEATURES:
                pv, ov = np.asarray([r[k] for r in sample_pre]), np.asarray([r[k] for r in sample_onb])
                auc = roc_auc_score(np.r_[np.zeros(len(pv)), np.ones(len(ov))], np.r_[pv, ov])
                contrast_rows.append({"noise": noise, "day": day, "feature": k,
                    "pre_wav": pre["source_wav_id"], "onb_wav": onb["source_wav_id"],
                    "pre_heat_flux": pre["heat_flux_kW_m2"], "onb_heat_flux": onb["heat_flux_kW_m2"],
                    "n_pre": len(pv), "n_onb": len(ov), "pre_mean": float(pv.mean()),
                    "onb_mean": float(ov.mean()), "mean_difference_onb_minus_pre": float(ov.mean()-pv.mean()),
                    "descriptive_chunk_auc_higher_feature_is_onb": float(auc)})
    for wav in sorted({r["source_wav_id"] for r in per_wav_rows}):
        c = next(r for r in per_wav_rows if r["source_wav_id"] == wav and r["noise"] == "clean")
        n = next(r for r in per_wav_rows if r["source_wav_id"] == wav and r["noise"] == "-20")
        for k in FEATURES:
            paired_rows.append({"source_wav_id": wav, "heat_flux_kW_m2": c["heat_flux_kW_m2"],
                "region": c["region"], "feature": k, "clean_mean": c[f"mean_{k}"],
                "minus20_mean": n[f"mean_{k}"], "paired_change": n[f"mean_{k}"]-c[f"mean_{k}"]})
    save_csv("feature_heat_flux_correlations.csv", correlation_rows)
    save_csv("adjacent_onb_feature_contrasts.csv", contrast_rows)
    save_csv("paired_noise_feature_changes.csv", paired_rows)
    (OUTPUT / "input_audit.json").write_text(json.dumps({
        "config": CONFIG_PATH.relative_to(ROOT).as_posix(), "source_runs": SCOPE["runs"],
        "oof_inputs": audits, "training_outer_overlap": 0, "outer_predictions_read": False,
        "distinct_oof_fits": 7, "clean_matched_oof_identical": True,
        "training_input_arrays_read": 3240, "outer_input_arrays_read": 0,
        "input_features": FEATURES, "units": "predictions and targets converted W/m2 to kW/m2",
        "limitations": ["Pooled OOF performance with OOF-derived weights is descriptive, not independent ensemble validation.",
                        "Input band summaries are not fitted-model attributions.",
                        "Adjacent ONB contrast involves one recording per stage and day, not independent chunk experiments.",
                        "No source run or main configuration modified; no existing models retrained."]
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in feasible_rows:
        if r["noise"] in ("clean", "-20") and r["scope"] in ("below_60", "ONB_to_1.5ONB"):
            print(r)


if __name__ == "__main__":
    main()
