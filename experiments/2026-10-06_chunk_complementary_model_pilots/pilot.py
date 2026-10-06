"""Training-only complementary-model pilots, followed by frozen holdout evaluation.

Run in order: --phase baseline, --phase representations, --phase integrate,
--phase evaluate. The evaluate phase never changes fitted choices or weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs/experiments/2026-10-06_chunk_complementary_model_pilots.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SCOPE = json.loads((ROOT / CONFIG["baseline_scope"]).read_text(encoding="utf-8"))
RUN = ROOT / SCOPE["runs"]["new_chunk_clean"]
KEYS = ("randomforest", "conformer", "alexnet")
WEIGHTS = np.asarray([.1556511420710336, .49131986430665925, .35302899362230716])
THRESHOLDS = {"20250611": 221.5051102, "20250618": 271.6776816}
NOISE_DIRS = {"clean": "heatflux_no_noise", **{n: f"heatflux_reference_SNR={n}" for n in ["0", "-4", "-8", "-12", "-16", "-20"]}}
POLICIES = CONFIG["nested_selection"]["policies"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


features_ref = module("prior_feature_definition", "experiments/2026-10-04_training_oof_diversity_diagnosis/analyze.py")
metric_ref = module("prior_metric_definition", "experiments/2026-10-01_3khz_22khz_tuned_outer_comparison/analyze.py")


def lp(path):
    value = str(path.resolve())
    return Path(value if os.name != "nt" or value.startswith("\\\\?\\") else "\\\\?\\" + value)


def read_json(path):
    return json.loads(lp(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with lp(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(name, rows):
    assert rows, name
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return str(row["source_wav_id"]), int(float(row["chunk_index"]))


def setup():
    folder = RUN / "maxfreq=3kHz/heatflux_no_noise"
    oof = read_json(folder / "internal_validation_fold1.json")
    rows = oof["samples"]  # Saved fold indices refer to THIS order, not sorted IDs.
    ids = [identity(r) for r in rows]
    assert len(ids) == len(set(ids)) == 1620
    assert oof["method"] == "chunk_kfold" or oof["split_unit"] == "chunk"
    folds = [(np.asarray(f["fit_indices"], int), np.asarray(f["validation_indices"], int)) for f in oof["folds"]]
    coverage = np.zeros(len(ids), int)
    for (fit, held), (expected_fit, expected_held) in zip(folds, KFold(3, shuffle=True, random_state=42).split(ids)):
        assert np.array_equal(fit, expected_fit) and np.array_equal(held, expected_held)
        assert set(fit).isdisjoint(held) and len(fit) == 1080 and len(held) == 540
        coverage[held] += 1
    assert np.all(coverage == 1)
    manifest = read_json(folder / "run_manifest.json")
    data_root = Path(manifest["dataset"]["source_dir"]) / "maxfreq=3kHz"
    metadata = sorted(read_csv(data_root / NOISE_DIRS["clean"] / "chunk_manifest.csv"), key=lambda r: r["sample_filename"])
    split = read_json(folder / "split_manifest.json")["folds"][0]
    outer = [metadata[i] for i in split["evaluation_sample_indices"]]
    assert len(outer) == 540 and not set(ids) & {identity(r) for r in outer}
    assert set(ids) | {identity(r) for r in outer} == {identity(r) for r in metadata}
    base = np.column_stack([[float(r[k]) / 1000 for r in rows] for k in KEYS])
    y = np.asarray([float(r["heat_flux"]) / 1000 for r in rows])
    days = np.asarray([r["source_wav_id"][:8] for r in rows])
    return rows, folds, data_root, outer, base, y, days


def acoustic_features(array):
    x = np.asarray(array, dtype=np.float64)
    baseline = features_ref.input_features(x)
    floor = CONFIG["log_floor"]
    values = [baseline[name] for name in features_ref.FEATURES]
    names = list(features_ref.FEATURES)
    for low in range(0, 3000, 125):
        a, b = round(224 * low / 3000), round(224 * (low + 125) / 3000)
        values.append(np.log10(max(float(x[:, a:b].mean()), floor)) - baseline["log_power_total"])
        names.append(f"relative_log_power_{low}_{low+125}")
    for low in range(0, 3000, 500):
        a, b = round(224 * low / 3000), round(224 * (low + 500) / 3000)
        temporal = x[:, a:b].mean(axis=1)
        mean = max(float(temporal.mean()), floor)
        q = np.quantile(temporal, [.1, .5, .9])
        values.extend([*np.log10(np.maximum(q, floor)), temporal.std()/mean, temporal.max()/mean, float((temporal > 2*mean).mean())])
        names.extend([f"time_{low}_{low+500}_{stat}" for stat in ["log_q10", "log_q50", "log_q90", "cv", "peak_mean", "fraction_gt2mean"]])
    assert len(values) == 70 and np.isfinite(values).all()
    return values, names


def load_features(rows, data_root, noise, unit):
    path = OUT / f"features_{unit}_{noise}.npz"
    ids = np.asarray([f"{r['source_wav_id']}|{int(float(r['chunk_index']))}" for r in rows])
    if path.exists():
        with np.load(path) as saved:
            assert np.array_equal(saved["ids"], ids)
            return saved["X"].copy(), saved["names"].tolist()
    metadata = {identity(r): r for r in read_csv(data_root / NOISE_DIRS[noise] / "chunk_manifest.csv")}
    values = []
    for r in rows:
        raw = np.load(lp(data_root / NOISE_DIRS[noise] / metadata[identity(r)]["sample_filename"]), allow_pickle=False)
        feature, names = acoustic_features(raw)
        values.append(feature)
    X = np.asarray(values)
    assert X.shape == (len(rows), 70)
    np.savez_compressed(path, X=X, ids=ids, names=np.asarray(names))
    print(f"Extracted {unit} {noise}: {X.shape}", flush=True)
    return X, names


def representation(X, name):
    if name == "band10":
        return X[:, :10]
    if name == "frequency34":
        return X[:, :34]
    assert name == "temporal46"
    return np.column_stack([X[:, :10], X[:, 34:]])


def endpoints(y, p, days):
    result = []
    for day in THRESHOLDS:
        mask = days == day
        threshold = THRESHOLDS[day]
        q, gap = metric_ref.q100(y[mask], p[mask], threshold)
        result.append({"day": day, "q100": q, "g100": gap, **metric_ref.metrics(y[mask], p[mask], threshold)})
    return result


def score(y, p, days, policy):
    rmse = float(np.sqrt(np.mean((p-y)**2)))
    ep = endpoints(y, p, days)
    gaps = [r["g100"] if np.isfinite(r["g100"]) else float("inf") for r in ep]
    fp = sum(r["fp"] for r in ep)
    if policy == "rmse":
        return (rmse,)
    if policy == "q100_then_fp":
        return max(gaps), float(np.mean(gaps)), fp, rmse
    return fp, max(gaps), float(np.mean(gaps)), rmse


def make_model(family, parameter, seed):
    params = CONFIG["models"][family].copy()
    grid_key = "C" if family == "SVR" else "max_leaf_nodes"
    params[grid_key] = parameter
    regressor = make_pipeline(StandardScaler(), SVR(**params)) if family == "SVR" else HistGradientBoostingRegressor(**params, random_state=seed)
    return TransformedTargetRegressor(regressor=regressor, transformer=MinMaxScaler())


def record_summary(method, unit, noise, y, p, days):
    threshold = np.asarray([THRESHOLDS[d] for d in days])
    return [{"method": method, "unit": unit, "noise": noise, "day": "two_day", "q100": float("nan"), "g100": float("nan"), **metric_ref.metrics(y, p, threshold)},
            *[{"method": method, "unit": unit, "noise": noise, **e} for e in endpoints(y, p, days)]]


def candidates():
    return read_json(OUT / "candidate_registry.json") if (OUT / "candidate_registry.json").exists() else []


def train(representations):
    rows, folds, data_root, outer, base, y, days = setup()
    X = {noise: load_features(rows, data_root, noise, "train")[0] for noise in ["clean", "-20"]}
    # Reusing the old feature definition must preserve every value for the same IDs.
    previous = { (r["noise"], identity(r)): r for r in read_csv(ROOT / "experiments/2026-10-04_training_oof_diversity_diagnosis/training_input_features.csv") }
    for noise in X:
        expected = np.asarray([[float(previous[(noise, identity(r))][name]) for name in features_ref.FEATURES] for r in rows])
        np.testing.assert_allclose(X[noise][:, :10], expected, rtol=1e-12, atol=1e-12)
    registry = candidates()
    for rep in representations:
        assert not any(r["representation"] == rep for r in registry), "Do not silently overwrite completed pilots"
        xr = {n: representation(x, rep) for n, x in X.items()}
        for family in CONFIG["models"]:
            pred = {(policy, noise): np.full(len(y), np.nan) for policy in POLICIES for noise in X}
            audits, inner_records = [], []
            fits = 0
            for fold, (fit, held) in enumerate(folds, 1):
                inner_folds = [(fit[a], fit[b]) for a, b in KFold(3, shuffle=True, random_state=42+fold).split(fit)]
                options = []
                for parameter in CONFIG["models"][family]["C" if family == "SVR" else "max_leaf_nodes"]:
                    inner_pred = np.full(len(y), np.nan)
                    coverage = np.zeros(len(y), int)
                    for inner_fit, inner_held in inner_folds:
                        assert not set(inner_fit) & set(held) and not set(inner_held) & set(held)
                        model = make_model(family, parameter, 42+fold)
                        model.fit(xr["clean"][inner_fit], y[inner_fit]); fits += 1
                        inner_pred[inner_held] = model.predict(xr["clean"][inner_held])
                        coverage[inner_held] += 1
                    assert np.all(coverage[fit] == 1) and np.all(coverage[held] == 0)
                    scores = {policy: score(y[fit], inner_pred[fit], days[fit], policy) for policy in POLICIES}
                    options.append((parameter, scores))
                    inner_records.append({"fold": fold, "parameter": parameter, "scores": scores})
                selected = {policy: min(options, key=lambda c: (c[1][policy], c[0]))[0] for policy in POLICIES}
                for parameter in sorted(set(selected.values())):
                    model = make_model(family, parameter, 42+fold)
                    model.fit(xr["clean"][fit], y[fit]); fits += 1
                    filename = f"{rep}_{family}_fold{fold}_p{parameter}.joblib"
                    joblib.dump(model, OUT / filename, compress=3)
                    result = {noise: model.predict(x[held]) for noise, x in xr.items()}
                    np.testing.assert_allclose(joblib.load(OUT / filename).predict(xr["clean"][held]), result["clean"], rtol=1e-12, atol=1e-10)
                    for policy in POLICIES:
                        if selected[policy] == parameter:
                            for noise in X:
                                pred[(policy, noise)][held] = result[noise]
                            audits.append({"policy": policy, "fold": fold, "parameter": parameter, "fit_indices": fit.tolist(), "held_indices": held.tolist(), "artifact": filename, "reload_verified": True, "shared_chunks": 0, "shared_wavs": 36})
                print(f"{rep} {family} fold{fold}: {selected}", flush=True)
            assert all(np.isfinite(p).all() for p in pred.values())
            # Final parameter selection uses ONLY training CV on all 1620 chunks.
            full_options = []
            for parameter in CONFIG["models"][family]["C" if family == "SVR" else "max_leaf_nodes"]:
                full_pred = np.full(len(y), np.nan)
                for fold, (fit, held) in enumerate(folds, 1):
                    model = make_model(family, parameter, 42+fold)
                    model.fit(xr["clean"][fit], y[fit]); fits += 1
                    full_pred[held] = model.predict(xr["clean"][held])
                full_options.append((parameter, {policy: score(y, full_pred, days, policy) for policy in POLICIES}))
            full_selected = {policy: min(full_options, key=lambda c: (c[1][policy], c[0]))[0] for policy in POLICIES}
            for parameter in sorted(set(full_selected.values())):
                model = make_model(family, parameter, 42)
                model.fit(xr["clean"], y); fits += 1
                joblib.dump(model, OUT / f"{rep}_{family}_final_p{parameter}.joblib", compress=3)
                np.testing.assert_allclose(joblib.load(OUT / f"{rep}_{family}_final_p{parameter}.joblib").predict(xr["clean"][:8]), model.predict(xr["clean"][:8]), rtol=1e-12, atol=1e-10)
            summary = []
            for policy in POLICIES:
                name = f"{rep}_{family}_{policy}"
                np.savez_compressed(OUT / f"{name}_oof.npz", clean=pred[(policy, "clean")], minus20=pred[(policy, "-20")])
                registry.append({"method": name, "representation": rep, "family": family, "policy": policy, "final_parameter": full_selected[policy], "final_artifact": f"{rep}_{family}_final_p{full_selected[policy]}.joblib", "n_features": xr["clean"].shape[1]})
                for noise in X:
                    summary.extend(record_summary(name, "training_nested_oof", noise, y, pred[(policy, noise)], days))
            save_csv(f"{rep}_{family}_metrics.csv", summary)
            save_json(f"{rep}_{family}_audit.json", {"fit_operations": fits, "outer_labels_used": False, "noise_used_for_selection": False, "sklearn_version": sklearn.__version__, "nested_fits": audits, "inner_grid": inner_records, "final_cv_scores": full_options})
            save_json("candidate_registry.json", registry)
            for r in summary:
                if r["noise"] == "clean" and r["day"] != "two_day":
                    print(r["method"], r["day"], "q100", round(r["q100"], 2), "FP", r["fp"], "FN", r["fn"], flush=True)


def fit_simplex(matrix, y, correlation_penalty=False):
    n = matrix.shape[1]
    residuals = matrix-y[:, None]
    correlation = np.corrcoef(residuals, rowvar=False)
    penalty = np.maximum(correlation, 0)
    np.fill_diagonal(penalty, 0)
    strength = .25 * float(np.mean((matrix[:, :3] @ WEIGHTS - y)**2)) if correlation_penalty else 0.0
    def objective(w):
        return float(np.mean((matrix @ w-y)**2) + strength*(w @ penalty @ w))
    fitted = minimize(objective, np.full(n, 1/n), method="SLSQP", bounds=[(0, 1)]*n, constraints={"type": "eq", "fun": lambda w: w.sum()-1}, options={"ftol": 1e-9, "maxiter": 1000})
    assert fitted.success, fitted.message
    w = np.maximum(fitted.x, 0); w /= w.sum()
    return w, correlation, strength


def integrate():
    rows, folds, data_root, outer, base, y, days = setup()
    assert len(candidates()) == 18
    pool = {r["method"]: np.load(OUT / f"{r['method']}_oof.npz")["clean"] for r in candidates()}
    t = np.asarray([THRESHOLDS[d] for d in days])
    common = (y >= t) & np.all(base < t[:, None], axis=1)
    last = np.zeros(len(y), bool)
    for day in THRESHOLDS:
        mask = days == day
        failure = mask & (y >= t) & (base @ WEIGHTS < t)
        last |= common & mask & (y == y[failure].max())
    assert last.sum() == 5 and common.sum() == 52
    shortlist = []
    for policy in POLICIES:
        shortlist.append(min(pool, key=lambda name: (score(y, pool[name], days, policy), name)))
    shortlist.append(min(pool, key=lambda name: (-int((last & (pool[name] >= t)).sum()), -int((common & (pool[name] >= t)).sum()), score(y, pool[name], days, "q100_then_fp"), name)))
    unique = []
    for name in shortlist:
        if not any(np.array_equal(pool[name], pool[other]) for other in unique):
            unique.append(name)
    catalog = [ {"method": "baseline_performance", "candidate": None, "weights": WEIGHTS.tolist(), "kind": "linear"},
                {"method": "baseline_equal", "candidate": None, "weights": [1/3]*3, "kind": "linear"},
                {"method": "conformer_alexnet_equal", "candidate": None, "weights": [0, .5, .5], "kind": "linear"},
                {"method": "rf_conformer_equal", "candidate": None, "weights": [.5, .5, 0], "kind": "linear"},
                {"method": "rf_alexnet_equal", "candidate": None, "weights": [.5, 0, .5], "kind": "linear"} ]
    correlations = []
    for name in unique:
        matrix = np.column_stack([base, pool[name]])
        mse = np.mean((matrix-y[:, None])**2, axis=0)
        inverse = 1/mse; inverse /= inverse.sum()
        for label, weight in [("equal4", np.full(4, .25)), ("inverse_mse4", inverse), ("baseline75_candidate25", np.r_[.75*WEIGHTS, .25])]:
            catalog.append({"method": f"{name}__{label}", "candidate": name, "weights": weight.tolist(), "kind": "linear"})
        for diverse in [False, True]:
            weight, correlation, strength = fit_simplex(matrix, y, diverse)
            label = "diversity_simplex" if diverse else "mse_simplex"
            catalog.append({"method": f"{name}__{label}", "candidate": name, "weights": weight.tolist(), "kind": "linear", "correlation_penalty_strength": strength})
        grid = []
        for alpha in CONFIG["onb_weight_grid"]["candidate_fraction"]:
            for beta in CONFIG["onb_weight_grid"]["rf_fraction_within_existing"]:
                for ca in CONFIG["onb_weight_grid"]["conformer_alexnet_relative_weights"]:
                    existing = np.asarray([beta, (1-beta)*ca[0], (1-beta)*ca[1]])
                    grid.append(np.r_[(1-alpha)*existing, alpha])
        for policy in ["q100_then_fp", "fp_then_q100"]:
            weight = min(grid, key=lambda w: (score(y, matrix @ w, days, policy), tuple(w)))
            catalog.append({"method": f"{name}__{policy}_weight_grid", "candidate": name, "weights": weight.tolist(), "kind": "linear", "selection_status": "training fit diagnostic"})
        for j, key in enumerate(KEYS):
            correlations.append({"candidate": name, "existing_model": key, "residual_correlation": float(correlation[j, 3]), "common_fn_corrected": int((common & (pool[name] >= t)).sum()), "last_common_fn_corrected": int((last & (pool[name] >= t)).sum()), "new_fn_vs_performance": int(((y >= t) & (base @ WEIGHTS >= t) & (pool[name] < t)).sum())})
        if (last & (pool[name] >= t)).any():
            meta = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["ridge_alpha"]))
            meta.fit(matrix, y)
            artifact = f"{name}_affine_ridge.joblib"
            joblib.dump(meta, OUT / artifact, compress=3)
            np.testing.assert_allclose(meta.predict(matrix), joblib.load(OUT / artifact).predict(matrix), rtol=1e-12, atol=1e-10)
            catalog.append({"method": f"{name}__affine_ridge", "candidate": name, "kind": "ridge", "artifact": artifact})
    save_csv("shortlist_diversity.csv", correlations)
    save_json("frozen_integration.json", {"selected_from": "clean training OOF only", "outer_labels_used": False, "shortlist": unique, "methods": catalog, "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), "ensemble_oof_status": "fit diagnostic, not nested independent validation", "matched_not_retrained": True})
    summaries = []
    diagnostics = []
    all_predictions = {"randomforest": base[:, 0], "conformer": base[:, 1], "alexnet": base[:, 2], **pool}
    for method in catalog:
        matrix = base if method["candidate"] is None else np.column_stack([base, pool[method["candidate"]]])
        all_predictions[method["method"]] = matrix @ method["weights"] if method["kind"] == "linear" else joblib.load(OUT / method["artifact"]).predict(matrix)
    for name, p in all_predictions.items():
        summaries.extend(record_summary(name, "training_fit_diagnostic" if name not in pool and name not in KEYS else "training_oof", "clean", y, p, days))
        diagnostics.append({"method": name, "common_fn_corrected": int((common & (p >= t)).sum()), "last_common_fn_corrected": int((last & (p >= t)).sum()), "new_fn_vs_performance": int(((y >= t) & (base @ WEIGHTS >= t) & (p < t)).sum()), "fp": int(((y < t) & (p >= t)).sum())})
    save_csv("training_comparison.csv", summaries)
    save_csv("training_complementarity.csv", diagnostics)
    save_json("training_predictions.json", {"ids": [identity(r) for r in rows], "methods": {k: v.tolist() for k, v in all_predictions.items()}})
    print("Frozen integration shortlist:", unique, "methods:", len(catalog), flush=True)


def evaluate():
    rows, folds, data_root, outer, base, y_train, train_days = setup()
    frozen_path = OUT / "frozen_integration.json"
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    assert frozen["config_sha256"] == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    registry = candidates()
    summaries, predictions, complementarity, stage_profiles = [], [], [], []
    for noise, dirname in NOISE_DIRS.items():
        suffix = "no_noise" if noise == "clean" else noise
        saved = read_csv(RUN / "maxfreq=3kHz" / dirname / "fold_pred" / f"pred_f1_{suffix}.csv")
        lookup = {identity(r): r for r in saved}
        assert set(lookup) == {identity(r) for r in outer}
        ordered = [lookup[identity(r)] for r in outer]
        y = np.asarray([float(r["y_true"])/1000 for r in ordered])
        days = np.asarray([r["source_wav_id"][:8] for r in outer])
        thresholds = np.asarray([THRESHOLDS[d] for d in days])
        base_matrix = np.column_stack([[float(r[k])/1000 for r in ordered] for k in KEYS])
        full_features = load_features(outer, data_root, noise, "outer")[0]
        pool = {}
        for item in registry:
            model = joblib.load(OUT / item["final_artifact"])
            pool[item["method"]] = model.predict(representation(full_features, item["representation"]))
        all_pred = {key: base_matrix[:, j] for j, key in enumerate(KEYS)}
        all_pred.update(pool)
        for method in frozen["methods"]:
            matrix = base_matrix if method["candidate"] is None else np.column_stack([base_matrix, pool[method["candidate"]]])
            all_pred[method["method"]] = matrix @ method["weights"] if method["kind"] == "linear" else joblib.load(OUT / method["artifact"]).predict(matrix)
        np.testing.assert_allclose(all_pred["baseline_performance"], [float(r["ensemble__performance_kfold"])/1000 for r in ordered], rtol=1e-6, atol=1e-4)
        baseline = all_pred["baseline_performance"]
        for name, p in all_pred.items():
            assert np.isfinite(p).all()
            summaries.extend(record_summary(name, "outer_unused_chunk", noise, y, p, days))
            for day in THRESHOLDS:
                mask = days == day
                bfn = mask & (y >= thresholds) & (baseline < thresholds)
                common = mask & (y >= thresholds) & np.all(base_matrix < thresholds[:, None], axis=1)
                last = float(y[bfn].max()) if bfn.any() else float("nan")
                complementarity.append({"method": name, "noise": noise, "day": day, "baseline_fn": int(bfn.sum()), "fn_corrected": int((bfn & (p >= thresholds)).sum()), "common_fn": int(common.sum()), "common_fn_corrected": int((common & (p >= thresholds)).sum()), "last_baseline_stage": last, "last_negatives_corrected": int((bfn & (y == last) & (p >= thresholds)).sum()), "new_fn": int((mask & (y >= thresholds) & (baseline >= thresholds) & (p < thresholds)).sum()), "new_fn_at_or_above_last": int((mask & (y >= last) & (baseline >= thresholds) & (p < thresholds)).sum()), "fp": int((mask & (y < thresholds) & (p >= thresholds)).sum())})
            for wav in sorted({r["source_wav_id"] for r in outer}):
                mask = np.asarray([r["source_wav_id"] == wav for r in outer])
                stage_profiles.append({"method": name, "noise": noise, "source_wav_id": wav, "day": wav[:8], "y": float(y[mask][0]), "threshold": float(thresholds[mask][0]), "n": int(mask.sum()), "n_positive": int((p[mask] >= thresholds[mask]).sum()), "min_margin": float(np.min(p[mask]-thresholds[mask]))})
            for i, r in enumerate(outer):
                predictions.append({"method": name, "noise": noise, "source_wav_id": r["source_wav_id"], "chunk_index": int(r["chunk_index"]), "y_kW_m2": y[i], "onb_kW_m2": thresholds[i], "prediction_kW_m2": p[i]})
        print(f"Evaluated {noise}: {len(all_pred)} methods x {len(outer)} chunks", flush=True)
    assert frozen_path.read_bytes() == frozen_bytes
    save_csv("outer_metrics.csv", summaries)
    save_csv("outer_predictions.csv", predictions)
    save_csv("outer_complementarity.csv", complementarity)
    save_csv("outer_stage_profiles.csv", stage_profiles)
    save_json("evaluation_audit.json", {"training_chunks": 1620, "outer_chunks": 540, "shared_chunks": 0, "shared_wavs": 36, "noises": list(NOISE_DIRS), "candidate_methods": len(registry), "integration_methods": len(frozen["methods"]), "frozen_integration_sha256": hashlib.sha256(frozen_bytes).hexdigest(), "config_sha256": frozen["config_sha256"], "selection_on_outer_labels": False, "frozen_unchanged_after_evaluation": True, "production_model_retraining": False, "matched_retraining": False, "status": "Completed developmental comparison on known-WAV unused chunks"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["baseline", "representations", "integrate", "evaluate"])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.phase in ["baseline", "representations"]:
            train(["band10"] if args.phase == "baseline" else ["frequency34", "temporal46"])
        elif args.phase == "integrate":
            integrate()
        else:
            evaluate()
