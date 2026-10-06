"""Nested clean-only candidates added to the unchanged original three models.

Run --phase train, --phase integrate, --phase evaluate, then --phase verify.
All predictions/targets use kW/m2. Historical fits and predictions are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor, XGBRFRegressor

from run_hgb_complementarity_validation import NOISES, THRESHOLDS, read_json, save_csv, save_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/2026-10-06_fixed_three_additions.json"
CONFIG = read_json(CONFIG_PATH)
CONFIG_HASH = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
OUT = ROOT / "experiments/2026-10-06_fixed_three_additions"
SOURCE = ROOT / CONFIG["source"]
FAMILIES = list(CONFIG["families"])
KEYS = ["randomforest", "conformer", "alexnet", "hgb", *FAMILIES]


def load_seed(seed):
    folder = SOURCE / f"seed{seed}"
    manifest = read_json(folder / "manifest.json")
    with np.load(folder / "base_predictions.npz", allow_pickle=False) as saved:
        base = {k: saved[k].copy() for k in saved.files}
    np.testing.assert_array_equal(base["train_indices"], manifest["train_indices"])
    np.testing.assert_array_equal(base["test_indices"], manifest["test_indices"])
    np.testing.assert_array_equal(base["keys"], KEYS[:4])
    np.testing.assert_array_equal(base["noises"], NOISES)
    assert set(base["train_indices"]).isdisjoint(base["test_indices"])
    return manifest, base


def labels(manifest, indices):
    # Caller passes fit or evaluation indices explicitly; selection never reads test labels.
    return np.asarray([float(manifest["samples"][int(i)]["sample_filename"].split("_")[0]) / 1000 for i in indices])


def prepare_features():
    manifest, _ = load_seed(CONFIG["seeds"][0])
    ids = np.asarray([f"{r['source_wav_id']}|{int(float(r['chunk_index']))}" for r in manifest["samples"]])
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    features = {}
    for noise in NOISES:
        with np.load(SOURCE / "cache" / f"features_{noise}.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], ids)
            frequency = saved["X"].copy()
        features[(noise, "frequency34")] = frequency
        path = cache / f"temporal46_{noise}.npz"
        if not path.exists():
            raw = np.load(SOURCE / "cache" / f"raw_{noise}.npy", mmap_mode="r", allow_pickle=False)
            values = [frequency[:, :10]]
            for low in range(0, 3000, 500):
                a, b = round(224 * low / 3000), round(224 * (low + 500) / 3000)
                temporal = raw[:, :, a:b, 0].mean(axis=2, dtype=np.float64)
                mean = np.maximum(temporal.mean(axis=1), 1e-20)
                quantiles = np.log10(np.maximum(np.quantile(temporal, [.1, .5, .9], axis=1), 1e-20)).T
                values.append(np.column_stack([quantiles, temporal.std(axis=1) / mean,
                    temporal.max(axis=1) / mean, (temporal > 2 * mean[:, None]).mean(axis=1)]))
            temporal46 = np.column_stack(values)
            assert temporal46.shape == (len(ids), 46) and np.isfinite(temporal46).all()
            np.savez_compressed(path, X=temporal46, ids=ids)
            print(f"[features] {noise}: temporal46 aligned", flush=True)
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], ids)
            features[(noise, "temporal46")] = saved["X"].copy()
    return features


def make_model(family, params, seed):
    if family in ["hgb_regularized", "hgb_temporal"]:
        regressor = HistGradientBoostingRegressor(**{**CONFIG["hgb_defaults"], **params}, random_state=seed)
    elif family == "extra_trees":
        regressor = ExtraTreesRegressor(n_estimators=256, n_jobs=2, random_state=seed, **params)
    elif family == "rf34":
        regressor = XGBRFRegressor(n_estimators=100, subsample=.6, colsample_bynode=.6,
            n_jobs=2, tree_method="hist", random_state=seed, **params)
    elif family == "gradient_boosting":
        regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=.05,
            min_samples_leaf=10, random_state=seed, **params)
    elif family == "xgb_boosting":
        regressor = XGBRegressor(n_estimators=250, learning_rate=.05, min_child_weight=10,
            subsample=.9, colsample_bytree=.9, n_jobs=2, tree_method="hist", random_state=seed, **params)
    elif family == "ridge_direct":
        regressor = make_pipeline(StandardScaler(), Ridge(**params))
    elif family == "pls":
        regressor = make_pipeline(StandardScaler(), PLSRegression(scale=False, max_iter=1000, **params))
    elif family == "kernel_ridge":
        regressor = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", **params))
    else:
        raise ValueError(family)
    transformer = StandardScaler() if family == "kernel_ridge" else MinMaxScaler()
    return TransformedTargetRegressor(regressor=regressor, transformer=transformer)


def nested_fit(family, X, y, seed):
    """Receives only the fit subset: preprocessing and tuning cannot access held labels."""
    splits = list(KFold(3, shuffle=True, random_state=seed).split(X))
    scores = []
    for params in CONFIG["families"][family]["grid"]:
        prediction = np.full(len(y), np.nan)
        for inner, (fit, held) in enumerate(splits, 1):
            model = make_model(family, params, seed + inner)
            model.fit(X[fit], y[fit])
            prediction[held] = np.asarray(model.predict(X[held])).ravel()
        assert np.isfinite(prediction).all()
        scores.append(float(np.mean((prediction - y) ** 2)))
    best = int(np.argmin(scores))
    params = CONFIG["families"][family]["grid"][best]
    model = make_model(family, params, seed)
    model.fit(X, y)
    return model, {"selected_params": params, "inner_cv_mse": scores, "selected_index": best,
        "selection_labels": "fit subset only", "selection_fits": 3 * len(scores), "final_fit": 1}


def train(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    features = prepare_features()
    for seed in seeds:
        manifest, base = load_seed(seed)
        folder = OUT / f"seed{seed}"
        folder.mkdir(exist_ok=True)
        source_manifest = SOURCE / f"seed{seed}" / "manifest.json"
        source_prediction = SOURCE / f"seed{seed}" / "base_predictions.npz"
        save_json(folder / "source_audit.json", {"config_sha256": CONFIG_HASH,
            "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "source_prediction_sha256": hashlib.sha256(source_prediction.read_bytes()).hexdigest(),
            "existing_three_refitted": False})
        stages = [(f"fold{f['fold']}", seed + f["fold"], np.asarray(f["fit_indices"]), np.asarray(f["held_indices"])) for f in manifest["folds"]]
        stages.append(("final", seed + 1, base["train_indices"], base["test_indices"]))
        for stage_name, model_seed, fit, held in stages:
            assert not set(fit) & set(held)
            assert set(fit) <= set(base["train_indices"])
            stage = folder / stage_name
            stage.mkdir(exist_ok=True)
            y = labels(manifest, fit)
            for family in FAMILIES:
                complete = stage / f"{family}_complete.json"
                predicted = stage / f"{family}_predictions.npz"
                artifact = stage / f"{family}.joblib"
                if complete.exists():
                    audit = read_json(complete)
                    assert audit["config_sha256"] == CONFIG_HASH and predicted.exists() and artifact.exists()
                    continue
                started = time.monotonic()
                representation = CONFIG["families"][family]["representation"]
                model, audit = nested_fit(family, features[("clean", representation)][fit], y, model_seed)
                joblib.dump(model, artifact, compress=3)
                restored = joblib.load(artifact)
                predictions = {}
                for noise in NOISES:
                    X = features[(noise, representation)][held]
                    p = np.asarray(model.predict(X)).ravel()
                    assert len(p) == len(held) and np.isfinite(p).all()
                    np.testing.assert_allclose(np.asarray(restored.predict(X)).ravel(), p, rtol=1e-10, atol=1e-8)
                    predictions[noise] = p
                np.savez_compressed(predicted, indices=held, **predictions)
                save_json(complete, {"config_sha256": CONFIG_HASH, "family": family, "representation": representation,
                    "fit_indices": fit.tolist(), "held_indices": held.tolist(), "held_overlap": 0,
                    "outer_labels_used_for_selection": False, "noise_used_for_selection": False,
                    "reload_all_seven_verified": True, "model_seed": model_seed,
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "elapsed_seconds": time.monotonic() - started, **audit})
                print(f"[fit] seed{seed} {stage_name} {family}: {audit['selected_params']}, {time.monotonic()-started:.1f}s", flush=True)
        assemble(seed, manifest, base)
    save_json(OUT / "environment.json", {"sklearn": sklearn.__version__, "numpy": np.__version__,
        "config_sha256": CONFIG_HASH, "python": sys.version})


def assemble(seed, manifest, base):
    train_indices, test_indices = base["train_indices"], base["test_indices"]
    oof = np.full((7, len(train_indices), len(KEYS)), np.nan)
    outer = np.full((7, len(test_indices), len(KEYS)), np.nan)
    oof[:, :, :4], outer[:, :, :4] = base["oof"], base["outer"]
    positions = {int(i): j for j, i in enumerate(train_indices)}
    folder = OUT / f"seed{seed}"
    for j, family in enumerate(FAMILIES, 4):
        coverage = np.zeros(len(train_indices), int)
        for fold in manifest["folds"]:
            with np.load(folder / f"fold{fold['fold']}" / f"{family}_predictions.npz") as saved:
                np.testing.assert_array_equal(saved["indices"], fold["held_indices"])
                pos = [positions[int(i)] for i in saved["indices"]]
                coverage[pos] += 1
                for n, noise in enumerate(NOISES):
                    oof[n, pos, j] = saved[noise]
        assert np.all(coverage == 1)
        with np.load(folder / "final" / f"{family}_predictions.npz") as saved:
            np.testing.assert_array_equal(saved["indices"], test_indices)
            for n, noise in enumerate(NOISES):
                outer[n, :, j] = saved[noise]
    assert np.isfinite(oof).all() and np.isfinite(outer).all()
    np.testing.assert_array_equal(oof[:, :, :4], base["oof"])
    np.testing.assert_array_equal(outer[:, :, :4], base["outer"])
    np.savez_compressed(folder / "predictions.npz", oof=oof, outer=outer,
        train_indices=train_indices, test_indices=test_indices, keys=np.asarray(KEYS), noises=np.asarray(NOISES))


def constrained_weights(matrix, target, lower):
    """MSE simplex fit with explicit lower bounds retaining original-three block."""
    lower = np.asarray(lower, float)
    if matrix.ndim != 2 or matrix.shape[1] != len(lower) or np.any(lower < 0) or lower.sum() >= 1:
        raise ValueError("Invalid matrix or weight bounds")
    scale = max(float(np.var(target)), 1e-12)
    initial = lower + (1 - lower.sum()) / len(lower)
    result = minimize(lambda w: float(np.mean((matrix @ w - target) ** 2)) / scale,
        initial, jac=lambda w: 2 * matrix.T @ (matrix @ w - target) / len(target) / scale,
        method="SLSQP", bounds=[(float(v), 1.) for v in lower],
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(len(w))},
        options={"ftol": 1e-12, "maxiter": 1000})
    assert result.success, result.message
    assert abs(result.x.sum() - 1) < 1e-8 and np.all(result.x >= lower - 1e-8)
    return result.x


def expand_weights(core, added_indices, blocks):
    weights = np.zeros(len(KEYS))
    weights[:3] = np.asarray(core) * blocks[0]
    for idx, value in zip(added_indices, blocks[1:]):
        weights[idx] = value
    return weights


def fit_catalog(oof_clean, y, original_methods):
    """Only aligned clean OOF predictions and their training labels are accepted."""
    catalog = []
    candidates = []
    floor = CONFIG["integration"]["minimum_original_three_total_weight"]
    minimum_add = CONFIG["integration"]["minimum_each_added_model_weight_for_genuine_four_five"]

    def add(name, weights, anchor="none", kind="control", family="none"):
        p = oof_clean @ weights
        method = {"method": name, "weights": weights.tolist(), "anchor": anchor, "kind": kind,
            "family": family, "active_members": int(np.sum(weights > 1e-8)),
            "clean_oof_fit_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "preserves_original_three": bool(np.all(weights[:3] > 1e-8))}
        catalog.append(method)
        return method

    for key in KEYS:
        w = np.zeros(len(KEYS)); w[KEYS.index(key)] = 1
        add(key, w, kind="individual")
    add("existing3_equal", expand_weights([1/3]*3, [], [1]))
    add("hgb4_equal", expand_weights([1/3]*3, [3], [.75, .25]))
    for anchor in CONFIG["integration"]["anchors"]:
        core = np.asarray(original_methods[anchor]["weights"])
        assert np.all(core > 0) and abs(core.sum() - 1) < 1e-8
        baseline = oof_clean[:, :3] @ core
        add(anchor, expand_weights(core, [], [1]), anchor, "baseline")
        blocks = constrained_weights(np.column_stack([baseline, oof_clean[:, 3]]), y, [floor, minimum_add])
        add(f"{anchor}__hgb4_anchor", expand_weights(core, [3], blocks), anchor, "anchored4")
        add(f"{anchor}__hgb4_fixed25", expand_weights(core, [3], [.75, .25]), anchor, "fixed4")
        scored_fives = []
        for idx, family in enumerate(FAMILIES, 4):
            # Four-model controls ask whether the new candidate itself can supplement the core.
            four = constrained_weights(np.column_stack([baseline, oof_clean[:, idx]]), y, [floor, minimum_add])
            add(f"{anchor}__{family}4_anchor", expand_weights(core, [idx], four), anchor, "anchored4", family)
            five = constrained_weights(np.column_stack([baseline, oof_clean[:, 3], oof_clean[:, idx]]), y,
                [floor, minimum_add, minimum_add])
            selected = add(f"{anchor}__hgb_{family}5_anchor", expand_weights(core, [3, idx], five), anchor, "anchored5", family)
            scored_fives.append(selected)
            add(f"{anchor}__hgb_{family}5_fixed25", expand_weights(core, [3, idx], [.5, .25, .25]), anchor, "fixed5", family)
        best = min(scored_fives, key=lambda m: m["clean_oof_fit_rmse"])
        candidates.append({"anchor": anchor, "family": best["family"], "selected_method": best["method"],
            "selection_rmse": best["clean_oof_fit_rmse"], "selection_scope": "clean OOF integration fit diagnostic"})
        add(f"{anchor}__selected5", np.asarray(best["weights"]), anchor, "selected5", best["family"])
    for idx, family in enumerate(FAMILIES, 4):
        w = np.zeros(len(KEYS)); w[[0, 1, 2, 3, idx]] = .2
        add(f"hgb_{family}5_equal", w, kind="equal5", family=family)
    unrestricted = constrained_weights(oof_clean[:, :4], y, [0]*4)
    w = np.zeros(len(KEYS)); w[:4] = unrestricted
    add("hgb4_unrestricted_diagnostic", w, kind="unrestricted")
    return catalog, candidates


def integrate(seeds):
    for seed in seeds:
        manifest, _ = load_seed(seed)
        folder = OUT / f"seed{seed}"
        with np.load(folder / "predictions.npz") as saved:
            oof_clean = saved["oof"][0].copy()
            train_indices = saved["train_indices"].copy()
        methods = {m["method"]: m for m in read_json(SOURCE / f"seed{seed}" / "frozen_integration.json")["methods"]}
        catalog, selected = fit_catalog(oof_clean, labels(manifest, train_indices), methods)
        save_json(folder / "frozen_integration.json", {"config_sha256": CONFIG_HASH, "keys": KEYS,
            "outer_labels_used_for_selection": False, "noisy_oof_used_for_selection": False,
            "existing_three_predictions_fixed": True, "methods": catalog, "selected_fifth": selected})
        print(f"[integration] seed{seed}: {len(catalog)} methods; selected fifth {[s['family'] for s in selected]}", flush=True)


def metric_module():
    spec = importlib.util.spec_from_file_location("addition_metrics", ROOT / "experiments/2026-10-06_chunk_complementary_model_pilots/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(seeds):
    reference = metric_module()
    metrics, weights, paired, regions, wavs, predictions = [], [], [], [], [], []
    for seed in seeds:
        manifest, _ = load_seed(seed)
        folder = OUT / f"seed{seed}"
        frozen = read_json(folder / "frozen_integration.json")
        assert frozen["config_sha256"] == CONFIG_HASH
        with np.load(folder / "predictions.npz") as saved:
            outer, indices = saved["outer"].copy(), saved["test_indices"].copy()
        rows = [manifest["samples"][int(i)] for i in indices]
        y = labels(manifest, indices)
        days = np.asarray([r["source_wav_id"][:8] for r in rows])
        thresholds = np.asarray([THRESHOLDS[d] for d in days])
        wav_ids = np.asarray([r["source_wav_id"] for r in rows])
        method_predictions = np.stack([outer @ np.asarray(m["weights"]) for m in frozen["methods"]], axis=-1)
        np.savez_compressed(folder / "outer_integrated.npz", predictions=method_predictions,
            indices=indices, methods=np.asarray([m["method"] for m in frozen["methods"]]), noises=np.asarray(NOISES))
        lookup = {m["method"]: i for i, m in enumerate(frozen["methods"])}
        for j, method in enumerate(frozen["methods"]):
            for k, weight in zip(KEYS, method["weights"]):
                weights.append({"seed": seed, "method": method["method"], "model": k, "weight": weight,
                    "active_members": method["active_members"], "preserves_original_three": method["preserves_original_three"]})
            for n, noise in enumerate(NOISES):
                p = method_predictions[n, :, j]
                metrics.extend({"seed": seed, **r} for r in reference.record_summary(method["method"], "outer_unused_chunk", noise, y, p, days))
                if method["anchor"] == "none" or method["kind"] == "baseline":
                    continue
                baseline = method_predictions[n, :, lookup[method["anchor"]]]
                new_positive, old_positive, actual = p >= thresholds, baseline >= thresholds, y >= thresholds
                paired.append({"seed": seed, "method": method["method"], "anchor": method["anchor"], "noise": noise,
                    "delta_rmse": float(np.sqrt(np.mean((p-y)**2))-np.sqrt(np.mean((baseline-y)**2))),
                    "delta_mae": float(np.mean(np.abs(p-y))-np.mean(np.abs(baseline-y))),
                    "corrected_fn": int(np.sum(actual & ~old_positive & new_positive)),
                    "added_fn": int(np.sum(actual & old_positive & ~new_positive)),
                    "corrected_fp": int(np.sum(~actual & old_positive & ~new_positive)),
                    "added_fp": int(np.sum(~actual & ~old_positive & new_positive))})
                for name, mask in [("low_lt60", y < 60), ("pre_onb", y < thresholds),
                    ("near_onb", abs(y-thresholds) <= .1*thresholds), ("post_onb", y >= thresholds),
                    ("high_ge1.2onb", y >= 1.2*thresholds)]:
                    regions.append({"seed": seed, "method": method["method"], "anchor": method["anchor"], "noise": noise,
                        "region": name, "n": int(mask.sum()), "rmse": float(np.sqrt(np.mean((p[mask]-y[mask])**2))),
                        "baseline_rmse": float(np.sqrt(np.mean((baseline[mask]-y[mask])**2)))})
                for wav in np.unique(wav_ids):
                    mask = wav_ids == wav
                    wavs.append({"seed": seed, "method": method["method"], "anchor": method["anchor"], "noise": noise,
                        "source_wav_id": wav, "n": int(mask.sum()), "delta_mse": float(np.mean((p[mask]-y[mask])**2)-np.mean((baseline[mask]-y[mask])**2))})
        predictions.append({"seed": seed, "methods": len(frozen["methods"]), "outer_chunks": len(indices)})
    for filename, data in [("metrics.csv", metrics), ("weights.csv", weights), ("paired_deltas.csv", paired),
        ("region_errors.csv", regions), ("wav_errors.csv", wavs)]:
        save_csv(OUT / filename, data)
    save_json(OUT / "completed.json", {"config_sha256": CONFIG_HASH, "seeds": seeds,
        "evaluation": predictions, "metric_rows": len(metrics), "outer_labels_used_for_selection": False,
        "existing_three_refitted": False, "status": "nested candidates and fixed-core additions evaluated"})
    print(f"[evaluation] {len(metrics)} metric rows", flush=True)


def verify(seeds):
    from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
    from run_hgb_complementarity_validation import read_csv
    metrics = read_csv(OUT / "metrics.csv")
    checked, reloaded, fits = 0, 0, 0
    for seed in seeds:
        manifest, source = load_seed(seed)
        folder = OUT / f"seed{seed}"
        audit = read_json(folder / "source_audit.json")
        assert audit["source_prediction_sha256"] == hashlib.sha256((SOURCE / f"seed{seed}" / "base_predictions.npz").read_bytes()).hexdigest()
        with np.load(folder / "predictions.npz") as saved:
            np.testing.assert_array_equal(saved["oof"][:, :, :4], source["oof"])
            np.testing.assert_array_equal(saved["outer"][:, :, :4], source["outer"])
        frozen = read_json(folder / "frozen_integration.json")
        originals = {m["method"]: m for m in read_json(SOURCE / f"seed{seed}" / "frozen_integration.json")["methods"]}
        for method in frozen["methods"]:
            if method["anchor"] != "none":
                w = np.asarray(method["weights"])
                np.testing.assert_allclose(w[:3]/w[:3].sum(), originals[method["anchor"]]["weights"], atol=1e-10)
                assert np.all(w[:3] > 0) and w[:3].sum() >= .25-1e-8
            if method["kind"] in ["anchored4", "fixed4"]:
                assert method["active_members"] == 4
            if method["kind"] in ["anchored5", "fixed5", "selected5"]:
                assert method["active_members"] == 5
        for stage in ["fold1", "fold2", "fold3", "final"]:
            for family in FAMILIES:
                fit_audit = read_json(folder / stage / f"{family}_complete.json")
                assert fit_audit["config_sha256"] == CONFIG_HASH
                assert set(fit_audit["fit_indices"]).isdisjoint(fit_audit["held_indices"])
                assert set(fit_audit["fit_indices"]) <= set(source["train_indices"])
                assert fit_audit["reload_all_seven_verified"]
                assert fit_audit["artifact_sha256"] == hashlib.sha256((folder / stage / f"{family}.joblib").read_bytes()).hexdigest()
                fits += fit_audit["selection_fits"] + fit_audit["final_fit"]
                reloaded += 1
        with np.load(folder / "outer_integrated.npz") as saved:
            indices = saved["indices"]
            y = labels(manifest, indices)
            days = np.asarray([manifest["samples"][int(i)]["source_wav_id"][:8] for i in indices])
            for j, name in enumerate(saved["methods"]):
                for n, noise in enumerate(NOISES):
                    p = saved["predictions"][n, :, j]
                    for day in ["two_day", *THRESHOLDS]:
                        mask = np.ones(len(y), bool) if day == "two_day" else days == day
                        yt, pt = y[mask], p[mask]
                        threshold = np.asarray([THRESHOLDS[d] for d in days[mask]])
                        actual, estimated = yt >= threshold, pt >= threshold
                        near = abs(yt-threshold) <= .1*abs(threshold)
                        row = next(r for r in metrics if int(r["seed"]) == seed and r["method"] == name and r["noise"] == noise and r["day"] == day)
                        fp, fn = int(np.sum(~actual & estimated)), int(np.sum(actual & ~estimated))
                        tp = int(np.sum(actual & estimated))
                        checks = {"n": len(yt), "rmse": np.sqrt(np.mean((pt-yt)**2)), "mae": np.mean(abs(pt-yt)),
                            "r2": r2_score(yt, pt), "fp": fp, "fn": fn, "recall": tp/actual.sum(),
                            "fpr": fp/(~actual).sum(), "precision": tp/(tp+fp), "f1": 2*tp/(2*tp+fp+fn),
                            "rmse_pre": np.sqrt(np.mean((pt[~actual]-yt[~actual])**2)),
                            "rmse_post": np.sqrt(np.mean((pt[actual]-yt[actual])**2)),
                            "rmse_onb": np.sqrt(np.mean((pt[near]-yt[near])**2)),
                            "roc_auc": roc_auc_score(actual, pt), "pr_auc": average_precision_score(actual, pt)}
                        for key, value in checks.items():
                            np.testing.assert_allclose(float(row[key]), value, rtol=1e-10, atol=1e-8)
                        if day != "two_day":
                            reached = [v for v in np.unique(yt) if np.all(pt[yt>=v]>=THRESHOLDS[day])]
                            q = reached[0] if reached else np.nan
                            np.testing.assert_allclose(float(row["q100"]), q, atol=1e-8, equal_nan=True)
                            np.testing.assert_allclose(float(row["g100"]), q-THRESHOLDS[day], atol=1e-8, equal_nan=True)
                        checked += 1
    save_json(OUT / "verification.json", {"config_sha256": CONFIG_HASH, "verified_metric_rows": checked,
        "persisted_nested_models": reloaded, "total_candidate_fits_including_inner_selection": fits,
        "original_four_prediction_arrays_unchanged": True, "original_three_ratios_preserved": True,
        "genuine_four_five_member_counts_verified": True, "all_seven_reload_verified": True,
        "outer_labels_used_for_selection": False, "status": "passed"})
    print(f"[verify] {checked} metrics, {reloaded} models, {fits} fits passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "integrate", "evaluate", "verify"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    seeds = args.seeds or CONFIG["seeds"]
    if not set(seeds) <= set(CONFIG["seeds"]):
        raise ValueError("Use predefined seeds")
    with threadpool_limits(limits=CONFIG["cpu_threads"]):
        {"train": train, "integrate": integrate, "evaluate": evaluate, "verify": verify}[args.phase](seeds)
