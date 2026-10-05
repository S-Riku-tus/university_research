"""Nested, source-WAV-disjoint training-side pilot for one explicit-feature SVR."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from analyze import CONFIG as DIAGNOSTIC_CONFIG, FEATURES, MODELS, OUTPUT, ROOT, metrics, read_csv, save_csv


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/experiments/2026-10-04_band_feature_svr_training_pilot.json")
CONFIG_PATH = ROOT / parser.parse_args().config
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PREFIX = CONFIG.get("output_prefix", "svr")


def save_pilot_csv(name, rows):
    save_csv(name.replace("svr_", PREFIX + "_", 1), rows)


def make_model(c):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(C=c, epsilon=CONFIG["epsilon"],
                                                     gamma=CONFIG["gamma"], kernel="rbf")),
        transformer=MinMaxScaler())


def main():
    rows = [r for r in read_csv(OUTPUT / "oof_sample_diagnostics.csv") if r["noise"] == "clean"]
    feature_lookup = {(r["noise"], r["source_wav_id"], int(r["chunk_index"])): r
                      for r in read_csv(OUTPUT / "training_input_features.csv")}
    y = np.asarray([float(r["y_kW_m2"]) for r in rows])
    t = np.asarray([float(r["onb_kW_m2"]) for r in rows])
    groups = np.asarray([r["source_wav_id"] for r in rows])
    fold_ids = np.asarray([int(r["fold"]) for r in rows])
    regions = np.asarray([r["region"] for r in rows])
    existing = np.column_stack([[float(r[m]) for r in rows] for m in MODELS])
    old = np.asarray([float(r["performance_oof_diagnostic"]) for r in rows])
    X = {noise: np.asarray([[float(feature_lookup[(noise, r["source_wav_id"], int(r["chunk_index"]))][k])
                             for k in CONFIG["features"]] for r in rows])
         for noise in ["clean", *CONFIG["stress_prediction_conditions"]]}
    assert len(y) == 1620 and np.isfinite(X["clean"]).all()
    assert set(CONFIG["features"]) <= set(FEATURES) and len(set(groups)) == 36
    assert not set(CONFIG["features"]) & {"heat_flux", "source_day", "source_wav_id", "chunk_index"}
    predictions = {noise: np.full(len(y), np.nan) for noise in X}
    tune_rows, fold_records = [], []
    model_dir = OUTPUT / f"{PREFIX}_pilot_models"
    model_dir.mkdir(exist_ok=True)
    for fold_id in [1, 2, 3]:
        fit, held = np.flatnonzero(fold_ids != fold_id), np.flatnonzero(fold_ids == fold_id)
        fit_groups, held_groups = set(groups[fit]), set(groups[held])
        assert fit_groups.isdisjoint(held_groups)
        unique_groups = np.unique(groups[fit])
        inner_splits = list(KFold(n_splits=3, shuffle=True, random_state=42+fold_id).split(unique_groups))
        candidate_scores = []
        for c in CONFIG["C_candidates"]:
            inner_pred = np.full(len(y), np.nan)
            for inner_id, (ig_fit, ig_held) in enumerate(inner_splits, 1):
                inner_fit = fit[np.isin(groups[fit], unique_groups[ig_fit])]
                inner_held = fit[np.isin(groups[fit], unique_groups[ig_held])]
                assert set(groups[inner_fit]).isdisjoint(groups[inner_held])
                assert not set(groups[inner_fit]) & held_groups
                model = make_model(c)
                model.fit(X["clean"][inner_fit], y[inner_fit])
                inner_pred[inner_held] = model.predict(X["clean"][inner_held])
            assert np.isfinite(inner_pred[fit]).all()
            score = float(np.sqrt(np.mean((inner_pred[fit] - y[fit])**2)))
            candidate_scores.append((score, c))
            tune_rows.append({"outer_oof_fold": fold_id, "C": c, "pooled_inner_rmse_kW_m2": score,
                              "inner_wav_folds": 3, "n_inner_oof": len(fit), "held_outer_wavs_used": False})
        selected_c = min(candidate_scores)[1]
        model = make_model(selected_c)
        model.fit(X["clean"][fit], y[fit])
        for noise in X:
            predictions[noise][held] = model.predict(X[noise][held])
        artifact = model_dir / f"svr_oof_fold{fold_id}.joblib"
        joblib.dump(model, artifact)
        reloaded = joblib.load(artifact)
        assert np.allclose(reloaded.predict(X["clean"][held[:5]]), predictions["clean"][held[:5]], rtol=1e-12, atol=1e-10)
        fold_records.append({"fold": fold_id, "selected_C": selected_c, "fit_chunks": len(fit),
            "held_chunks": len(held), "fit_wavs": sorted(fit_groups), "held_wavs": sorted(held_groups),
            "fit_held_shared_wavs": 0, "artifact": artifact.relative_to(OUTPUT).as_posix(),
            "reload_prediction_verified": True})
        print(f"{PREFIX} fold {fold_id}: C={selected_c:g}, clean held RMSE={np.sqrt(np.mean((predictions['clean'][held]-y[held])**2)):.2f}", flush=True)
    assert all(np.isfinite(p).all() for p in predictions.values())
    save_pilot_csv("svr_inner_parameter_selection.csv", tune_rows)
    out_rows, metric_rows, correction_rows, pair_rows, wav_rows = [], [], [], [], []
    clean_svr = predictions["clean"]
    methods = {"SVR": clean_svr, "existing3_performance_oof_diagnostic": old,
               "fixed_4_equal": (existing.sum(axis=1)+clean_svr)/4,
               "fixed_old75_SVR25": .75*old+.25*clean_svr}
    scopes = {"all": np.ones(len(y), bool), **{r: regions == r for r in np.unique(regions)},
              **{f"day_{d}": np.asarray([g.startswith(d) for g in groups]) for d in ["20250611","20250618"]},
              **{f"fold_{f}": fold_ids == f for f in [1,2,3]}}
    common_fn = (y >= t) & (existing.max(axis=1) < t)
    for scope_name, mask in scopes.items():
        for name, pred in methods.items():
            metric_rows.append({"condition": "clean", "scope": scope_name, "method": name,
                                **metrics(y[mask], pred[mask], t[mask])})
        metric_rows.append({"condition": "-20_clean_fitted_transfer", "scope": scope_name, "method": "SVR",
                            **metrics(y[mask], predictions["-20"][mask], t[mask])})
        truth = y >= t
        old_ok = (old >= t) == truth
        for name, pred in methods.items():
            new_ok = (pred >= t) == truth
            correction_rows.append({"scope": scope_name, "method": name, "n": int(mask.sum()),
                "baseline": "existing3_performance_oof_diagnostic",
                "binary_errors_corrected": int((mask & ~old_ok & new_ok).sum()),
                "binary_errors_added": int((mask & old_ok & ~new_ok).sum()),
                "common_fn_corrected": int((mask & common_fn & (pred >= t)).sum()),
                "common_fn_total": int((mask & common_fn).sum()),
                "pre_fp_added": int((mask & ~truth & old_ok & ~new_ok).sum()),
                "post_fn_added": int((mask & truth & old_ok & ~new_ok).sum())})
        for j, m in enumerate(MODELS):
            ea, eb = existing[mask,j]-y[mask], clean_svr[mask]-y[mask]
            pair_rows.append({"scope": scope_name, "model_a": m, "model_b": "band_feature_SVR",
                "n": int(mask.sum()), "mean_residual_product": float(np.mean(ea*eb)),
                "residual_correlation": float(np.corrcoef(ea,eb)[0,1]),
                "opposite_sign_n": int((ea*eb<0).sum())})
    for i,r in enumerate(rows):
        out_rows.append({"source_wav_id": r["source_wav_id"], "chunk_index": int(r["chunk_index"]),
            "fold": int(fold_ids[i]), "heat_flux_kW_m2": float(y[i]), "onb_kW_m2": float(t[i]),
            "region": r["region"], "SVR_clean": float(clean_svr[i]),
            "SVR_minus20_clean_fitted_transfer": float(predictions["-20"][i]),
            **{name: float(pred[i]) for name,pred in methods.items() if name != "SVR"},
            "existing_common_fn": bool(common_fn[i]), "SVR_corrects_common_fn": bool(common_fn[i] and clean_svr[i]>=t[i])})
    for wav in sorted(set(groups)):
        mask = groups == wav
        wav_rows.append({"source_wav_id": wav, "fold": int(fold_ids[mask][0]),
            "heat_flux_kW_m2": float(y[mask][0]), "onb_kW_m2": float(t[mask][0]), "region": regions[mask][0],
            "n": int(mask.sum()), "common_fn": int(common_fn[mask].sum()),
            "SVR_common_fn_corrected": int((common_fn[mask]&(clean_svr[mask]>=t[mask])).sum()),
            "SVR_mean": float(clean_svr[mask].mean()), "SVR_bias": float((clean_svr[mask]-y[mask]).mean()),
            "SVR_minus20_mean": float(predictions["-20"][mask].mean()),
            **{f"{name}_fn": metrics(y[mask],pred[mask],t[mask])["fn"] for name,pred in methods.items()}})
    for name,data in [("svr_training_oof_predictions.csv",out_rows), ("svr_training_oof_metrics.csv",metric_rows),
                      ("svr_error_corrections.csv",correction_rows), ("svr_residual_pairs.csv",pair_rows),
                      ("svr_wav_profiles.csv",wav_rows)]:
        save_pilot_csv(name,data)
    (OUTPUT / f"{PREFIX}_pilot_audit.json").write_text(json.dumps({
        "config": CONFIG_PATH.relative_to(ROOT).as_posix(), "training_chunk_count": len(y),
        "input_features": CONFIG["features"],
        "source_wav_count": 36, "source_wav_disjoint_folds": fold_records,
        "fit_operations": 30, "new_weights_fitted": False, "outer_test_used": False,
        "noisy_data_used_in_hyperparameter_selection": False,
        "noisy_results_used_in_feature_design": CONFIG.get("noisy_results_used_in_feature_design", False),
        "status": "Exploratory training-side pilot; not adopted; not a held-out outer comparison",
        "limitations": CONFIG["limitations"]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for r in metric_rows:
        if r["scope"] in ["all","ONB_to_1.5ONB"]:
            print(r)
    for r in correction_rows:
        if r["scope"] == "all":print(r)


if __name__ == "__main__":
    main()
