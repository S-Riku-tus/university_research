"""Small nested training-only comparison of split, model and selection objective."""

import json

import joblib
import numpy as np
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from analyze import (CONFIG_PATH, OUTPUT, PREVIOUS, THRESHOLDS, endpoint, group_records,
                     rank_balanced_folds, read_csv, save_csv, score_for_selection,
                     summarize_predictions, training_rows)

FEATURES = ["log_power_512_1000", "log_power_1000_2000", "log_power_2000_3000",
    "log_power_2100_2500", "log_power_total", "log_ratio_2000_3000_to_1000_2000",
    "log_ratio_2100_2500_to_1000_2000", "spectral_centroid_hz",
    "normalized_spectral_entropy", "temporal_power_cv"]


def make_model(family, parameter, seed, config):
    params = config["model_families"][family].copy()
    grid_key = {"SVR": "C", "ExtraTrees": "min_samples_leaf", "HistGradientBoosting": "max_leaf_nodes"}[family]
    params[grid_key] = parameter
    if family == "SVR":
        regressor = make_pipeline(StandardScaler(), SVR(**params))
    elif family == "ExtraTrees":
        regressor = ExtraTreesRegressor(**params, random_state=seed)
    else:
        regressor = HistGradientBoostingRegressor(**params, random_state=seed)
    return TransformedTargetRegressor(regressor=regressor, transformer=MinMaxScaler())


def nested_splits(fit, rows, groups, strategy, seed):
    if strategy == "original_random":
        unique = np.unique(groups[fit])
        return [(fit[np.isin(groups[fit], unique[a])], fit[np.isin(groups[fit], unique[b])])
                for a, b in KFold(n_splits=3, shuffle=True, random_state=seed).split(unique)]
    mapping, _ = rank_balanced_folds(group_records([rows[i] for i in fit]), seed=seed)
    return [(fit[np.asarray([mapping[groups[i]] != fold for i in fit])],
             fit[np.asarray([mapping[groups[i]] == fold for i in fit])]) for fold in [1, 2, 3]]


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = training_rows()
    lookup = {(r["noise"], r["source_wav_id"], r["chunk_index"]): r
              for r in read_csv(PREVIOUS / "training_input_features.csv")}
    y = np.asarray([float(r["y_kW_m2"]) for r in rows])
    groups = np.asarray([r["source_wav_id"] for r in rows])
    days = np.asarray([g[:8] for g in groups])
    thresholds = np.asarray([THRESHOLDS[d] for d in days])
    X = {noise: np.asarray([[float(lookup[(noise, r["source_wav_id"], r["chunk_index"])][f])
                           for f in FEATURES] for r in rows]) for noise in ["clean", "-20"]}
    assert all(np.isfinite(x).all() for x in X.values())
    assert len(groups) == 1620 and len(set(groups)) == 36
    records = group_records(rows)
    strategies = {"original_random": {r["source_wav_id"]: int(r["fold"]) for r in rows},
                  "rank_balanced": rank_balanced_folds(records)[0]}
    selections, audit, predictions = [], [], []
    endpoint_rows, stage_rows, inner_support = [], [], []
    model_dir = OUTPUT / "pilot_models"
    model_dir.mkdir(exist_ok=True)
    fits = 0
    grid_keys = {"SVR": "C", "ExtraTrees": "min_samples_leaf", "HistGradientBoosting": "max_leaf_nodes"}
    objectives = config["nested_selection"]["policies"]
    for strategy, mapping in strategies.items():
        fold_ids = np.asarray([mapping[g] for g in groups])
        for family, family_config in config["model_families"].items():
            pred = {(objective, noise): np.full(len(rows), np.nan) for objective in objectives for noise in X}
            for fold in [1, 2, 3]:
                fit, held = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
                held_groups = set(groups[held])
                assert set(groups[fit]).isdisjoint(held_groups)
                splits = nested_splits(fit, rows, groups, strategy, 42+fold)
                support = []
                for inner_fold, (inner_fit, inner_held) in enumerate(splits, 1):
                    assert set(groups[inner_fit]).isdisjoint(groups[inner_held])
                    assert not (set(groups[inner_fit]) | set(groups[inner_held])) & held_groups
                    near_groups = {groups[i] for i in inner_fit if .9 <= y[i]/thresholds[i] <= 1.1}
                    support.append({"strategy": strategy, "family": family, "outer_oof_fold": fold,
                        "inner_fold": inner_fold, "fit_wavs": len(set(groups[inner_fit])),
                        "held_wavs": len(set(groups[inner_held])), "fit_near_onb_wavs": len(near_groups)})
                inner_support.extend(support)
                candidates = []
                for parameter in family_config[grid_keys[family]]:
                    inner_pred = np.full(len(rows), np.nan)
                    coverage = np.zeros(len(rows), int)
                    for inner_fit, inner_held in splits:
                        model = make_model(family, parameter, 42+fold, config)
                        model.fit(X["clean"][inner_fit], y[inner_fit])
                        fits += 1
                        inner_pred[inner_held] = model.predict(X["clean"][inner_held])
                        coverage[inner_held] += 1
                    assert np.all(coverage[fit] == 1) and np.all(coverage[held] == 0)
                    assert np.isfinite(inner_pred[fit]).all()
                    scores = {objective: score_for_selection(y[fit], inner_pred[fit], days[fit], objective)
                              for objective in objectives}
                    candidates.append((parameter, scores))
                    selections.append({"strategy": strategy, "family": family, "outer_oof_fold": fold,
                        "parameter": parameter, "inner_rmse_kW_m2": scores["rmse"][0],
                        "inner_fp": scores["fp_then_q100"][0],
                        "inner_worst_g100_kW_m2": scores["fp_then_q100"][1],
                        "inner_mean_g100_kW_m2": scores["fp_then_q100"][2],
                        "held_outer_wavs_used": False, "noise_used_for_selection": False})
                selected = {objective: min(candidates, key=lambda c: (c[1][objective], c[0]))[0]
                            for objective in objectives}
                # Reuse one fitted artifact when both objectives select the same parameter.
                for parameter in sorted(set(selected.values())):
                    model = make_model(family, parameter, 42+fold, config)
                    model.fit(X["clean"][fit], y[fit])
                    fits += 1
                    result = {noise: model.predict(x[held]) for noise, x in X.items()}
                    artifact = model_dir / f"{strategy}_{family}_fold{fold}_p{parameter}.joblib"
                    joblib.dump(model, artifact, compress=3)
                    reloaded = joblib.load(artifact)
                    assert np.allclose(reloaded.predict(X["clean"][held[:5]]), result["clean"][:5],
                                       rtol=1e-12, atol=1e-10)
                    for objective in objectives:
                        if selected[objective] == parameter:
                            for noise in X:
                                pred[(objective, noise)][held] = result[noise]
                            audit.append({"strategy": strategy, "family": family, "objective": objective,
                                "fold": fold, "selected_parameter": parameter, "fit_wavs": sorted(set(groups[fit])),
                                "held_wavs": sorted(held_groups), "shared_wavs": 0,
                                "fit_chunks": len(fit), "held_chunks": len(held),
                                "artifact": artifact.relative_to(OUTPUT).as_posix(), "reload_verified": True})
                print(f"{strategy} {family} fold {fold}: selections {selected}", flush=True)
            assert all(np.isfinite(p).all() for p in pred.values())
            for (objective, noise), p in pred.items():
                endpoint_rows.extend(summarize_predictions(family, noise, rows, p,
                                     strategy=strategy, objective=objective, policy="clean_only"))
                for i, row in enumerate(rows):
                    predictions.append({"strategy": strategy, "family": family, "objective": objective,
                        "condition": noise, "source_wav_id": groups[i], "chunk_index": row["chunk_index"],
                        "fold": int(fold_ids[i]), "heat_flux_kW_m2": y[i], "onb_kW_m2": thresholds[i],
                        "prediction_kW_m2": p[i]})
                for group, rec in records.items():
                    mask = groups == group
                    stage_rows.append({"strategy": strategy, "family": family, "objective": objective,
                        "condition": noise, "source_wav_id": group, "day": rec["day"],
                        "heat_flux_kW_m2": rec["heat_flux_kW_m2"], "n": int(mask.sum()),
                        "n_positive": int((p[mask] >= rec["onb_kW_m2"]).sum()),
                        "minimum_prediction_margin_kW_m2": float(p[mask].min()-rec["onb_kW_m2"])})
    save_csv("candidate_inner_selection.csv", selections)
    save_csv("candidate_training_oof_predictions.csv", predictions)
    save_csv("candidate_endpoints.csv", endpoint_rows)
    save_csv("candidate_stage_profiles.csv", stage_rows)
    save_csv("nested_support_audit.csv", inner_support)
    (OUTPUT / "candidate_audit.json").write_text(json.dumps({
        "sklearn_version": sklearn.__version__, "fit_operations": fits, "features": FEATURES,
        "training_chunks": len(rows), "source_wavs": len(records), "outer_test_used": False,
        "noise_used_for_hyperparameter_selection": False, "main_models_retrained": False,
        "split_and_objective_not_adopted": True, "models": audit,
        "status": "Exploratory shared-feature pilots; no main-model split sensitivity or outer-test adoption claim"
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for row in endpoint_rows:
        if row["condition"] == "clean":
            print(row["strategy"], row["objective"], row["method"], row["day"],
                  "q100", round(row["q100_kW_m2"], 2), "FP", row["fp"], flush=True)
    print("Fit operations:", fits)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
