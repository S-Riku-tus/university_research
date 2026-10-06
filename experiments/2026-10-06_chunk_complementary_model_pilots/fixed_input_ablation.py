"""Fixed max_leaf_nodes=15 contrast; does not alter frozen integration."""
import hashlib

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from pilot import (CONFIG, NOISE_DIRS, OUT, ROOT, candidates, endpoints,
                   load_features, make_model, read_json, record_summary,
                   representation, save_csv, save_json, setup)


def main():
    config = read_json(ROOT / "configs/experiments/2026-10-06_chunk_fixed_input_ablation.json")
    rows, folds, data_root, outer, base, y, days = setup()
    frozen = (OUT / "frozen_integration.json").read_bytes()
    X = load_features(rows, data_root, "clean", "train")[0]
    parameter = config["fixed_parameter"]["max_leaf_nodes"]
    summary, audit, fits = [], [], 0
    cv_predictions = {}
    final_models = {}
    for rep in config["representations"]:
        x = representation(X, rep)
        p = np.full(len(y), np.nan)
        for fold, (fit, held) in enumerate(folds, 1):
            model = make_model("HGB", parameter, 42+fold)
            model.fit(x[fit], y[fit]); fits += 1
            p[held] = model.predict(x[held])
            artifact = OUT / f"fixed15_{rep}_fold{fold}.joblib"
            joblib.dump(model, artifact, compress=3)
            np.testing.assert_allclose(joblib.load(artifact).predict(x[held]), p[held], atol=1e-10, rtol=1e-12)
        assert np.isfinite(p).all()
        summary.extend(record_summary(f"fixed15_{rep}", "training_fixed_oof", "clean", y, p, days))
        cv_predictions[rep] = p
        prior = next((item for item in candidates() if item["representation"] == rep and item["family"] == "HGB" and item["final_parameter"] == parameter), None)
        if prior:
            artifact = OUT / prior["final_artifact"]
            final_models[rep] = joblib.load(artifact)
        else:
            model = make_model("HGB", parameter, 42)
            model.fit(x, y); fits += 1
            artifact = OUT / f"fixed15_{rep}_final.joblib"
            joblib.dump(model, artifact, compress=3)
            final_models[rep] = joblib.load(artifact)
            np.testing.assert_allclose(final_models[rep].predict(x[:8]), model.predict(x[:8]), rtol=1e-12, atol=1e-10)
        audit.append({"representation": rep, "parameter": parameter, "final_model": artifact.name, "reused_final_model": prior is not None, "reload_verified": True})
    np.savez_compressed(OUT / "fixed_input_ablation_oof.npz", **cv_predictions)
    predictions = []
    # Labels here are used exclusively to score the already-fixed models.
    from pilot import RUN, read_csv, identity
    for noise, dirname in NOISE_DIRS.items():
        suffix = "no_noise" if noise == "clean" else noise
        lookup = {identity(r): r for r in read_csv(RUN / "maxfreq=3kHz" / dirname / "fold_pred" / f"pred_f1_{suffix}.csv")}
        ordered = [lookup[identity(r)] for r in outer]
        yo = np.asarray([float(r["y_true"])/1000 for r in ordered])
        do = np.asarray([r["source_wav_id"][:8] for r in outer])
        xfull = load_features(outer, data_root, noise, "outer")[0]
        for rep, model in final_models.items():
            p = model.predict(representation(xfull, rep))
            summary.extend(record_summary(f"fixed15_{rep}", "outer_unused_chunk", noise, yo, p, do))
            for i, row in enumerate(outer):
                predictions.append({"method": f"fixed15_{rep}", "noise": noise, "source_wav_id": row["source_wav_id"], "chunk_index": int(row["chunk_index"]), "y_kW_m2": yo[i], "prediction_kW_m2": p[i]})
    assert (OUT / "frozen_integration.json").read_bytes() == frozen
    save_csv("fixed_input_ablation_metrics.csv", summary)
    save_csv("fixed_input_ablation_predictions.csv", predictions)
    save_json("fixed_input_ablation_audit.json", {"fit_operations": fits, "max_leaf_nodes": parameter, "outer_labels_used_for_selection": False, "timing": config["timing"], "models": audit, "frozen_integration_sha256": hashlib.sha256(frozen).hexdigest()})
    print("Fixed input ablation complete; fit operations:", fits)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
