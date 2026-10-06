"""Describe input shift and saved HGB sensitivity; no model selection or refit."""
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_hgb_complementarity_validation as run
from utils.dataloading.acoustic_summary_features import AcousticFrequency34


def main():
    feature = AcousticFrequency34()
    names = feature.get_feature_names_out()
    features = {n: np.load(run.OUT / "cache" / f"features_{n}.npz")["X"] for n in run.NOISES}
    representatives = run.read_csv(run.OUT / "representative_chunks.csv")
    shift = []; dependency = []
    for seed in run.CONFIG["replication_seeds"]:
        folder = run.OUT / f"seed{seed}"
        manifest = run.read_json(folder / "manifest.json")
        samples = manifest["samples"]
        train = np.asarray(manifest["train_indices"]); test = np.asarray(manifest["test_indices"])
        index = {run.identity(r): i for i, r in enumerate(samples)}
        model = joblib.load(folder / "final/hgb.joblib")
        scaler = joblib.load(folder / "final/target_scaler.joblib")
        predict = lambda f: scaler.inverse_transform(model.predict(f).reshape(-1, 1)).ravel() / 1000
        clean = features["clean"][train]
        low, high = clean.min(axis=0), clean.max(axis=0)
        q01, q99 = np.quantile(clean, [.01, .99], axis=0)
        y = np.asarray([float(samples[i]["sample_filename"].split("_")[0]) / 1000 for i in test])
        t = np.asarray([run.THRESHOLDS[samples[i]["source_wav_id"][:8]] for i in test])
        with np.load(folder / "base_predictions.npz") as saved:
            final = saved["outer"][:, :, 3]
        for noise_idx, noise in enumerate(run.NOISES):
            x = features[noise][test]; p = predict(x)
            np.testing.assert_allclose(p, final[noise_idx], rtol=1e-12, atol=1e-10)
            scopes = {"all_below_onb": y < t, "below_onb_false_positive": (y < t) & (p >= t),
                "below_onb_true_negative": (y < t) & (p < t), "onb_and_above": y >= t}
            for scope, mask in scopes.items():
                if not mask.any():
                    continue
                for j, name in enumerate(names):
                    values = x[mask, j]
                    shift.append({"seed": seed, "noise": noise, "scope": scope, "feature": name, "chunks": int(mask.sum()),
                        "clean_fit_min": float(low[j]), "clean_fit_max": float(high[j]), "clean_fit_q01": float(q01[j]), "clean_fit_q99": float(q99[j]),
                        "evaluation_mean": float(values.mean()), "outside_clean_fit_range": int(((values < low[j]) | (values > high[j])).sum()),
                        "outside_clean_fit_q01_q99": int(((values < q01[j]) | (values > q99[j])).sum())})
        for case, row in enumerate(representatives):
            i = index[run.identity(row)]
            raw = np.load(run.OUT / "cache" / f"raw_{row['noise']}.npy", mmap_mode="r")[i:i+1].copy()
            base = float(predict(feature.transform(raw))[0])
            changes = {"unchanged": raw}
            for start, end in [(1700, 2100), (2100, 2500), (2500, 2900)]:
                changed = raw.copy(); a, b = round(224 * start / 3000), round(224 * end / 3000)
                changed[:, :, a:b, :] *= .1
                changes[f"attenuate_{start}_{end}_power_x0p1"] = changed
            changes["attenuate_all_power_x0p1"] = raw * .1
            changes["time_shuffle"] = raw[:, np.random.default_rng(42).permutation(224), :, :]
            for operation, changed in changes.items():
                p = float(predict(feature.transform(changed))[0])
                dependency.append({"seed": seed, "case": case, "scope": row["scope"], "noise": row["noise"], "source_wav_id": row["source_wav_id"],
                    "chunk_index": row["chunk_index"], "used_in_this_seed_final_fit": bool(i in train), "operation": operation,
                    "unchanged_prediction": base, "changed_prediction": p, "prediction_change": p - base})
                if operation == "time_shuffle":
                    np.testing.assert_allclose(p, base, atol=1e-10, rtol=0)
    run.save_csv(run.OUT / "hgb_replication_input_shift.csv", shift)
    run.save_csv(run.OUT / "hgb_replication_dependency.csv", dependency)
    run.save_json(run.OUT / "hgb_replication_diagnostic_definition.json", {
        "input_shift": "Final saved HGB, each seed's own 540 unused chunks compared with its clean fit feature range; no noisy refit",
        "representative_dependency": "Same seed42-selected cases under seed43/44 HGB models; some cases were used for these models' fits and are flagged",
        "interpretation": "Descriptive range and input dependence; not independent performance, physical causality or proof of the reason for split sensitivity",
        "model_selection": False, "representative_rows": len(dependency), "feature_range_rows": len(shift)})
    print("HGB replication diagnostics:", len(shift), "feature-range rows and", len(dependency), "intervention rows")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
