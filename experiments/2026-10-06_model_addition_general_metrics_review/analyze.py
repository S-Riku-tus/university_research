"""General-metric benefit and matched-algorithm controls from saved predictions."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / "experiments/2026-10-06_chunk_complementary_model_pilots"
sys.path.insert(0, str(PRIOR))
import pilot as p

CONFIG_PATH = ROOT / "configs/experiments/2026-10-06_model_addition_general_metrics_review.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
HGB = "frequency34_HGB_rmse"
FOUR = HGB + "__mse_simplex"
DIVERSE = HGB + "__diversity_simplex"
AFFINE = HGB + "__affine_ridge"
ORIGINAL = [*p.KEYS, "baseline_performance", "baseline_equal", "conformer_alexnet_equal", "band10_HGB_rmse", HGB, "temporal46_HGB_rmse", HGB+"__equal4", HGB+"__inverse_mse4", FOUR, DIVERSE, AFFINE]


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def save_csv(name, rows):
    assert rows
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simplex(matrix, y, penalty=0):
    n = matrix.shape[1]
    residual = matrix-y[:, None]
    corr = np.maximum(np.corrcoef(residual, rowvar=False), 0)
    np.fill_diagonal(corr, 0)
    norm = float(np.mean((p.WEIGHTS @ matrix[:, :3].T-y)**2)) if n >= 3 else float(np.var(y))
    def loss(w):
        return (float(np.mean((matrix@w-y)**2)) + penalty*float(w@corr@w))/norm
    def grad(w):
        return (2*matrix.T@(matrix@w-y)/len(y) + 2*penalty*corr@w)/norm
    result = minimize(loss, np.full(n, 1/n), jac=grad, method="SLSQP", bounds=[(0, 1)]*n,
        constraints={"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(n)},
        options={"ftol": 1e-12, "maxiter": 1000})
    assert result.success, result.message
    weights = np.maximum(result.x, 0); weights /= weights.sum()
    # MSE is convex; check KKT directional derivatives for all feasible exchanges.
    if penalty == 0:
        gradient = 2*matrix.T@(matrix@weights-y)/len(y)
        active = weights > 1e-7
        assert np.ptp(gradient[active]) < .02
        assert np.min(gradient[~active], initial=float("inf")) >= gradient[active].mean()-.02
    return weights


def fit_controls(base, hgb, y):
    matrix = np.column_stack([base, hgb])
    names = [*p.KEYS, HGB]
    penalty = .25*float(np.mean((base@p.WEIGHTS-y)**2))
    catalog = []
    for label, indices in [
        ("existing3_mse", [0, 1, 2]), ("existing_ca_mse", [1, 2]),
        ("conformer_hgb_mse", [1, 3]), ("alexnet_hgb_mse", [2, 3]),
        ("ca_hgb_mse", [1, 2, 3])]:
        weights = simplex(matrix[:, indices], y)
        catalog.append({"method": label, "inputs": [names[i] for i in indices], "weights": weights.tolist(), "kind": "linear"})
    weights = simplex(base, y, penalty)
    catalog.append({"method": "existing3_diversity", "inputs": list(p.KEYS), "weights": weights.tolist(), "kind": "linear", "penalty": penalty})
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(base, y)
    artifact = OUT / "existing3_affine_ridge.joblib"
    joblib.dump(model, artifact, compress=3)
    np.testing.assert_allclose(model.predict(base), joblib.load(artifact).predict(base), rtol=1e-12, atol=1e-10)
    catalog.append({"method": "existing3_affine", "inputs": list(p.KEYS), "artifact": artifact.name, "kind": "affine"})
    save_json("frozen_controls.json", {"fit_scope": "clean training OOF only", "outer_labels_used_for_fit": False, "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(), "methods": catalog, "timing": "Follow-up diagnostic after the preceding developmental pilot; not pristine confirmation"})
    return catalog


def apply_control(control, pool):
    matrix = np.column_stack([pool[k] for k in control["inputs"]])
    return matrix@control["weights"] if control["kind"] == "linear" else joblib.load(OUT/control["artifact"]).predict(matrix)


def regions(y, t):
    return {"below60": y < 60, "60_to_ONB": (y >= 60) & (y < t), "ONB_to_1.5ONB": (y >= t) & (y < 1.5*t),
        "above_1.5ONB": y >= 1.5*t, "near_ONB_0.9_to_1.1": np.abs(y-t) <= .1*t}


def paired_bootstrap(outer, y, predictions, noise):
    groups = np.asarray([r["source_wav_id"] for r in outer])
    wavs = sorted(set(groups))
    cells = [np.flatnonzero(groups == g) for g in wavs]
    day_wavs = [np.asarray([j for j, g in enumerate(wavs) if g.startswith(day)]) for day in p.THRESHOLDS]
    rng = np.random.default_rng(CONFIG["bootstrap"]["seed"])
    sampled = np.column_stack([rng.choice(ids, size=(CONFIG["bootstrap"]["repeats"], len(ids)), replace=True) for ids in day_wavs])
    t = np.asarray([p.THRESHOLDS[g[:8]] for g in groups])
    n = np.asarray([len(idx) for idx in cells])
    stats = {}
    for method, prediction in predictions.items():
        stats[method] = {"sse": np.asarray([np.sum((prediction[idx]-y[idx])**2) for idx in cells]),
                         "sae": np.asarray([np.abs(prediction[idx]-y[idx]).sum() for idx in cells]),
                         "fn": np.asarray([((y[idx] >= t[idx]) & (prediction[idx] < t[idx])).sum() for idx in cells])}
    contrasts = [(FOUR, "baseline_performance"), (FOUR, "existing3_mse"), (FOUR, HGB), (DIVERSE, FOUR), (AFFINE, "existing3_affine")]
    result = []
    for candidate, reference in contrasts:
        for metric in ["rmse", "mae", "fn"]:
            def values(method, selections):
                s = stats[method]
                if metric == "rmse":
                    return np.sqrt(s["sse"][selections].sum(axis=-1)/n[selections].sum(axis=-1))
                if metric == "mae":
                    return s["sae"][selections].sum(axis=-1)/n[selections].sum(axis=-1)
                return s["fn"][selections].sum(axis=-1)
            delta = values(candidate, sampled)-values(reference, sampled)
            point = values(candidate, np.arange(len(wavs)))-values(reference, np.arange(len(wavs)))
            low, high = np.quantile(delta, [.025, .975])
            result.append({"noise": noise, "candidate": candidate, "reference": reference, "metric": metric,
                           "candidate_minus_reference": float(point), "bootstrap_2.5pct": float(low), "bootstrap_97.5pct": float(high),
                           "bootstrap_unit": "paired source WAV within day", "repeats": len(delta), "conditional_developmental_diagnostic": True})
    return result


def main():
    rows, folds, data_root, outer, base, y, days = p.setup()
    hgb = np.load(PRIOR / f"{HGB}_oof.npz")["clean"]
    controls = fit_controls(base, hgb, y)
    frozen = (OUT / "frozen_controls.json").read_bytes()
    train_pool = {k: base[:, i] for i, k in enumerate(p.KEYS)}
    train_pool[HGB] = hgb
    train_metrics = []
    for control in controls:
        prediction = apply_control(control, train_pool)
        train_metrics.extend(p.record_summary(control["method"], "training_fit_diagnostic", "clean", y, prediction, days))
    save_csv("training_control_metrics.csv", train_metrics)
    outer_ids = [p.identity(r) for r in outer]
    lookup = {noise: {} for noise in p.NOISE_DIRS}
    # Only the predefined original methods are read; no outer best-method search.
    with (PRIOR / "outer_predictions.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["method"] in ORIGINAL:
                lookup[row["noise"]].setdefault(row["method"], {})[p.identity(row)] = row
    metrics, region_rows, wav_rows, attribution, bootstrap, complementarity = [], [], [], [], [], []
    prediction_rows = []
    contrasts = [(FOUR, "baseline_performance"), (FOUR, "existing3_mse"), (FOUR, HGB), (DIVERSE, FOUR), (AFFINE, "existing3_affine")]
    for noise in p.NOISE_DIRS:
        pool = {method: np.asarray([float(saved[idx]["prediction_kW_m2"]) for idx in outer_ids]) for method, saved in lookup[noise].items()}
        assert all(set(saved) == set(outer_ids) for saved in lookup[noise].values())
        baseline_rows = [lookup[noise]["baseline_performance"][idx] for idx in outer_ids]
        yo = np.asarray([float(r["y_kW_m2"]) for r in baseline_rows])
        do = np.asarray([r["source_wav_id"][:8] for r in outer])
        t = np.asarray([p.THRESHOLDS[d] for d in do])
        for control in controls:
            pool[control["method"]] = apply_control(control, pool)
        for name, prediction in pool.items():
            metrics.extend(p.record_summary(name, "outer_unused_chunk", noise, yo, prediction, do))
            for region, mask in regions(yo, t).items():
                e = prediction[mask]-yo[mask]
                region_rows.append({"method": name, "noise": noise, "region": region, "n": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "bias": float(e.mean()),
                    "fp": int(((yo[mask] < t[mask]) & (prediction[mask] >= t[mask])).sum()),
                    "fn": int(((yo[mask] >= t[mask]) & (prediction[mask] < t[mask])).sum())})
            for wav in sorted({r["source_wav_id"] for r in outer}):
                mask = np.asarray([r["source_wav_id"] == wav for r in outer])
                e = prediction[mask]-yo[mask]
                wav_rows.append({"method": name, "noise": noise, "source_wav_id": wav, "day": wav[:8], "y": float(yo[mask][0]),
                    "n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(e**2))), "sse": float(np.sum(e**2)), "mae": float(np.mean(np.abs(e))),
                    "fp": int(((yo[mask] < t[mask]) & (prediction[mask] >= t[mask])).sum()), "fn": int(((yo[mask] >= t[mask]) & (prediction[mask] < t[mask])).sum())})
            if name not in ORIGINAL:
                for i, row in enumerate(outer):
                    prediction_rows.append({"method": name, "noise": noise, "source_wav_id": row["source_wav_id"], "chunk_index": row["chunk_index"], "y_kW_m2": yo[i], "prediction_kW_m2": prediction[i]})
        for candidate, reference in contrasts:
            cp, rp = pool[candidate], pool[reference]
            for region, mask in {"all": np.ones(len(yo), bool), **regions(yo, t)}.items():
                sse_ref = float(((rp[mask]-yo[mask])**2).sum())
                sse_new = float(((cp[mask]-yo[mask])**2).sum())
                attribution.append({"noise": noise, "candidate": candidate, "reference": reference, "region": region, "n": int(mask.sum()),
                    "reference_sse": sse_ref, "candidate_sse": sse_new, "sse_reduction": sse_ref-sse_new,
                    "mean_squared_error_reduction": (sse_ref-sse_new)/int(mask.sum())})
            positive = yo >= t
            complementarity.append({"noise": noise, "candidate": candidate, "reference": reference,
                "fn_corrected": int((positive & (rp<t) & (cp>=t)).sum()), "new_fn": int((positive & (rp>=t) & (cp<t)).sum()),
                "fp_corrected": int((~positive & (rp>=t) & (cp<t)).sum()), "new_fp": int((~positive & (rp<t) & (cp>=t)).sum())})
        if noise in ["clean", "-20"]:
            bootstrap.extend(paired_bootstrap(outer, yo, pool, noise))
    assert (OUT / "frozen_controls.json").read_bytes() == frozen
    save_csv("all_metrics.csv", metrics)
    save_csv("regional_metrics.csv", region_rows)
    save_csv("wav_metrics.csv", wav_rows)
    save_csv("sse_attribution.csv", attribution)
    save_csv("paired_bootstrap.csv", bootstrap)
    save_csv("paired_error_changes.csv", complementarity)
    save_csv("new_control_predictions.csv", prediction_rows)
    records = {(r["method"], r["noise"], r["day"]): r for r in metrics}
    summary, variability = [], []
    for method in pool:
        clean = records[(method, "clean", "two_day")]
        strong = records[(method, "-20", "two_day")]
        noisy = [records[(method, n, "two_day")] for n in p.NOISE_DIRS if n != "clean"]
        summary.append({"method": method, **{f"clean_{k}": clean[k] for k in ["rmse", "mae", "r2", "recall", "precision", "f1", "roc_auc", "pr_auc", "fp", "fn"]},
            "noise_mean_rmse": float(np.mean([r["rmse"] for r in noisy])), "noise_mean_mae": float(np.mean([r["mae"] for r in noisy])),
            "noise_mean_recall": float(np.mean([r["recall"] for r in noisy])), "noise_fp_sum": sum(r["fp"] for r in noisy),
            **{f"minus20_{k}": strong[k] for k in ["rmse", "mae", "r2", "recall", "f1", "fp", "fn", "rmse_onb"]},
            "rmse_increase_clean_to_minus20": strong["rmse"]-clean["rmse"]})
    for candidate, reference in contrasts:
        for noise in p.NOISE_DIRS:
            pairs = {r["source_wav_id"]: r for r in wav_rows if r["method"] == reference and r["noise"] == noise}
            delta = [(r["source_wav_id"], float(pairs[r["source_wav_id"]]["sse"])-r["sse"]) for r in wav_rows if r["method"] == candidate and r["noise"] == noise]
            delta.sort(key=lambda v: v[1], reverse=True)
            total = sum(d[1] for d in delta)
            variability.append({"noise": noise, "candidate": candidate, "reference": reference,
                "wavs_improved": sum(d[1] > 1e-8 for d in delta), "wavs_worsened": sum(d[1] < -1e-8 for d in delta),
                "total_sse_reduction": total, "top_sse_reduction_wav": delta[0][0], "top_sse_reduction": delta[0][1],
                "top_share_of_net_sse_reduction": delta[0][1]/total if total > 0 else float("nan")})
    save_csv("summary_metrics.csv", summary)
    save_csv("wav_gain_distribution.csv", variability)
    # Verify preserved methods against the prior, and validate the new control metrics independently.
    prior_metrics = {(r["method"], r["noise"], r["day"]): r for r in p.read_csv(PRIOR/"outer_metrics.csv")}
    for row in metrics:
        if row["method"] in ORIGINAL:
            expected = prior_metrics[(row["method"], row["noise"], row["day"])]
            for key in ["rmse", "mae", "r2", "recall", "fpr", "fp", "fn", "q100"]:
                np.testing.assert_allclose(row[key], float(expected[key]), rtol=1e-12, atol=1e-10, equal_nan=True)
    for row in metrics:
        if row["method"] not in ORIGINAL:
            samples = [r for r in prediction_rows if r["method"] == row["method"] and r["noise"] == row["noise"] and (row["day"] == "two_day" or r["source_wav_id"].startswith(row["day"]))]
            yt = np.asarray([r["y_kW_m2"] for r in samples]); pt = np.asarray([r["prediction_kW_m2"] for r in samples])
            tt = np.asarray([p.THRESHOLDS[r["source_wav_id"][:8]] for r in samples])
            np.testing.assert_allclose(row["rmse"], np.linalg.norm(pt-yt)/np.sqrt(len(yt)), rtol=1e-12)
            assert row["fp"] == int(((yt<tt) & (pt>=tt)).sum()) and row["fn"] == int(((yt>=tt) & (pt<tt)).sum())
    labels = ["Existing performance", "Existing, MSE weights", "Frequency HGB", "Add HGB, MSE weights", "Add HGB, diversity weights"]
    shown = ["baseline_performance", "existing3_mse", HGB, FOUR, DIVERSE]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for j, method in enumerate(shown):
        for ax, field, unit in [(axes[0], "rmse", "RMSE (kW/m2)"), (axes[1], "recall", "ONB recall (%)")]:
            values = [records[(method, noise, "two_day")][field] * (100 if field == "recall" else 1) for noise in p.NOISE_DIRS]
            ax.plot(range(7), values, marker="o", label=labels[j])
            ax.set_ylabel(unit); ax.set_xticks(range(7), list(p.NOISE_DIRS)); ax.set_xlabel("Evaluation noise (dB)"); ax.grid(alpha=.25)
    axes[0].set_title("Regression benefit under the same weight fitting")
    axes[1].set_title("Detection benefit beyond the all-positive endpoint")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, 1)); fig.savefig(OUT/"general_metric_comparison.png", dpi=170); fig.savefig(OUT/"general_metric_comparison.pdf"); plt.close(fig)
    save_json("audit.json", {"status": "passed", "outer_chunks": 540, "training_chunks": 1620, "shared_chunks": 0, "additional_weight_controls": len(controls),
        "new_base_model_training": False, "new_affine_meta_fit": 1, "outer_labels_used_for_weight_selection": False,
        "frozen_controls_sha256": hashlib.sha256(frozen).hexdigest(), "metric_rows": len(metrics), "regional_rows": len(region_rows), "bootstrap_contrasts": len(bootstrap),
        "prior_metrics_reproduced": True, "new_control_rmse_fp_fn_independently_verified": True, "main_runner_changed": False})
    print("Review completed:", len(metrics), "metric rows;", len(controls), "matched-algorithm controls")
    for row in summary:
        if row["method"] in shown or row["method"] in ["existing3_affine", AFFINE]:
            print(row["method"], "clean RMSE", round(row["clean_rmse"], 3), "MAE", round(row["clean_mae"], 3), "recall", round(row["clean_recall"], 4), "mean noise", round(row["noise_mean_rmse"], 3), "minus20", round(row["minus20_rmse"], 3), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
