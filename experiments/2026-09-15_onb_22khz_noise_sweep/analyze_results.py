"""Freeze and analyze the completed September 15, 22 kHz, seven-noise run.

Reads original results only. No training or TensorFlow import.
Usage: python analyze_results.py --output <new-directory>
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RUN = "Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2/regression_result/npy/ensemble/20260915_selected_log_architecture"
NOISES = ["no_noise", "0", "-4", "-8", "-12", "-16", "-20"]
MODELS = ["rf", "cnntf_v2_gap", "alexnet", "ensemble__simple_equal", "ensemble__inner_holdout"]
LABELS = ["RF", "CNN+Transformer", "AlexNet", "Equal ensemble", "Inner-holdout ensemble"]


def longpath(p):
    s = str(p.absolute())
    return Path("\\\\?\\" + s) if os.name == "nt" and not s.startswith("\\\\?\\") else p


def relative(p):
    s = str(p)
    return Path(s[4:] if s.startswith("\\\\?\\") else s).relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a new empty directory; preserve existing snapshots.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir()
    sources, tables, checks, configs = {}, {}, [], []

    def read(p, mutable=False):
        data = p.read_bytes()
        sources[relative(p)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                "shared_with_other_frequency_runs": mutable}
        return data

    def obj(p):
        return json.loads(read(p).decode("utf-8-sig"))

    def rows(p):
        return list(csv.DictReader(io.StringIO(read(p).decode("utf-8-sig"))))

    def add(name, values):
        tables.setdefault(name, []).extend(values)

    def savefig(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / "figures" / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    base = longpath(ROOT / RUN)
    paths = list(base.glob("maxfreq=22kHz/*/*/run_manifest.json"))
    assert len(paths) == 7, f"Expected exactly 7 manifests, got {len(paths)}"
    runs = {}
    for p in paths:
        m = obj(p)
        snr = m["dataset"]["snr_value"]
        assert snr in NOISES and snr not in runs
        runs[snr] = (p.parent, m)
    folds_reference = None
    for snr in NOISES:
        d, m = runs[snr]
        assert m["run_hash"] == "2b4cef"
        cfg = m["validation_config"]
        assert cfg["learning_policy"] == {"split_mode": "within_day", "training_noise": "matched"}
        configs.append(cfg)
        threshold = float(m["dataset"]["threshold"])
        marker = obj(d / "completed.json")
        assert marker["run_hash"] == m["run_hash"]
        split = obj(d / "split_manifest.json")["folds"]
        assert len(split) == 3
        identity = []
        for fold in split:
            train, val = set(fold["training_wav_groups"]), set(fold["evaluation_wav_groups"])
            assert not train & val and len(train) == 12 and len(val) == 6
            assert fold["n_training_chunks"] == 720 and fold["n_evaluation_chunks"] == 360
            identity.append((fold["fold"], sorted(train), sorted(val)))
            add("split_audit", [{"snr": snr, "fold": fold["fold"], "fit_id": fold["fit_id"],
                                 "training_wavs": len(train), "validation_wavs": len(val), "overlap": len(train & val),
                                 "inner_holdout_errors_json": json.dumps(fold["inner_holdout_errors"])}])
        if folds_reference is None:
            folds_reference = identity
        assert identity == folds_reference
        chunks = []
        fold_rows = {}
        for p in sorted((d / "fold_pred").glob("*.csv")):
            rr = rows(p)
            fold_rows[int(rr[0]["fold"])] = rr
            chunks.extend(rr)
        assert len(fold_rows) == 3 and len(chunks) == 1080
        chunk_df = pd.DataFrame(chunks)
        assert not chunk_df.duplicated(["source_wav_id", "chunk_index"]).any()
        assert chunk_df.groupby("source_wav_id").size().eq(60).all()
        assert chunk_df.groupby("source_wav_id")["fold"].nunique().eq(1).all()
        for c in ["y_true", *MODELS]:
            chunk_df[c] = pd.to_numeric(chunk_df[c])
        pred = rows(d / "wav_eval" / f"wav_predictions_{snr}.csv")
        assert len(pred) == 18
        ev = obj(d / "wav_eval" / f"evaluation_manifest_{snr}.json")
        assert ev["n_wavs"] == 18 and ev["n_oof_chunks"] == 1080
        metric = [x for x in rows(d / "wav_eval" / f"wav_metrics_{snr}.csv") if x["aggregation"] == "median"]
        assert {x["model_key"] for x in metric} == set(MODELS)
        y = np.array([float(x["y_true"]) for x in pred])
        assert (y >= threshold).sum() == 8 and (np.abs(y-threshold) <= threshold*.1).sum() == 1
        for row in metric:
            key = row["model_key"]
            yy = np.array([float(x[f"{key}_pred_median"]) for x in pred])
            for x in pred:
                raw = chunk_df.loc[chunk_df.source_wav_id == x["source_wav_id"], key]
                assert np.isclose(np.median(raw), float(x[f"{key}_pred_median"]), rtol=0, atol=1e-6)
            fn = int(((y >= threshold) & (yy < threshold)).sum())
            fp = int(((y < threshold) & (yy >= threshold)).sum())
            near = np.argmin(abs(y-threshold))
            computed = {"r2": r2_score(y, yy), "rmse_all": mean_squared_error(y, yy)**.5,
                        "recall": 1-fn/8, "roc_auc_cont": roc_auc_score(y >= threshold, yy)}
            for key_metric, value in computed.items():
                assert np.isclose(value, float(row[key_metric]), rtol=0, atol=1e-6)
            checks.append({"snr": snr, "model": key, "metrics_and_chunk_to_wav_verified": True})
            add("wav_median_metrics", [{"snr": snr, **row, "false_negatives": fn, "false_positives": fp,
                                       "onb_prediction": yy[near], "onb_signed_error": yy[near]-threshold}])
        add("wav_predictions", [{"snr": snr, **x} for x in pred])
        add("chunk_metrics", [{"snr": snr, **x} for x in rows(d / f"metrics_summary_{snr}.csv")])
        add("ensemble_weights", [{"snr": snr, **x} for x in rows(d / f"ensemble_weights_{snr}.csv")])
        add("onb_transitions", [{"snr": snr, **x} for x in rows(d / "wav_eval" / f"onb_transition_summary_{snr}.csv") if x["aggregation"] == "median"])
        add("run_inventory", [{"snr": snr, "run_hash": m["run_hash"], "run_instance_id": m["run_instance_id"],
                              "created_at": m["created_at"], "completion_marker_mtime": datetime.fromtimestamp((d/'completed.json').stat().st_mtime).astimezone().isoformat(),
                              "run_manifest": relative(d/'run_manifest.json'), "n_wavs": 18, "n_chunks": 1080, "complete": True}])
        for f in range(1, 4):
            for model in MODELS[:3]:
                folder = d / "explainability" / f"fold{f}" / model
                ctx = {"snr": snr, "fold": f, "model": model}
                required = ["explainability_summary.csv", "group_mask_performance.csv", "explained_samples.csv", "explainability_config.csv"]
                required += (["treeshap_pca_summary.csv", "treeshap_status.csv"] if model == "rf" else ["input_stability.csv", "top_layer_randomization_sanity.csv"])
                assert all((folder/n).exists() for n in required)
                for n in required:
                    rr = rows(folder/n)
                    add(n[:-4], [{**ctx, **x} for x in rr])
                samples = rows(folder / "explained_samples.csv")
                assert len(samples) == 5
                for s in samples:
                    raw = fold_rows[f][int(s["val_local_index"])]
                    assert np.isclose(float(s["y_true"]), float(raw["y_true"]), atol=1e-6)
                    assert np.isclose(float(s["y_pred"]), float(raw[model]), atol=.5, rtol=1e-5)
                    add("sample_correspondence", [{**ctx, **s, "source_wav_id": raw["source_wav_id"],
                         "chunk_index": raw["chunk_index"], "sample_filename": raw["sample_filename"],
                         "sample_dir": relative(folder/s["sample_id"]), "within_onb_10pct": abs(float(s["y_true"])-threshold) <= threshold*.1}])
                    if model != "rf":
                        q = folder / s["sample_id"] / "integrated_gradients_signed.npy"
                        arr = np.load(io.BytesIO(read(q)), allow_pickle=False)
                        assert np.isfinite(arr).all()
                add("xai_output_inventory", [{**ctx, "samples": len(samples), "required_files_present": True}])
        # Pooled-WAV residual correlation describes complementarity, not an independent test.
        residual = pd.DataFrame({key: [float(x[f"{key}_pred_median"])-float(x["y_true"]) for x in pred] for key in MODELS[:3]})
        for i in range(3):
            for j in range(i+1, 3):
                add("residual_correlations", [{"snr": snr, "model_a": MODELS[i], "model_b": MODELS[j], "pearson": residual.iloc[:,i].corr(residual.iloc[:,j])}])

    # This summary is shared with the user's ongoing 3 kHz run. Freeze only target rows.
    p = base / "tuning_summary.csv"
    tuning = list(csv.DictReader(io.StringIO(read(p, mutable=True).decode("utf-8-sig"))))
    tuning = [r for r in tuning if r["max_freq_hz"] == "maxfreq=22kHz" and r["run_hash"] == "2b4cef"]
    assert len(tuning) == 35
    add("training_summary", tuning)
    metric = pd.DataFrame(tables["wav_median_metrics"])
    for c in ["r2", "rmse_all", "recall", "onb_signed_error"]:
        metric[c] = pd.to_numeric(metric[c])
    pivot = metric.pivot(index="snr", columns="model_key", values="r2").reindex(NOISES)
    for snr in NOISES:
        v = pivot.loc[snr]
        best_single = v[MODELS[:3]].idxmax()
        best_all = v.idxmax()
        add("model_comparison", [{"snr": snr, "best_single": best_single, "best_overall": best_all,
                                 "best_single_r2": v[best_single], "best_overall_r2": v[best_all],
                                 "equal_minus_best_single": v[MODELS[3]]-v[best_single],
                                 "inner_minus_best_single": v[MODELS[4]]-v[best_single]}])
    masks = pd.DataFrame(tables["group_mask_performance"])
    for c in ["r2_drop", "rmse_all_increase", "low", "high"]:
        masks[c] = pd.to_numeric(masks[c])
    for (snr, model, axis), group in masks.groupby(["snr", "model", "axis"]):
        for name, gg in group.groupby("group"):
            add("mask_aggregates", [{"snr": snr, "model": model, "axis": axis, "group": name,
                "low": gg.low.iloc[0], "high": gg.high.iloc[0], "mean_r2_drop": gg.r2_drop.mean(),
                "min_r2_drop": gg.r2_drop.min(), "max_r2_drop": gg.r2_drop.max(),
                "mean_rmse_increase": gg.rmse_all_increase.mean()}])
    for (snr, model, fold), g in masks[masks.axis == "frequency"].groupby(["snr", "model", "fold"]):
        top = g.loc[g.r2_drop.idxmax()]
        add("top_mask_bands", [{"snr": snr, "model": model, "fold": fold, "group": top.group, "r2_drop": top.r2_drop}])
    exp = pd.DataFrame(tables["explainability_summary"])
    ig = exp[exp.method == "integrated_gradients"].copy()
    for c in ["completeness_relative_error", "completeness_error", "attribution_sum", "output_delta_from_baseline"]:
        ig[c] = pd.to_numeric(ig[c])
    for (snr, model), g in ig.groupby(["snr", "model"]):
        add("ig_diagnostics", [{"snr": snr, "model": model, "n": len(g),
            "n_relative_error_gt_005": int((g.completeness_relative_error>.05).sum()),
            "median_relative_error": g.completeness_relative_error.median(), "max_relative_error": g.completeness_relative_error.max(),
            "median_absolute_error": g.completeness_error.abs().median(),
            "median_output_delta_abs": g.output_delta_from_baseline.abs().median()}])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(7)
    ticks = ["Clean", "0", "-4", "-8", "-12", "-16", "-20"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for key, label in zip(MODELS, LABELS):
        g = metric[metric.model_key == key].set_index("snr").reindex(NOISES)
        axes[0].plot(x, g.r2, marker="o", label=label)
        axes[1].plot(x, g.rmse_all/1000, marker="o", label=label)
    for ax in axes:
        ax.set_xticks(x, ticks); ax.set_xlabel("Reference SNR (dB); clean shown separately")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("R2 (pooled OOF WAV medians)")
    axes[1].set_ylabel("RMSE (kW/m2)")
    axes[0].legend(fontsize=8)
    fig.suptitle("2025-06-11 / 22 kHz / matched training / 18 source WAVs / seed 42")
    fig.tight_layout(); savefig(fig, "performance_by_noise")

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for key, label in zip(MODELS, LABELS):
        g = metric[metric.model_key == key].set_index("snr").reindex(NOISES)
        ax.plot(x, g.onb_signed_error/1000, marker="o", label=label)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, ticks); ax.set_xlabel("Reference SNR (dB)")
    ax.set_ylabel("Prediction minus ONB threshold (kW/m2)")
    ax.set_title("First ONB WAV (368.978 kW/m2): negative values mean a miss")
    ax.legend(fontsize=8, ncols=2); ax.grid(alpha=.2)
    fig.tight_layout(); savefig(fig, "first_onb_prediction")

    ag = pd.DataFrame(tables["mask_aggregates"])
    freq_order = ag[ag.axis == "frequency"].sort_values("low").group.unique()
    matrix_list = [ag[(ag.axis == "frequency") & (ag.model == k)].pivot(index="group", columns="snr", values="mean_r2_drop").reindex(index=freq_order, columns=NOISES) for k in MODELS[:3]]
    bound = max(abs(m.to_numpy()).max() for m in matrix_list)
    from matplotlib.colors import SymLogNorm
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)
    for ax, mat, label in zip(axes, matrix_list, LABELS[:3]):
        im = ax.imshow(mat, cmap="RdBu_r", norm=SymLogNorm(linthresh=.1, vmin=-bound, vmax=bound), aspect="auto")
        ax.set_xticks(x, ticks, rotation=35); ax.set_title(label)
        ax.set_yticks(range(8), [n.replace("freq_", "").replace("Hz", "") for n in freq_order])
        for i in range(8):
            for j in range(7):
                val = mat.iloc[i,j]
                ax.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(val)>2 else "black")
        ax.set_xlabel("Reference SNR (dB)")
    axes[0].set_ylabel("Masked frequency band (Hz)")
    fig.suptitle("Mean fold-local WAV R2 drop after masking; unequal band widths; shared symmetric-log color scale")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label="R2 drop (positive = worse after masking)")
    savefig(fig, "frequency_mask_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, label in zip(axes, MODELS[1:3], LABELS[1:3]):
        data = [ig[(ig.snr == s) & (ig.model == key)].completeness_relative_error.to_numpy() for s in NOISES]
        ax.boxplot(data, labels=ticks, showfliers=True)
        ax.set_yscale("log"); ax.axhline(.05,color="red",ls="--",label="0.05 descriptive reference")
        ax.set_title(label); ax.set_xlabel("Reference SNR (dB)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("IG completeness\nrelative error")
    fig.suptitle("15 selected samples per model and noise condition; logarithmic error scale")
    fig.tight_layout(); savefig(fig, "ig_completeness")

    for name, values in tables.items():
        pd.DataFrame(values).to_csv(out/f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest = {"collected_at": datetime.now().astimezone().isoformat(), "scope": RUN+"/maxfreq=22kHz",
                "run_hash": "2b4cef", "sources": sources, "counts": {k:len(v) for k,v in tables.items()},
                "verification": checks, "same_source_wav_outer_splits_across_all_noise_conditions": True,
                "notes": ["Source files and ongoing other-frequency runs were not modified.",
                          "Training summary file is shared with later runs; extracted 22 kHz rows are frozen here.",
                          "Mask R2 values are fold-local; main R2 values pool OOF source WAV medians.",
                          "0.05 IG reference is descriptive, not a preregistered validity threshold."]}
    (out/"snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"]))
    print(pivot[MODELS].round(6).to_string())
    print(pd.DataFrame(tables["ig_diagnostics"]).to_string(index=False))


if __name__ == "__main__":
    main()
