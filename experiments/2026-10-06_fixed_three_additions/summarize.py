"""Save readable profiles, cluster uncertainty, error complementarity and figures."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_fixed_three_additions as run
from run_hgb_complementarity_validation import read_csv, read_json, save_csv, save_json

OUT = Path(__file__).resolve().parent
METRICS = read_csv(OUT / "metrics.csv")
ANCHOR = "existing3_clean_mse"
METHODS = [ANCHOR, ANCHOR+"__hgb4_anchor", ANCHOR+"__extra_trees4_anchor", ANCHOR+"__selected5"]
LABELS = ["Original three", "+ HGB (4)", "+ ExtraTrees (4)", "+ HGB + ExtraTrees (5)"]
COLORS = ["#525252", "#d89026", "#35988b", "#345cad"]


def row(seed, name, noise, day="two_day"):
    return next(r for r in METRICS if int(r["seed"]) == seed and r["method"] == name and r["noise"] == noise and r["day"] == day)


def profiles(names):
    result = []
    for seed in run.CONFIG["seeds"]:
        for name in names:
            clean, strong = row(seed, name, "clean"), row(seed, name, "-20")
            result.append({"seed": seed, "method": name,
                **{f"clean_{k}": float(clean[k]) for k in ["rmse", "mae", "r2", "recall", "f1", "fp", "fn", "rmse_onb", "roc_auc", "pr_auc"]},
                "mean_six_noise_rmse": float(np.mean([float(row(seed, name, n)["rmse"]) for n in run.NOISES[1:]])),
                "max_fp_all_seven": max(int(row(seed, name, n)["fp"]) for n in run.NOISES),
                **{f"minus20_{k}": float(strong[k]) for k in ["rmse", "mae", "fn", "fp", "rmse_onb"]},
                **{f"clean_q100_{d}": float(row(seed, name, "clean", d)["q100"]) for d in run.THRESHOLDS},
                **{f"minus20_q100_{d}": float(row(seed, name, "-20", d)["q100"]) for d in run.THRESHOLDS}})
    return result


def diagnostics():
    complementarity, bootstrap, inventory, imports = [], [], [], []
    rng = np.random.default_rng(20261006)
    for seed in run.CONFIG["seeds"]:
        manifest, _ = run.load_seed(seed)
        folder = OUT / f"seed{seed}"
        with np.load(folder / "predictions.npz") as saved:
            base, indices = saved["outer"].copy(), saved["test_indices"].copy()
        y = run.labels(manifest, indices)
        wavs = np.asarray([manifest["samples"][int(i)]["source_wav_id"] for i in indices])
        threshold = np.asarray([run.THRESHOLDS[w[:8]] for w in wavs])
        actual = y >= threshold
        with np.load(folder / "outer_integrated.npz") as saved:
            p = saved["predictions"].copy()
            names = saved["methods"].tolist()
        for n, noise in enumerate(run.NOISES):
            common = actual & np.all(base[n, :, :3] < threshold[:, None], axis=1)
            for j, name in enumerate(run.KEYS[3:], 3):
                candidate = base[n, :, j]
                residual_corr = np.corrcoef(candidate-y, base[n, :, 3]-y)[0, 1]
                complementarity.append({"seed": seed, "noise": noise, "candidate": name,
                    "original_three_common_fn": int(common.sum()),
                    "common_fn_corrected_by_candidate": int(np.sum(common & (candidate >= threshold))),
                    "candidate_fp": int(np.sum(~actual & (candidate >= threshold))),
                    "residual_correlation_with_hgb": float(residual_corr)})
            if noise not in ["clean", "-20"]:
                continue
            draw = rng.integers(0, 36, size=(3000, 36))
            unique_wavs = sorted(set(wavs))
            for baseline, addition in [(METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]),
                (METHODS[0], METHODS[3]), (METHODS[1], METHODS[3]), (METHODS[2], METHODS[3])]:
                a, b = p[n, :, names.index(baseline)], p[n, :, names.index(addition)]
                before = np.asarray([np.mean((a[wavs==w]-y[wavs==w])**2) for w in unique_wavs])
                after = np.asarray([np.mean((b[wavs==w]-y[wavs==w])**2) for w in unique_wavs])
                delta = np.sqrt(after[draw].mean(axis=1))-np.sqrt(before[draw].mean(axis=1))
                bootstrap.append({"seed": seed, "noise": noise, "baseline": baseline, "addition": addition,
                    "delta_rmse": float(np.sqrt(after.mean())-np.sqrt(before.mean())),
                    "cluster_bootstrap_low95": float(np.quantile(delta, .025)),
                    "cluster_bootstrap_high95": float(np.quantile(delta, .975)),
                    "improved_wavs": int(np.sum(after<before)), "wavs": len(unique_wavs),
                    "resamples": len(draw), "scope": "descriptive uncertainty conditional on reused development data"})
        for stage in ["fold1", "fold2", "fold3", "final"]:
            for family in run.FAMILIES:
                audit = read_json(folder / stage / f"{family}_complete.json")
                inventory.append({"seed": seed, "stage": stage, "family": family,
                    "representation": audit["representation"], "selected_params": str(audit["selected_params"]),
                    "selected_inner_cv_mse": audit["inner_cv_mse"][audit["selected_index"]],
                    "fit_chunks": len(audit["fit_indices"]), "held_chunks": len(audit["held_indices"]),
                    "elapsed_seconds": audit["elapsed_seconds"]})
        # A descriptive structural dependence; impurity importance is not causality.
        model = run.joblib.load(folder / "final/extra_trees.joblib").regressor_
        groups = {"absolute_power": range(0, 5), "power_ratios": range(5, 7),
            "centroid_entropy": range(7, 9), "temporal_cv": [9], "frequency_shape": range(10, 34)}
        for group, idx in groups.items():
            imports.append({"seed": seed, "model": "extra_trees", "group": group,
                "impurity_importance_sum": float(model.feature_importances_[list(idx)].sum()),
                "interpretation": "descriptive split dependence, not physical source attribution"})
    save_csv(OUT / "error_complementarity.csv", complementarity)
    save_csv(OUT / "wav_cluster_bootstrap.csv", bootstrap)
    save_csv(OUT / "candidate_fit_inventory.csv", inventory)
    save_csv(OUT / "extra_trees_dependence.csv", imports)
    return bootstrap


def plot():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for c, seed in enumerate(run.CONFIG["seeds"]):
        clean = [row(seed, name, "clean") for name in METHODS]
        bars = axes[0, c].bar(np.arange(4), [float(r["rmse"]) for r in clean], color=COLORS)
        for bar, r in zip(bars, clean):
            axes[0, c].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
                f"{float(r['rmse']):.2f}\nFN {r['fn']} / FP {r['fp']}", ha="center", va="bottom", fontsize=9)
        axes[0, c].set_xticks(np.arange(4))
        axes[0, c].set_xticklabels(["3", "3+HGB", "3+ET", "3+HGB+ET"])
        axes[0, c].set_ylim(0, 44)
        axes[0, c].set_ylabel("Clean RMSE (kW/m2)")
        axes[0, c].set_title(f"Seed {seed}: original models and their ratios retained")
        for name, label, color in zip(METHODS, LABELS, COLORS):
            axes[1, c].plot(np.arange(7), [float(row(seed, name, n)["rmse"]) for n in run.NOISES],
                marker="o", markersize=4, color=color, label=label)
        axes[1, c].set_xticks(np.arange(7))
        axes[1, c].set_xticklabels(["clean", "0", "-4", "-8", "-12", "-16", "-20"])
        axes[1, c].set_xlabel("Reference SNR (dB)")
        axes[1, c].set_ylabel("RMSE (kW/m2)")
        axes[1, c].set_ylim(15, 105)
        axes[1, c].grid(axis="y", alpha=.2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .01), frameon=False)
    fig.suptitle("Clean-only fixed-core additions; 540 unused chunks per split, 36 reused recordings")
    fig.tight_layout(rect=(0, .09, 1, .97))
    fig.savefig(OUT / "fixed_three_comparison.png", dpi=180)
    fig.savefig(OUT / "fixed_three_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    headline = METHODS + ["existing_performance", "existing_performance__hgb4_anchor",
        "existing_performance__extra_trees4_anchor", "existing_performance__selected5",
        "hgb4_equal", "hgb_extra_trees5_equal"]
    save_csv(OUT / "headline_profiles.csv", profiles(headline))
    save_csv(OUT / "candidate_profiles.csv", profiles(run.KEYS))
    bootstrap = diagnostics()
    plot()
    summary = {"config_sha256": run.CONFIG_HASH, "evaluation_scope": run.CONFIG["evaluation_scope"],
        "selected_fifth": {str(s): read_json(OUT / f"seed{s}/frozen_integration.json")["selected_fifth"] for s in run.CONFIG["seeds"]},
        "primary_candidate": {"models": ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"],
            "method": "existing3_clean_mse__selected5", "original_three_retained": True,
            "core_internal_ratio": "Saved clean-OOF MSE for original three; kept fixed when adding candidates",
            "minimum_core_total": .25, "minimum_each_addition": .05,
            "weights": "Per-run clean OOF fitting, not average of test-seed weights"},
        "headline_profiles": profiles(METHODS), "descriptive_cluster_uncertainty": bootstrap,
        "limitations": ["Same two days reused in development", "Five does not improve every metric versus four",
            "Seed43 q100 worse than HGB4 on June18", "ExtraTrees-only fourth also competitive",
            "Normal ONB main runner still retains original three; experimental entrypoint provides additions"]}
    save_json(OUT / "summary.json", summary)
    print("[summary] profiles, complementarity, descriptive cluster intervals, PNG/PDF saved")
