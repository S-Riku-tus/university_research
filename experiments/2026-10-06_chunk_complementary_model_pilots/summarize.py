"""Compact derived tables and exportable figures from completed pilot outputs."""
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot import (KEYS, OUT, THRESHOLDS, WEIGHTS, candidates, endpoints,
                   identity, read_csv, read_json, save_csv, save_json, setup)


DISPLAY = {
    "baseline_performance": "Existing performance",
    "conformer_alexnet_equal": "Conformer + AlexNet equal",
    "band10_HGB_rmse": "HGB band10",
    "frequency34_HGB_rmse": "HGB frequency34",
    "temporal46_HGB_rmse": "HGB temporal46",
    "frequency34_HGB_rmse__equal4": "Frequency HGB, equal4",
    "frequency34_HGB_rmse__mse_simplex": "Frequency HGB, MSE weights",
    "frequency34_HGB_rmse__diversity_simplex": "Frequency HGB, diversity weights",
    "frequency34_HGB_rmse__affine_ridge": "Frequency HGB, affine Ridge",
}


def main():
    rows, folds, data_root, outer, base, y, days = setup()
    t = np.asarray([THRESHOLDS[d] for d in days])
    pool = {r["method"]: np.load(OUT / f"{r['method']}_oof.npz")["clean"] for r in candidates()}
    prediction = {key: base[:, j] for j, key in enumerate(KEYS)}
    prediction.update(pool)
    coverage, margins, feasibility = [], [], []
    for day in THRESHOLDS:
        mask = days == day
        bfn = mask & (y >= t) & (base @ WEIGHTS < t)
        last = float(y[bfn].max())
        last_neg = bfn & (y == last)
        common = mask & (y >= t) & np.all(base < t[:, None], axis=1)
        for name, p in pool.items():
            coverage.append({"method": name, "day": day, "last_baseline_stage": last, "last_negatives": int(last_neg.sum()), "last_negatives_corrected": int((last_neg & (p >= t)).sum()), "common_fn": int(common.sum()), "common_fn_corrected": int((common & (p >= t)).sum()), "new_fn": int((mask & (y >= t) & (base @ WEIGHTS >= t) & (p < t)).sum()), "new_fn_at_or_above_last": int((mask & (y >= last) & (base @ WEIGHTS >= t) & (p < t)).sum())})
        for i in np.flatnonzero(last_neg):
            for name, p in prediction.items():
                margins.append({"day": day, "source_wav_id": rows[i]["source_wav_id"], "chunk_index": rows[i]["chunk_index"], "y_kW_m2": y[i], "onb_kW_m2": t[i], "method": name, "prediction_kW_m2": p[i], "margin_kW_m2": p[i]-t[i]})
            maximum = max(p[i] for p in prediction.values())
            feasibility.append({"day": day, "source_wav_id": rows[i]["source_wav_id"], "chunk_index": rows[i]["chunk_index"], "y_kW_m2": y[i], "candidate_positive_count": int(sum(p[i] >= t[i] for p in pool.values())), "maximum_existing_plus_candidate_margin": maximum-t[i], "impossible_for_any_convex_average_of_evaluated_oof_predictions": bool(maximum < t[i])})
    save_csv("training_last_stage_corrections.csv", coverage)
    save_csv("training_last_stage_margins.csv", margins)
    save_csv("training_convex_feasibility.csv", feasibility)
    metrics = read_csv(OUT / "outer_metrics.csv")
    lookup = {(r["method"], r["noise"], r["day"]): r for r in metrics}
    summaries, noise_aggregate = [], []
    for method in DISPLAY:
        r = lookup[(method, "clean", "two_day")]
        summaries.append({"method": method, "rmse_clean": float(r["rmse"]), "fp_clean": int(r["fp"]), "fn_clean": int(r["fn"]), "q100_0611": float(lookup[(method, "clean", "20250611")]["q100"]), "q100_0618": float(lookup[(method, "clean", "20250618")]["q100"]), "rmse_minus20": float(lookup[(method, "-20", "two_day")]["rmse"]), "fp_minus20": int(lookup[(method, "-20", "two_day")]["fp"]), "fn_minus20": int(lookup[(method, "-20", "two_day")]["fn"])})
        noisy = [lookup[(method, n, "two_day")] for n in ["0", "-4", "-8", "-12", "-16", "-20"]]
        noise_aggregate.append({"method": method, "mean_noise_rmse": float(np.mean([float(r["rmse"]) for r in noisy])), "mean_strong_noise_rmse": float(np.mean([float(r["rmse"]) for r in noisy[-3:]])), "mean_noise_fp": float(np.mean([int(r["fp"]) for r in noisy])), "mean_noise_fn": float(np.mean([int(r["fn"]) for r in noisy]))})
    save_csv("summary_comparison.csv", summaries)
    save_csv("noise_aggregates.csv", noise_aggregate)
    save_csv("candidate_metrics_all.csv", [r for path in sorted(OUT.glob('*_metrics.csv')) if path.name not in ["outer_metrics.csv", "fixed_input_ablation_metrics.csv", "candidate_metrics_all.csv"] for r in read_csv(path)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    selected = ["baseline_performance", "frequency34_HGB_rmse", "frequency34_HGB_rmse__mse_simplex", "frequency34_HGB_rmse__diversity_simplex", "frequency34_HGB_rmse__affine_ridge"]
    for index, name in enumerate(selected):
        clean = lookup[(name, "clean", "20250618")]
        axes[0].barh(index, float(clean["q100"]), color=f"C{index}", alpha=.85)
        axes[0].text(15, index, f"q100 {float(clean['q100']):.2f}; FP {clean['fp']}/120", va="center", fontsize=9)
        noises = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
        axes[1].plot(range(7), [float(lookup[(name, n, "two_day")]["rmse"]) for n in noises], marker="o", label=DISPLAY[name])
    axes[0].set(xlabel="June 18 q100 (kW/m2)", title="Clean q100 and pre-ONB false alarms", xlim=(0, 460))
    axes[0].set_yticks(range(5), ["Existing", "HGB", "MSE weights", "Diversity weights", "Affine Ridge"])
    axes[0].invert_yaxis()
    axes[1].set(xlabel="Evaluation noise (dB)", ylabel="Two-day RMSE (kW/m2)", title="Same clean-fitted models and fixed weights")
    axes[1].set_xticks(range(7), noises)
    for ax in axes:
        ax.grid(alpha=.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, 1))
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)
    base_fits = sum(read_json(a)["fit_operations"] for a in OUT.glob('*_audit.json') if a.name != "evaluation_audit.json")
    meta_fits = sum(item["kind"] == "ridge" for item in read_json(OUT / "frozen_integration.json")["methods"])
    save_json("summary.json", {"candidate_policies": 18, "model_families": 2, "representations": 3, "base_model_fit_operations": base_fits, "meta_model_fit_operations": meta_fits, "total_new_fit_operations": base_fits+meta_fits, "saved_joblib_models": len(list(OUT.glob('*.joblib'))), "external_evaluation": read_json(OUT / "evaluation_audit.json"), "verification": read_json(OUT / "verification.json"), "comparison": summaries, "noise_aggregates": noise_aggregate, "next_research_question": "Two June 11 last-stage chunks remain below threshold for every evaluated OOF base model; distinguish inadequate representation from a weak/noisy or mislabeled acoustic interval"})
    print("Summary tables, bottleneck margins and figures saved")


if __name__ == "__main__":
    main()
