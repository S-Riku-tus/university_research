"""Recompute saved results, separate input/weight effects, and render figures."""
import numpy as np
import pandas as pd

from common import (ARTIFACTS, CLEAN, CONFIG, CORE, FIVE, MEMBERS, OUT, POOLED, ROOT, THRESHOLDS,
    digest, frozen_fit, metric_rows, read_csv, read_json, save_csv, save_json, source_metadata,
    source_predictions, training_data, windows_long_path)


def main():
    audited = []
    for stage in ["stage1", "stage2", "stage3"]:
        verification = read_json(OUT / stage / "verification.json")
        assert verification["status"] == "passed"
        for item in verification["input_files"]:
            assert digest(ROOT / item["path"]) == item["sha256"]
        audited.append({"stage": stage, "verification_sha256": digest(OUT / stage / "verification.json")})
    addendum = ROOT / "configs/experiments/2026-10-07_onb_fixed_weight_control.json"
    fits = read_csv(OUT / "stage3/model_fit_audit.csv")
    assert len(fits) == 16 and fits.passed.all()
    for row in fits.itertuples():
        assert digest(OUT / "stage3/models" / row.model_artifact) == row.model_sha256
        assert digest(OUT / "stage3/models" / row.target_scaler_artifact) == row.scaler_sha256
    weights = frozen_fit()["weights"][FIVE.removeprefix("ensemble__")]
    frozen = read_json(OUT / "stage3/frozen_integration.json")["contrasts"]
    training, audit, source_oof = training_data()
    with np.load(OUT / "stage3/oof_predictions.npz") as stored:
        oof = {k: stored[k].copy() for k in frozen}
    saved = read_csv(OUT / "stage3/predictions.csv")
    fixed_metrics, losses, paired = [], [], []
    for scope, noises in [("training_oof_weight_fit_diagnostic", ["clean"]), ("outer_development", CONFIG["evaluation_noise_conditions"])]:
        for noise in noises:
            frame = training if scope.startswith("training") else source_predictions(noise)
            original = source_oof[MEMBERS].to_numpy() if scope.startswith("training") else frame[MEMBERS].to_numpy()
            source = original@np.array([weights[k] for k in MEMBERS])
            predictions = {FIVE: source}
            days = np.array([r["experiment_name"] for r in source_metadata(frame)])
            thresholds = np.array([THRESHOLDS[d] for d in days])
            y = frame.y_true.to_numpy()
            regions = {"all": np.ones(len(y), bool), "pre_onb": y < thresholds,
                "near_onb_10pct": abs(y-thresholds) <= .1*thresholds, "post_onb": y >= thresholds}
            for candidate, integration in frozen.items():
                if scope.startswith("training"):
                    value = oof[candidate]
                else:
                    rows = saved[(saved.noise == noise)&(saved.model_key == candidate)]
                    assert len(rows) == len(frame)
                    assert rows.source_wav_id.tolist() == frame.source_wav_id.tolist()
                    assert rows.chunk_index.tolist() == frame.chunk_index.tolist()
                    value = rows.prediction_kW_m2.to_numpy()*1000
                changed = original.copy()
                changed[:, MEMBERS.index(integration["replaced_member"])] = value
                fixed = changed@np.array([weights[k] for k in MEMBERS])
                refit = changed@np.array([integration["weights"][k] for k in MEMBERS])
                fixed_key, refit_key = "ensemble__fixed_weight_replace_"+candidate, "ensemble__replace_"+candidate
                predictions[fixed_key], predictions[refit_key] = fixed, refit
                if scope == "outer_development":
                    recorded = saved[(saved.noise == noise)&(saved.model_key == refit_key)].prediction_kW_m2.to_numpy()*1000
                    np.testing.assert_allclose(refit, recorded, rtol=1e-10, atol=1e-6)
                    common = (y >= thresholds)&np.all(original < thresholds[:, None], axis=1)
                    for rule, p in [("fixed_weight", fixed), ("refit_weight", refit)]:
                        paired.append({"noise": noise, "candidate": candidate, "rule": rule,
                            "common_clean_or_noise_fn_corrected": int((common & (p >= thresholds)).sum()),
                            "common_clean_or_noise_fn_remaining": int((common & (p < thresholds)).sum())})
                for region, mask in regions.items():
                    before, unchanged_weight, changed_weight = [p[mask]/1000 for p in [source, fixed, refit]]
                    truth = y[mask]/1000
                    for step, first, second in [("input_with_fixed_weights", before, unchanged_weight),
                            ("weights_after_input_replacement", unchanged_weight, changed_weight)]:
                        e, delta = first-truth, second-first
                        alignment, cost = float(2*np.mean(e*delta)), float(np.mean(delta**2))
                        mse_change = float(np.mean((second-truth)**2)-np.mean(e**2))
                        np.testing.assert_allclose(alignment+cost, mse_change, rtol=1e-10, atol=1e-8)
                        losses.append({"scope": scope, "noise": noise, "candidate": candidate, "region": region,
                            "step": step, "chunks": int(mask.sum()), "mse_change_kW_m2_squared": mse_change,
                            "error_adjustment_alignment": alignment, "adjustment_squared_cost": cost})
            fixed_metrics.extend(metric_rows(frame, predictions, noise, scope=scope))
    save_csv(OUT / "stage3/fixed_weight_control_metrics.csv", fixed_metrics)
    save_csv(OUT / "stage3/input_weight_loss_decomposition.csv", losses)
    save_csv(OUT / "stage3/fixed_weight_common_fn.csv", paired)
    # The original report rows must be identical in each follow-up scorer.
    source_metrics = read_csv(ROOT / CONFIG["source_run"] / "onb_comparison/main_metrics.csv")
    metric_columns = [k for k in source_metrics if k.endswith("_kW_m2") or k in ["r2", "r2_high", "roc_auc_cont", "pr_auc_cont", "tp", "fp", "tn", "fn", "recall", "precision", "accuracy", "f1", "fpr"]]
    checks = []
    for stage in ["stage1", "stage2", "stage3"]:
        metrics = read_csv(OUT / stage / "metrics.csv")
        if stage == "stage1":
            current = metrics[metrics.intervention == "unchanged"]
        else:
            current = metrics[(metrics.scope == "outer_development")&metrics.model_key.isin(source_metrics.model_key)]
        for row in current.to_dict("records"):
            expected = source_metrics[(source_metrics.noise == row["noise"])&(source_metrics.source_day == row["source_day"])&(source_metrics.model_key == row["model_key"])].iloc[0]
            a = np.array([row[k] for k in metric_columns], float)
            b = expected[metric_columns].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9, equal_nan=True)
        checks.append({"stage": stage, "source_metric_rows_recomputed": len(current), "passed": True})
    # Compact comparison: current five, free-ratio five, four input contrasts.
    stage2, stage3 = [read_csv(OUT / stage / "metrics.csv") for stage in ["stage2", "stage3"]]
    controls = pd.DataFrame(fixed_metrics)
    refitted = controls[controls.model_key.str.startswith("ensemble__replace_")|controls.model_key.eq(FIVE)]
    matched = refitted.merge(stage3, on=["scope", "noise", "source_day", "model_key"],
        suffixes=("_control", "_original"), validate="one_to_one")
    assert len(matched) == 120
    for column in metric_columns:
        np.testing.assert_allclose(matched[column+"_control"], matched[column+"_original"],
            rtol=1e-10, atol=1e-9, equal_nan=True)
    selected2 = stage2[(stage2.scope == "outer_development")&stage2.model_key.isin([FIVE, "ensemble__free_core_ratio_mse"])]
    selected3 = stage3[(stage3.scope == "outer_development")&stage3.model_key.str.startswith("ensemble__replace_")]
    combined = pd.concat([selected2, selected3], ignore_index=True)
    save_csv(OUT / "main_comparison.csv", combined.to_dict("records"))
    draw_figures()
    save_json(OUT / "verification.json", {"status": "passed", "stages": audited,
        "additional_model_fits": 0, "additional_model_inference": 0, "fixed_weight_control_is_post_hoc": True,
        "fixed_weight_control_protocol_sha256": digest(addendum), "fixed_weight_metric_rows": len(fixed_metrics),
        "refitted_control_metric_rows_reproduced": len(matched),
        "mse_decomposition_rows": len(losses), "baseline_checks": checks,
        "all_16_saved_models_and_scalers_hashes_verified": True,
        "source_models_defaults_and_original_results_preserved": True,
        "scope": CONFIG["report"]["scope"]})
    code_paths = [OUT / name for name in ["common.py", "stage1_common_interventions.py", "stage2_free_core_ratio.py", "stage3_input_removal.py", "analyze_followup.py"]]
    code_paths += [ROOT / "code" / name for name in ["run_ensemble_regression_onb.py",
        "utils/training/model_training.py", "utils/dataloading/acoustic_summary_features.py",
        "utils/ensemble/fixed_core_stacking.py", "utils/calculation/source_day_metrics.py"]]
    save_json(OUT / "code_fingerprint.json", [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in code_paths])
    print("[analysis] passed; source metrics reproduced, fixed-weight controls and loss identities verified")


def draw_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = read_csv(OUT / "stage1/metrics.csv")
    names = [r["name"] for r in CONFIG["stage1"]["interventions"] if r["type"] != "identity"]
    labels = ["2.1-2.5 kHz x0.1", "1.7-2.1 kHz x0.1", "2.5-2.9 kHz x0.1", "Power x0.5", "Power x2", "Time permutation"]
    models = [*MEMBERS, FIVE]
    model_labels = ["PCA + XGBRF", "CNN + Transformer", "AlexNet CNN", "HGB (34)", "ExtraTrees (34)", "Five-model ensemble"]
    arrays = []
    for noise in ["clean", "-20"]:
        data = m[(m.noise == noise)&(m.source_day == POOLED)]
        baseline = data[data.intervention == "unchanged"].set_index("model_key").rmse_all_kW_m2
        arrays.append(np.array([[data[(data.model_key == key)&(data.intervention == name)].iloc[0].rmse_all_kW_m2-baseline[key]
            for name in names] for key in models]))
    maximum = max(float(abs(a).max()) for a in arrays)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, array, noise in zip(axes, arrays, ["clean", "-20 dB"]):
        plot = ax.imshow(array, cmap="coolwarm", vmin=-maximum, vmax=maximum, aspect="auto")
        ax.set_xticks(range(6), labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(6), model_labels, fontsize=9)
        ax.set_title(noise)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{array[i,j]:+.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(plot, ax=axes, label="RMSE change (kW/m2); positive = worse", shrink=.8)
    fig.suptitle("Saved models: common raw-power interventions on all 540 paired chunks")
    fig.savefig(OUT / "stage1_effects.png", dpi=180)
    fig.savefig(OUT / "stage1_effects.svg")
    plt.close(fig)
    combined = read_csv(OUT / "main_comparison.csv")
    names2 = [FIVE, "ensemble__free_core_ratio_mse", *["ensemble__replace_"+key for key in read_json(OUT / "stage3/frozen_integration.json")["contrasts"]]]
    labels2 = ["Current five", "Free core ratios", "HGB: power5", "HGB: relative29", "ET: power5", "ET: relative29"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for noise, shift, color in [("clean", -.18, "#4375a5"), ("-20", .18, "#ce8649")]:
        rows = combined[(combined.noise == noise)&(combined.source_day == POOLED)].set_index("model_key").loc[names2]
        for ax, metric, title in zip(axes, ["rmse_all_kW_m2", "fn", "fp"], ["RMSE (kW/m2)", "False negatives / 315", "False positives / 225"]):
            bars = ax.barh(np.arange(6)+shift, rows[metric], height=.32, label="clean" if noise == "clean" else "-20 dB", color=color)
            ax.set_yticks(range(6), labels2, fontsize=9)
            ax.set_title(title)
            ax.grid(axis="x", alpha=.15)
            ax.bar_label(bars, labels=[f"{v:.2f}" if metric.startswith("rmse") else f"{v:.0f}" for v in rows[metric]], fontsize=8, padding=2)
            ax.margins(x=.2)
    for ax in axes:
        ax.invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Five-model contrasts; source original-three models retained; weights fit on clean OOF")
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(OUT / "integration_and_input_comparison.png", dpi=180)
    fig.savefig(OUT / "integration_and_input_comparison.svg")
    plt.close(fig)
    cases = read_csv(OUT / "stage1/representative_case_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, day in zip(axes, THRESHOLDS):
        chosen = cases[(cases.noise == "clean")&(cases.source_day == day)&cases.last_five_fn]
        first = chosen.sort_values(["source_wav_id", "chunk_index"]).iloc[0]
        rows = chosen[(chosen.source_wav_id == first.source_wav_id)&(chosen.chunk_index == first.chunk_index)]
        for key, label in zip(["hgb", "extra_trees", FIVE, "ensemble__original3_extra_trees"], ["HGB", "ExtraTrees", "Five", "ET4"]):
            values = rows[rows.model_key == key].set_index("intervention").loc[["unchanged", *names]].after_prediction_kW_m2
            ax.plot(range(7), values, marker="o", linewidth=1.4, label=label)
        ax.axhline(first.threshold_kW_m2, color="#777777", linestyle="--", label="Day ONB threshold")
        ax.set_xticks(range(7), ["Unchanged", *labels], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{day[:10]}: last original FN, true {first.target_kW_m2:.2f}\nchunk {int(first.chunk_index)}", fontsize=10)
        ax.set_ylabel("Predicted heat flux (kW/m2)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Prespecified endpoint cases; population statistics reported separately")
    fig.savefig(OUT / "endpoint_cases.png", dpi=180)
    fig.savefig(OUT / "endpoint_cases.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
