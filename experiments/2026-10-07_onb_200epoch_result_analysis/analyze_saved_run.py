"""Audit and describe the completed 200-epoch run; no model fits or inference.

All derived heat-flux quantities use kW/m2. Pointwise FN upper bounds use
labels only for retrospective diagnosis, never to select deployable weights.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from utils.calculation.source_day_metrics import source_day_metric_rows
from utils.experiment.run_helpers import open_text, windows_long_path

OUT = Path(__file__).resolve().parent
RUN = ROOT / ("Pool_boiling/Subcooling_20_degrees/0.3/"
    "2025.06.11_0.3_2_6.18_0.3_3/regression_result/npy/ensemble/202610/06/"
    "onb_wc-t0611-v0611_ic3-nc_c1s_s0_e200_193305/maxfreq=3kHz")
REPORT = RUN / "onb_comparison"
OLD = ROOT / "experiments/2026-10-06_onb_main_comparison_ready/saved_main_tables/main_metrics.csv"
MEMBERS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
ET4 = "ensemble__original3_extra_trees"
HGB4 = "ensemble__original3_hgb"
DAYS = {"20250611": "2025.06.11_0.3_2", "20250618": "2025.06.18_0.3_3"}
POOLED = "pooled_source_day_thresholds"


def read_json(path):
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path):
    with open_text(path, "r", encoding="utf-8-sig", newline="") as stream:
        return pd.read_csv(stream)


def save_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")


def main():
    verification = read_json(REPORT / "verification.json")
    config = read_json(REPORT / "run_conditions.json")
    epochs = read_csv(REPORT / "fit_epoch_audit.csv")
    reloads = read_csv(REPORT / "fitted_reload_checks.csv")
    reconstruction = read_csv(REPORT / "ensemble_reconstruction_checks.csv")
    main_metrics = read_csv(REPORT / "main_metrics.csv")
    clean = RUN / "heatflux_no_noise"
    fit = read_json(clean / "ensemble_training_oof_fit_f1.json")
    weights = np.array([fit["weights"][FIVE.removeprefix("ensemble__")][k] for k in MEMBERS])
    core_weights = np.array([fit["weights"][CORE.removeprefix("ensemble__")][k] for k in MEMBERS[:3]])
    thresholds = config["thresholds"]["by_experiment"]
    assert verification["status"] == "passed" and not verification["smoke_test"]
    assert config["run"]["epochs"] == 200 and config["run"]["folds"] == 3
    assert len(epochs) == 8 and epochs.complete.all() and (epochs.epochs_completed == 200).all()
    assert (epochs.actual_batch_size == epochs.requested_batch_size).all()
    assert len(reloads) == 35 and reloads.passed.all() and (reloads.samples == 540).all()
    assert len(reconstruction) == 42 and reconstruction.passed.all()
    assert len(main_metrics) == 231
    assert np.isclose(weights.sum(), 1) and np.all(weights > 0)
    assert np.isclose(weights[:3].sum(), .25)
    assert np.allclose(weights[:3], .25*core_weights)
    inputs = [REPORT / f for f in ["verification.json", "run_conditions.json", "main_metrics.csv",
        "fit_epoch_audit.csv", "fitted_reload_checks.csv", "ensemble_reconstruction_checks.csv"]]
    inputs += [clean / "ensemble_training_oof_fit_f1.json", clean / "ensemble_training_oof_f1.csv", OLD]
    missed, cases, paired, wav_errors, losses, endpoints, bootstraps, completed = [], [], [], [], [], [], [], []
    max_prediction_difference, max_metric_difference = 0., 0.
    identity = None
    group_rows = None
    bootstrap_indices = None
    rng = np.random.default_rng(20261007)
    # Day-stratified, paired WAV bootstrap: keep every chunk of a drawn WAV.
    # These are descriptive intervals on reused development data, not a test
    # of architecture superiority or uncertainty on the whole training process.
    for source in verification["sources"]:
        folder = RUN / source["noise_dir_name"]
        noise = "clean" if source["noise_dir_name"] == "heatflux_no_noise" else source["noise_dir_name"].split("SNR=")[1]
        pred_path = folder / "fold_pred" / ("pred_f1_no_noise.csv" if noise == "clean" else f"pred_f1_{noise}.csv")
        inputs += [pred_path, folder / "completed.json", folder / "split_manifest.json"]
        completion = read_json(folder / "completed.json")
        assert completion["fit_ids"] == [source["fit_id"]]
        assert completion["run_hash"] == read_json(clean / "completed.json")["run_hash"]
        completed.append({"noise": noise, "completion_record": completion})
        split = read_json(folder / "split_manifest.json")["folds"][0]
        assert split["n_training_chunks"] == 1620 and split["n_evaluation_chunks"] == 540
        assert source["fit_id"] == verification["sources"][0]["fit_id"]
        frame = read_csv(pred_path)
        current_identity = frame[["sample_index", "source_wav_id", "chunk_index", "y_true"]]
        if identity is None:
            identity = current_identity.copy()
            group_rows = [np.flatnonzero(frame.source_wav_id.to_numpy() == k) for k in sorted(frame.source_wav_id.unique())]
            assert len(group_rows) == 36 and all(len(i) == 15 for i in group_rows)
            groups_by_day = [np.array([j for j, i in enumerate(group_rows) if frame.source_wav_id.iloc[i[0]].startswith(d)]) for d in DAYS]
            assert all(len(g) == 18 for g in groups_by_day)
            bootstrap_indices = np.concatenate([rng.choice(g, (5000, len(g)), replace=True) for g in groups_by_day], axis=1)
        else:
            pd.testing.assert_frame_equal(current_identity, identity)
        days = frame.source_wav_id.str[:8].map(DAYS).to_numpy()
        assert pd.notna(days).all()
        y = frame.y_true.to_numpy()/1000
        threshold = np.array([thresholds[d]/1000 for d in days])
        actual = y >= threshold
        matrix = frame[MEMBERS].to_numpy()/1000
        p = frame[FIVE].to_numpy()/1000
        difference = float(np.max(np.abs(matrix@weights-p)))
        max_prediction_difference = max(max_prediction_difference, difference)
        np.testing.assert_allclose(matrix@weights, p, rtol=1e-12, atol=1e-9)
        # Independently recompute every saved source-day row from predictions.
        metadata = [{"experiment_name": d, "source_wav_id": w} for d, w in zip(days, frame.source_wav_id)]
        cols = [k for k in frame.columns if k in MEMBERS or k.startswith("ensemble__")]
        recomputed = source_day_metric_rows(frame.y_true.to_numpy(),
            {k: frame[k].to_numpy() for k in cols}, metadata, thresholds, 1)
        numeric = ["n", "n_pre_onb", "n_post_onb", "n_onb", "n_source_wavs", "tp", "fp", "tn", "fn",
            "recall", "precision", "accuracy", "f1", "fpr", "r2", "r2_high", "roc_auc_cont", "pr_auc_cont"]
        errors = ["rmse_all", "mae_all", "rmse_pre_onb", "rmse_high", "mae_high", "rmse_onb", "mae_onb",
            "bias_all", "bias_pre_onb", "bias_onb", "bias_high", "q100", "g100", "onb_threshold"]
        for row in recomputed:
            saved = main_metrics[(main_metrics.noise == noise)&(main_metrics.model_key == row["model_key"])&(main_metrics.source_day == row["source_day"])].iloc[0]
            a = np.array([row[k] for k in numeric]+[row[k]/1000 for k in errors])
            b = saved[numeric+[k+"_kW_m2" for k in errors]].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9, equal_nan=True)
            max_metric_difference = max(max_metric_difference, float(np.nanmax(np.abs(a-b))))
        blocks = np.column_stack([matrix[:, :3]@core_weights, matrix[:, 3:]])
        upper = blocks@np.array([.25, .05, .05])+.65*blocks.max(axis=1)
        fn = actual & (p < threshold)
        common = fn & np.all(matrix < threshold[:, None], axis=1)
        locked = fn & ~common & (blocks.max(axis=1) < threshold)
        floor = fn & ~common & ~locked & (upper < threshold)
        possible = fn & ~(common | locked | floor)
        assert np.array_equal(fn, common | locked | floor | possible)
        for day in [POOLED, *DAYS.values()]:
            mask = np.ones(len(y), bool) if day == POOLED else days == day
            missed.append({"noise": noise, "source_day": day, "fn": int((fn & mask).sum()),
                "all_five_common_fn": int((common & mask).sum()), "locked_core_unreachable_fn": int((locked & mask).sum()),
                "lower_bounds_unreachable_fn": int((floor & mask).sum()), "pointwise_possible_fn": int((possible & mask).sum())})
        for baseline in [CORE, ET4, HGB4, "extra_trees"]:
            before = frame[baseline].to_numpy()/1000
            before_fn, before_fp = actual & (before < threshold), ~actual & (before >= threshold)
            fp = ~actual & (p >= threshold)
            for day in [POOLED, *DAYS.values()]:
                mask = np.ones(len(y), bool) if day == POOLED else days == day
                paired.append({"noise": noise, "source_day": day, "baseline": baseline,
                    "corrected_fn": int((before_fn & ~fn & mask).sum()), "added_fn": int((~before_fn & fn & mask).sum()),
                    "retained_fn": int((before_fn & fn & mask).sum()), "corrected_fp": int((before_fp & ~fp & mask).sum()),
                    "added_fp": int((~before_fp & fp & mask).sum()), "retained_fp": int((before_fp & fp & mask).sum())})
            for i in np.flatnonzero((before_fn != fn) | (before_fp != fp)):
                cases.append({"noise": noise, "baseline": baseline, "source_day": days[i],
                    "source_wav_id": frame.source_wav_id.iloc[i], "chunk_index": int(frame.chunk_index.iloc[i]),
                    "target_kW_m2": y[i], "threshold_kW_m2": threshold[i], "baseline_prediction_kW_m2": before[i],
                    "five_prediction_kW_m2": p[i], "hgb_prediction_kW_m2": matrix[i, 3], "extra_trees_prediction_kW_m2": matrix[i, 4],
                    "change": "corrected_fn" if before_fn[i] and not fn[i] else "added_fn" if fn[i] and not before_fn[i]
                        else "corrected_fp" if before_fp[i] and not fp[i] else "added_fp"})
            for indices in group_rows:
                wav_errors.append({"noise": noise, "baseline": baseline, "source_day": days[indices[0]],
                    "source_wav_id": frame.source_wav_id.iloc[indices[0]], "chunks": len(indices),
                    "baseline_mse": float(np.mean((before[indices]-y[indices])**2)),
                    "five_mse": float(np.mean((p[indices]-y[indices])**2)),
                    "delta_mse": float(np.mean((p[indices]-y[indices])**2-(before[indices]-y[indices])**2))})
            if noise in ["clean", "-20"]:
                components = []
                for indices in group_rows:
                    components.append([np.mean((p[indices]-y[indices])**2), np.mean((before[indices]-y[indices])**2),
                        np.mean(abs(p[indices]-y[indices])), np.mean(abs(before[indices]-y[indices])),
                        np.sum(fn[indices]), np.sum(before_fn[indices])])
                c = np.array(components)
                resampled = c[bootstrap_indices].mean(axis=1)
                for metric, point, distribution in [
                    ("rmse_kW_m2", np.sqrt(c[:, 0].mean())-np.sqrt(c[:, 1].mean()), np.sqrt(resampled[:, 0])-np.sqrt(resampled[:, 1])),
                    ("mae_kW_m2", c[:, 2].mean()-c[:, 3].mean(), resampled[:, 2]-resampled[:, 3]),
                    ("fn_count", c[:, 4].sum()-c[:, 5].sum(), 36*(resampled[:, 4]-resampled[:, 5]))]:
                    low, high = np.quantile(distribution, [.025, .975])
                    bootstraps.append({"noise": noise, "baseline": baseline, "metric": metric,
                        "five_minus_baseline": point, "descriptive_2p5": low, "descriptive_97p5": high,
                        "wav_clusters": 36, "replicates": 5000, "stratified_by_day": True})
        for day in DAYS.values():
            mask = days == day
            day_fn = fn & mask
            last_flux = float(y[day_fn].max()) if day_fn.any() else np.nan
            last = day_fn & (y == last_flux)
            q = main_metrics[(main_metrics.noise == noise)&(main_metrics.model_key == FIVE)&(main_metrics.source_day == day)].iloc[0].q100_kW_m2
            endpoints.append({"noise": noise, "source_day": day, "q100_kW_m2": q,
                "last_fn_flux_kW_m2": last_flux, "fn_at_last_flux": int(last.sum()),
                "all_five_common_at_last_flux": int((last & common).sum()), "locked_core_unreachable_at_last_flux": int((last & locked).sum()),
                "lower_bounds_unreachable_at_last_flux": int((last & floor).sum()), "pointwise_possible_at_last_flux": int((last & possible).sum())})
        for baseline in [ET4, "extra_trees"]:
            before = frame[baseline].to_numpy()/1000
            for region, mask in {"all": np.ones(len(y), bool), "pre_onb": ~actual,
                    "near_onb_10pct": abs(y-threshold) <= .1*threshold, "post_onb": actual}.items():
                e = before[mask]-y[mask]
                delta = p[mask]-before[mask]
                alignment, cost = float(2*np.mean(e*delta)), float(np.mean(delta**2))
                change = float(np.mean((p[mask]-y[mask])**2)-np.mean(e**2))
                np.testing.assert_allclose(alignment+cost, change, rtol=1e-10, atol=1e-8)
                losses.append({"noise": noise, "baseline": baseline, "region": region, "chunks": int(mask.sum()),
                    "five_minus_baseline_mse": change, "error_adjustment_alignment": alignment, "adjustment_squared_cost": cost})
    # Clean OOF is the data used to fit the weights: keep it separate from outer performance.
    oof = read_csv(clean / "ensemble_training_oof_f1.csv")
    assert len(oof) == 1620 and (oof.groupby("source_wav_group").size() == 45).all()
    y = oof.y_true.to_numpy()/1000
    oof_rmse = []
    for profile, w in fit["weights"].items():
        p = (oof[MEMBERS].to_numpy()/1000)@np.array([w[k] for k in MEMBERS])
        rmse = float(np.sqrt(np.mean((p-y)**2)))
        np.testing.assert_allclose(rmse, fit["diagnostics"][profile]["oof_fit_rmse_w_m2"]/1000, atol=1e-8)
        oof_rmse.append({"model_key": "ensemble__"+profile, "scope": "clean_training_oof_weight_fit_diagnostic", "rmse_kW_m2": rmse})
    combined = pd.concat([main_metrics, read_csv(OLD)], ignore_index=True)
    selected = combined[(combined.source_day == POOLED)&combined.model_key.isin([CORE, HGB4, ET4, FIVE, "hgb", "extra_trees"])]
    save_csv("comparison_across_runs.csv", selected.to_dict("records"))
    for name, rows in [("paired_detection_changes.csv", paired), ("detection_change_cases.csv", cases),
            ("remaining_false_negatives.csv", missed), ("wav_error_changes.csv", wav_errors),
            ("loss_decomposition.csv", losses), ("q100_last_false_negative.csv", endpoints),
            ("wav_cluster_bootstrap.csv", bootstraps), ("oof_weight_fit_diagnostics.csv", oof_rmse)]:
        save_csv(name, rows)
    conditions = {k: config[k] for k in ["run", "learning_policy", "features", "explainability"]}
    conditions.update({"source_frequency_directory": str(RUN.relative_to(ROOT)),
        "source_report_directory": str(REPORT.relative_to(ROOT)), "source_day_thresholds_W_m2":
        {d: thresholds[d] for d in DAYS.values()}, "analysis_date": "2026-10-07",
        "parameter_sets": config["models"]["parameter_sets"],
        "comparison_note": "Current CNN fits use 200 epochs; previous seed43/44 use 150. Keep runs separate."})
    (OUT / "conditions.json").write_text(json.dumps(conditions, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    noises = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
    current = selected[selected.seed == 42]
    for key, label, color in [(CORE, "Original 3 (MSE)", "#555555"),
            (HGB4, "Original 3 + HGB", "#c17a16"), (ET4, "Original 3 + ExtraTrees", "#247b61"),
            (FIVE, "Original 3 + HGB + ExtraTrees", "#385aba")]:
        values = current[current.model_key == key].set_index("noise").loc[noises]
        for ax, metric, title in zip(axes, ["rmse_all_kW_m2", "fn", "fp"],
                ["RMSE (kW/m2)", "False negatives (315 positives)", "False positives (225 negatives)"]):
            ax.plot(range(7), values[metric], label=label, color=color, marker="o", linewidth=1.5, markersize=4)
            ax.set_xticks(range(7), noises)
            ax.set_xlabel("Reference SNR (dB); clean = no added noise")
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=.2)
    axes[-1].set_yticks([0, 1])
    axes[-1].set_ylim(-.1, 1.1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Seed 42 / 200 CNN epochs / source-day ONB thresholds / paired 540 chunks", fontsize=11)
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)
    hashes = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(Path(windows_long_path(p)).read_bytes()).hexdigest()} for p in inputs]
    result = {"status": "passed", "completed_full_200_epoch_run": True, "nominal_150_protocol_match": False,
        "new_model_fits": 0, "new_model_predictions": 0, "weights_changed": False,
        "seed": 42, "cnn_epochs": 200, "internal_folds": 3, "source_wavs": 36,
        "training_chunks": 1620, "evaluation_chunks_per_noise": 540, "positive_chunks": 315, "negative_chunks": 225,
        "evaluated_noise_conditions": 7, "reloaded_model_predictions": int(reloads.samples.sum()), "audited_cnn_fits": len(epochs),
        "max_saved_reload_difference_W_m2": float(reloads.max_abs_prediction_difference_w_m2.max()),
        "max_five_reconstruction_difference_kW_m2": max_prediction_difference,
        "recomputed_main_metric_rows": len(main_metrics), "max_main_metric_numeric_difference": max_metric_difference,
        "bootstrap_scope": "descriptive paired WAV resampling within each day; reused development data; no architecture selection",
        "across_run_scope": "seed42/200 epochs versus seed43,44/150 epochs; not an epoch-effect experiment or independent new recordings",
        "fn_upper_bound_scope": "per-sample oracle upper bound; no guarantee of one common weight vector, FP safety, or q100 improvement",
        "completion_records": completed, "input_files": hashes}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "completed_full_200_epoch_run", "new_model_fits",
        "reloaded_model_predictions", "max_five_reconstruction_difference_kW_m2", "max_main_metric_numeric_difference"]}))


if __name__ == "__main__":
    main()
