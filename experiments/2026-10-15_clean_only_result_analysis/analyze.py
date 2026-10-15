"""Analyze only the completed 143609 clean-only run; no fit or model inference."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from utils.calculation.source_day_metrics import source_day_metric_rows
from utils.experiment.run_helpers import open_text

RUN = ROOT / ("Pool_boiling/Subcooling_20_degrees/0.3/2025.06.11_0.3_2_6.18_0.3_3/"
              "regression_result/npy/ensemble/202610/15/"
              "onb_wc-t0611-v0611_ic3-nc_c1s_s0_e200_143609/maxfreq=3kHz")
REPORT = RUN / "onb_comparison"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]
MEMBERS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
CORE = "ensemble__original3_mse"
LEGACY = "ensemble__original3_performance"
FIVE = "ensemble__original3_hgb_extra_trees"
ET4 = "ensemble__original3_extra_trees"
HGB4 = "ensemble__original3_hgb"
EQUAL = "ensemble__simple_equal"
POOLED = "pooled_source_day_thresholds"
DAYS = {"20250611": "2025.06.11_0.3_2", "20250618": "2025.06.18_0.3_3"}
HASHES = {}


def fingerprint(path):
    with open_text(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def track(path):
    HASHES[str(path.relative_to(ROOT))] = fingerprint(path)


def read_json(path):
    track(path)
    with open_text(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path):
    track(path)
    with open_text(path, "r", encoding="utf-8-sig") as stream:
        return pd.read_csv(stream)


def save_csv(name, data):
    pd.DataFrame(data).to_csv(OUT / name, index=False, encoding="utf-8-sig")


def main():
    completion = read_json(REPORT / "verification.json")
    config = read_json(REPORT / "run_conditions.json")
    epochs = read_csv(REPORT / "fit_epoch_audit.csv")
    metrics = read_csv(REPORT / "main_metrics.csv")
    weights = read_csv(REPORT / "ensemble_weights.csv")
    assert completion["status"] == "passed" and completion["epochs_complete"]
    assert completion["reloaded_evaluation_predictions"] == 18900
    assert config["learning_policy"]["training_noise"] == "clean_only"
    assert config["learning_policy"]["evaluation_mode"] == "within_wav_chunk"
    assert config["run"]["epochs"] == 200 and config["run"]["random_seed"] == 42
    assert len(epochs) == 8 and epochs.complete.all() and epochs.epochs_completed.eq(200).all()
    assert epochs.actual_batch_size.eq(epochs.requested_batch_size).all()
    assert len(metrics) == 231 and set(metrics.noise) == set(NOISES)
    thresholds = config["thresholds"]["by_experiment"]
    lookup = metrics.set_index(["noise", "source_day", "model_key"])
    w = weights.pivot(index="strategy_name", columns="model_key", values="weight").fillna(0.)
    assert np.allclose(w.sum(axis=1), 1.) and (w.to_numpy() >= 0).all()
    identity = None
    metric_checks = reconstruction_checks = 0
    max_metric_error = max_reconstruction_error = 0.
    pairs, regions, directions, failures, near_bias, references, manifests = [], [], [], [], [], [], []
    onb_points = []

    for noise in NOISES:
        directory = RUN / ("heatflux_no_noise" if noise == "clean" else f"heatflux_reference_SNR={noise}")
        manifest = read_json(directory / "run_manifest.json")
        marker = read_json(directory / "completed.json")
        assert marker["run_hash"] == manifest["run_hash"]
        assert manifest["validation_config"]["learning_policy"]["training_noise"] == "clean_only"
        manifests.append(manifest)
        ref = read_json(directory / "fitted_state_reference_f1.json")
        references.append(ref["fit_id"])
        suffix = "no_noise" if noise == "clean" else noise
        frame = read_csv(directory / "fold_pred" / f"pred_f1_{suffix}.csv")
        assert len(frame) == 540 and frame.source_wav_id.nunique() == 36
        assert frame.groupby("source_wav_id").size().eq(15).all()
        frame["source_day"] = frame.source_wav_id.str.split("::").str[0].map(DAYS)
        assert frame.source_day.notna().all()
        ids = frame[["sample_index", "source_wav_id", "chunk_index", "y_true"]].to_records(index=False).tolist()
        if identity is None:
            identity = ids
        else:
            assert ids == identity
        model_keys = [*MEMBERS, *["ensemble__" + name for name in w.index]]
        metadata = [{"experiment_name": config["data"]["experiment_names"][0],
                     "source_experiment_name": row.source_day, "source_wav_id": row.source_wav_id}
                    for row in frame.itertuples()]
        recomputed = source_day_metric_rows(frame.y_true.to_numpy(),
            {key: frame[key].to_numpy() for key in model_keys}, metadata, thresholds, 1,
            band_frac=config["thresholds"]["onb_band_frac"])
        for row in recomputed:
            stored = lookup.loc[(noise, row["source_day"], row["model_key"])]
            for key, value in row.items():
                if key in {"fold", "model_key", "model_label", "source_day", "threshold_basis", "heat_flux_unit"}:
                    continue
                unit_key = key + "_kW_m2" if key in {
                    "rmse_all", "mae_all", "rmse_pre_onb", "rmse_onb", "mae_onb", "rmse_high",
                    "mae_high", "bias_all", "bias_pre_onb", "bias_onb", "bias_high", "q100", "g100", "onb_threshold"} else key
                current = value / 1000 if unit_key != key else value
                expected = float(stored[unit_key])
                assert np.isclose(current, expected, rtol=1e-9, atol=1e-7, equal_nan=True), (noise, row["model_key"], key)
                if np.isfinite(current):
                    max_metric_error = max(max_metric_error, abs(current-expected))
                metric_checks += 1
        for strategy in w.index:
            reconstructed = frame[MEMBERS].to_numpy() @ w.loc[strategy, MEMBERS].to_numpy()
            error = float(np.max(abs(reconstructed-frame["ensemble__"+strategy].to_numpy())))
            assert error < .001, (noise, strategy, error)
            max_reconstruction_error = max(max_reconstruction_error, error)
            reconstruction_checks += 1

        target = frame.y_true.to_numpy() / 1000
        threshold = frame.source_day.map(thresholds).to_numpy() / 1000
        actual = target >= threshold
        near = abs(target-threshold) <= .1*threshold
        exact_onb = np.isclose(target, threshold, rtol=0, atol=1e-8)
        assert np.array_equal(near, exact_onb) and exact_onb.sum() == 30
        assert frame.loc[exact_onb, "source_wav_id"].nunique() == 2
        for key in model_keys:
            positive = frame[key].to_numpy()/1000 >= threshold
            onb_points.append({"noise": noise, "model_key": key, "n_at_recorded_onb": int(exact_onb.sum()),
                "detected_onb_chunks": int(np.sum(exact_onb & positive)),
                "missed_onb_chunks": int(np.sum(exact_onb & ~positive)),
                "detection_fraction_at_recorded_onb": float(positive[exact_onb].mean())})
        five = frame[FIVE].to_numpy()/1000
        five_positive = five >= threshold
        original = frame[CORE].to_numpy()/1000
        for baseline in [LEGACY, CORE, ET4, HGB4, "extra_trees", "hgb", EQUAL]:
            before = frame[baseline].to_numpy()/1000
            positive = before >= threshold
            b = lookup.loc[(noise, POOLED, baseline)]
            f = lookup.loc[(noise, POOLED, FIVE)]
            pairs.append({"noise": noise, "baseline_key": baseline,
                "baseline_rmse_kW_m2": b.rmse_all_kW_m2, "five_rmse_kW_m2": f.rmse_all_kW_m2,
                "rmse_reduction_kW_m2": b.rmse_all_kW_m2-f.rmse_all_kW_m2,
                "rmse_reduction_percent": 100*(1-f.rmse_all_kW_m2/b.rmse_all_kW_m2),
                "near_rmse_change_kW_m2": f.rmse_onb_kW_m2-b.rmse_onb_kW_m2,
                "baseline_fn": int(b.fn), "five_fn": int(f.fn), "baseline_fp": int(b.fp), "five_fp": int(f.fp),
                "rescued_fn": int(np.sum(actual & ~positive & five_positive)),
                "new_fn": int(np.sum(actual & positive & ~five_positive)),
                "removed_fp": int(np.sum(~actual & positive & ~five_positive)),
                "new_fp": int(np.sum(~actual & ~positive & five_positive))})
        original_error, adjustment = original-target, five-original
        for stage, mask in [("all", np.ones(len(frame), bool)), ("pre_onb", ~actual),
                            ("near_onb", near), ("onb_and_above", actual)]:
            before = float(np.sqrt(np.mean(original_error[mask]**2)))
            after = float(np.sqrt(np.mean((five[mask]-target[mask])**2)))
            regions.append({"noise": noise, "region": stage, "n": int(mask.sum()),
                "original3_rmse_kW_m2": before, "five_rmse_kW_m2": after,
                "mse_change_kW2_m4": after**2-before**2,
                "adjustment_cross_term": float(2*np.mean(original_error[mask]*adjustment[mask])),
                "adjustment_square_term": float(np.mean(adjustment[mask]**2))})
        blocks = frame[[CORE, "hgb", "extra_trees"]].to_numpy()/1000-target[:, None]
        opposite = (blocks < 0).any(axis=1) & (blocks > 0).any(axis=1)
        improvement = original_error**2-(five-target)**2
        for group, mask in [("opposite_signs", opposite), ("same_direction_or_zero", ~opposite)]:
            contribution = float(improvement[mask].sum()/len(frame))
            directions.append({"noise": noise, "group": group, "n": int(mask.sum()),
                "improved_chunks": int(np.sum(improvement[mask] > 0)),
                "mse_improvement_contribution": contribution,
                "share_of_total_mse_improvement_percent": 100*contribution/float(improvement.mean())})
        for key in MEMBERS:
            near_bias.append({"noise": noise, "model_key": key,
                "weight_in_five": float(w.loc[FIVE.removeprefix("ensemble__"), key]),
                "near_bias_kW_m2": float(np.mean((frame[key].to_numpy()/1000-target)[near])),
                "weighted_near_bias_kW_m2": float(w.loc[FIVE.removeprefix("ensemble__"), key] *
                    np.mean((frame[key].to_numpy()/1000-target)[near]))})
        if noise in {"clean", "-20"}:
            for i in np.flatnonzero(actual != five_positive):
                failure = {"noise": noise, "sample_index": int(frame.sample_index.iloc[i]),
                    "source_day": frame.source_day.iloc[i], "source_wav_id": frame.source_wav_id.iloc[i],
                    "chunk_index": int(frame.chunk_index.iloc[i]), "true_q_kW_m2": target[i],
                    "threshold_kW_m2": threshold[i], "error": "FN" if actual[i] else "FP"}
                failure.update({key+"_prediction_kW_m2": frame[key].iloc[i]/1000 for key in [*MEMBERS, CORE, ET4, FIVE]})
                failure["positive_single_models"] = ";".join(key for key in MEMBERS if frame[key].iloc[i]/1000 >= threshold[i])
                failures.append(failure)

    assert len(set(references)) == 1 and len({m["run_hash"] for m in manifests}) == 1
    pooled = metrics[metrics.source_day.eq(POOLED)].copy()
    paired = pd.DataFrame(pairs)
    clean_lookup = pooled[pooled.noise.eq("clean")].set_index("model_key")
    degradation = []
    for row in pooled.itertuples():
        baseline = clean_lookup.loc[row.model_key]
        degradation.append({"noise": row.noise, "model_key": row.model_key,
            "rmse_kW_m2": row.rmse_all_kW_m2, "clean_rmse_kW_m2": baseline.rmse_all_kW_m2,
            "rmse_increase_kW_m2": row.rmse_all_kW_m2-baseline.rmse_all_kW_m2,
            "near_rmse_kW_m2": row.rmse_onb_kW_m2, "fn": int(row.fn), "fp": int(row.fp),
            "r2": row.r2, "recall": row.recall, "f1": row.f1})
    save_csv("pooled_metrics.csv", pooled)
    save_csv("endpoint_comparison.csv", pooled[pooled.noise.isin(["clean", "-20"])])
    save_csv("all_metrics_by_source_day.csv", metrics)
    save_csv("five_vs_baselines.csv", paired)
    save_csv("noise_degradation.csv", degradation)
    save_csv("regional_loss_changes.csv", regions)
    save_csv("error_direction_contributions.csv", directions)
    save_csv("near_bias_contributions.csv", near_bias)
    save_csv("recorded_onb_point_detection.csv", onb_points)
    save_csv("five_failure_cases.csv", failures)
    save_csv("weights.csv", weights)
    save_csv("q100_by_source_day.csv", metrics[~metrics.source_day.eq(POOLED)])
    failure_frame = pd.DataFrame(failures)
    save_csv("five_failure_groups.csv", failure_frame.groupby(
        ["noise", "error", "source_day", "true_q_kW_m2", "threshold_kW_m2"]).size().reset_index(name="chunks"))

    curves = [
        (LEGACY, "Original3 performance", "#D55E00", "s", ":"),
        (CORE, "Original3 MSE", "#0072B2", "o", "--"),
        (ET4, "Original3 + ET", "#009E73", "^", "-."),
        ("hgb", "HGB", "#CC79A7", "v", ":"),
        ("extra_trees", "ExtraTrees", "#56B4E9", "P", "--"),
        (FIVE, "Original3 + HGB + ET", "#222222", "D", "-"),
    ]
    for stem, specifications in [
        ("regression_noise_profiles", [("rmse_all_kW_m2", "Overall RMSE (kW/m²)"),
                                       ("rmse_onb_kW_m2", "Near-ONB RMSE (kW/m²)")]),
        ("onb_error_profiles", [("fn", "Missed positive chunks (FN)"),
                                ("fp", "False alarms (FP)")])]:
        with plt.rc_context({"font.family": "Times New Roman", "font.size": 14}):
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.9))
            for ax, (column, label) in zip(axes, specifications):
                for key, title, color, marker, style in curves:
                    rows = pooled[pooled.model_key.eq(key)].set_index("noise").loc[NOISES]
                    ax.plot(range(7), rows[column], label=title, color=color, marker=marker,
                            linestyle=style, linewidth=3 if key == FIVE else 1.8, markersize=6)
                ax.set_xticks(range(7), ["Clean", "0", "−4", "−8", "−12", "−16", "−20"])
                ax.set_xlabel("Reference SNR (dB)")
                ax.set_ylabel(label)
                ax.set_ylim(bottom=0)
                if column in {"fn", "fp"}:
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.grid(alpha=.2)
            handles, titles = axes[0].get_legend_handles_labels()
            fig.legend(handles, titles, loc="lower center", ncol=3, frameon=False, fontsize=12)
            fig.tight_layout(rect=(0, .15, 1, 1))
            for extension in ["png", "svg"]:
                fig.savefig(OUT / f"{stem}.{extension}", dpi=180)
            plt.close(fig)

    earlier = read_json(ROOT / "experiments/2026-10-15_bar_presentation_views/verification.json")
    current_clean_csv = RUN / "heatflux_no_noise/metrics_summary_no_noise.csv"
    track(current_clean_csv)
    same_clean_summary = fingerprint(current_clean_csv) == earlier["source_sha256"]
    noise_wins = {key: int(paired[paired.baseline_key.eq(key)].rmse_reduction_kW_m2.gt(0).sum())
                  for key in paired.baseline_key.unique()}
    summary = {"run_hash": manifests[0]["run_hash"], "fit_id": references[0],
        "rmse_wins_out_of_7": noise_wins, "clean_csv_identical_to_prior_200epoch_record": same_clean_summary,
        "clean_only_scope": True, "matched_read": False,
        "five_clean_rmse": float(clean_lookup.loc[FIVE].rmse_all_kW_m2),
        "five_minus20_rmse": float(lookup.loc[("-20", POOLED, FIVE)].rmse_all_kW_m2),
        "clean_fn_cases_with_any_positive_single": int(((failure_frame.noise=="clean") &
            (failure_frame.error=="FN") & failure_frame.positive_single_models.ne("")).sum()),
        "clean_fn_all_singles_negative": int(((failure_frame.noise=="clean") &
            (failure_frame.error=="FN") & failure_frame.positive_single_models.eq("")).sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(fingerprint(ROOT / path) == digest for path, digest in HASHES.items())
    verification = {"status": "passed", "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": str(RUN.relative_to(ROOT)), "run_hash": summary["run_hash"], "fit_id": references[0],
        "conditions_completed": 7, "evaluation_chunks_each": 540, "evaluation_source_wavs": 36,
        "same_evaluation_identities_and_targets": True, "same_clean_fit_and_weights": True,
        "source_day_metric_rows": 231, "numeric_metric_checks": metric_checks,
        "near_band_contains_only_recorded_onb_points": True,
        "max_metric_difference": max_metric_error, "ensemble_reconstruction_checks": reconstruction_checks,
        "max_reconstruction_difference_W_m2": max_reconstruction_error,
        "input_hashes": HASHES, "sources_unchanged": True, "matched_read": False,
        "new_fits": 0, "new_model_inference": 0, "model_binaries_loaded": False, "raw_arrays_loaded": False}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    conditions = {"source_run": verification["source_run"], "run_hash": summary["run_hash"],
        "fit_id": references[0], "validation_config": config,
        "scope": "Current completed clean_only run only; interrupted matched run excluded",
        "evaluation_unit": "One-second held-out chunks within known WAVs, 1620 train / 540 test",
        "primary_threshold_basis": "Source-day thresholds; pooled AUC uses raw prediction scores",
        "uncertainty": "One seed and one chunk holdout on reused development recordings; no independent replication claim"}
    (OUT / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "metric_rows": 231, "reconstructed_ensembles": reconstruction_checks,
                      "sources_unchanged": True, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
