"""Audit the user's completed new-day within-WAV run using saved predictions.

No fitting, model inference, input regeneration, or changes to the source run.
Derived heat flux and errors are in kW/m2; original CSVs use W/m2.
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
from utils.calculation.onb_comparison_report import comparison_tables

OUT = Path(__file__).resolve().parent
RUN = ROOT / ("Pool_boiling/Subcooling_20_degrees/0.3/2026.10.07_0.3_1/"
    "regression_result/npy/ensemble/202610/14/"
    "onb_wc-t1007-v1007_ic3-nc_c1s_s0_e200_114213/maxfreq=3kHz")
DAY = "2026.10.07_0.3_1"
MEMBERS = ["randomforest", "conformer", "alexnet", "hgb", "extra_trees"]
CORE = "ensemble__original3_mse"
FIVE = "ensemble__original3_hgb_extra_trees"
ET4 = "ensemble__original3_extra_trees"
HGB4 = "ensemble__original3_hgb"
NOISES = ["clean", "0", "-4", "-8", "-12", "-16", "-20"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    clean = RUN / "heatflux_no_noise"
    manifest = read_json(clean / "run_manifest.json")
    config = manifest["validation_config"]
    context = manifest["learning_context"]
    assert context["evaluation_mode"] == "within_wav_chunk"
    assert context["training_noise"] == "clean_only"
    assert context["training_experiments"] == context["evaluation_experiments"] == [DAY]
    assert config["run"]["epochs"] == 200 and not config["run"]["smoke_test"]
    assert config["run"]["internal_validation_split"] == "chunk_kfold"
    threshold = manifest["dataset"]["threshold"]
    thresholds = {DAY: threshold}
    fit = read_json(clean / "ensemble_training_oof_fit_f1.json")
    saved_weights = read_json(clean / "fitted_state/fold1/ensemble_weights.json")
    weights = {k: v["weights"] for k, v in saved_weights.items()}
    assert all(weights["ensemble__" + k] == v for k, v in fit["weights"].items())
    models = MEMBERS + list(weights)
    assert not fit["outer_labels_used"]
    oof = pd.read_csv(clean / "ensemble_training_oof_f1.csv")
    assert len(oof) == 495 and oof.training_row_index.nunique() == 495
    coverage = np.zeros(495, dtype=int)
    epochs = []
    for split in fit["splits"]:
        assert split["fit_chunks"] == 330 and split["validation_chunks"] == 165
        assert split["shared_samples"] == 0 and split["shared_source_wavs"] == 11
        train, test = set(split["fit_indices"]), set(split["validation_indices"])
        assert train.isdisjoint(test) and train | test == set(range(495))
        coverage[list(test)] += 1
        for key, values in split["model_training"].items():
            epochs.append({"fit_scope": "internal", "fold": split["fold"], "model_key": key, **values})
    assert len(fit["splits"]) == 3 and np.all(coverage == 1)
    artifacts = read_json(clean / "fitted_state/fold1/artifact_manifest.json")
    for values in artifacts["preprocessors"].values():
        artifact = values.get("artifact", values)
        if "sha256" in artifact:
            assert sha(clean / "fitted_state/fold1" / artifact["path"]) == artifact["sha256"]
    artifact_hashes = []
    reloads = []
    for key, values in artifacts["models"].items():
        artifact = values["artifact"]
        path = clean / "fitted_state/fold1" / artifact["path"]
        assert sha(path) == artifact["sha256"] and path.stat().st_size == artifact["bytes"]
        artifact_hashes.append({"model_key": key, **artifact})
        check = values["evaluation_verification"]
        assert check["performed"] and check["passed"] and check["total_predictions"] == 1155
        for condition in check["conditions"]:
            assert condition["passed"] and condition["samples"] == 165
            reloads.append({"model_key": key, **condition})
    assert len(reloads) == 35
    base_inputs = [clean / name for name in ["run_manifest.json", "split_manifest.json",
        "ensemble_training_oof_fit_f1.json", "ensemble_training_oof_f1.csv",
        "fitted_state/fold1/artifact_manifest.json", "fitted_state/fold1/ensemble_weights.json"]]
    rows, stage_rows, changes, endpoints, cases, correlations, losses, snrs = [], [], [], [], [], [], [], []
    completed, source_hashes = [], []
    max_reconstruction, max_metric_difference = 0., 0.
    identity = None
    frames = {}
    for noise in NOISES:
        tag = "no_noise" if noise == "clean" else noise
        folder = RUN / ("heatflux_no_noise" if noise == "clean" else "heatflux_reference_SNR=" + noise)
        paths = [folder / name for name in ["completed.json", "run_manifest.json", "split_manifest.json",
            "ensemble_training_oof_fit_f1.json", f"fold_pred/pred_f1_{tag}.csv", "metrics_by_source_day.csv"]]
        completion, current_manifest, split = [read_json(p) for p in paths[:3]]
        assert completion["run_hash"] == manifest["run_hash"]
        assert completion["fit_ids"] == [artifacts["fit_id"]]
        assert current_manifest["learning_context"]["training_noise_dir"] == "heatflux_no_noise"
        assert read_json(paths[3])["weights"] == fit["weights"]
        outer = split["folds"][0]
        assert outer["n_training_chunks"] == 495 and outer["n_evaluation_chunks"] == 165
        assert outer["fit_id"] == artifacts["fit_id"]
        assert len(set(outer["training_wav_groups"]) & set(outer["evaluation_wav_groups"])) == 11
        if noise == "clean":
            selection = read_json(folder / "training_selection_fold1.json")
            assert not selection["enabled"] and selection["n_before"] == selection["n_after"] == 495
            for key in ["conformer", "alexnet"]:
                values = {"requested_epochs": artifacts["model_epochs"][key],
                    **artifacts["models"][key]["training_history"]}
                epochs.append({"fit_scope": "final", "fold": 1, "model_key": key, **values})
        frame = pd.read_csv(paths[4])
        frames[noise] = frame
        assert len(frame) == 165 and frame.source_wav_id.nunique() == 11
        assert (frame.groupby("source_wav_id").size() == 15).all()
        assert frame.sample_index.tolist() == outer["evaluation_sample_indices"]
        current_identity = frame[["sample_index", "source_wav_id", "chunk_index", "y_true"]]
        if identity is None:
            identity = current_identity
        else:
            pd.testing.assert_frame_equal(identity, current_identity)
        data_manifest = Path(current_manifest["dataset"]["data_path"]) / "chunk_manifest.csv"
        generated = pd.read_csv(data_manifest).set_index("sample_filename")
        assert len(generated) == 660 and generated.index.is_unique
        selected = generated.loc[frame.sample_filename].copy()
        assert np.allclose(selected.heat_flux.to_numpy(), frame.y_true.to_numpy())
        assert np.allclose(selected.fixed_reference_signal_rms, .011917774650702801)
        if noise != "clean":
            assert (selected.requested_snr_db == float(noise)).all()
            assert np.allclose(selected.scaled_noise_chunk_power,
                .011917774650702801 ** 2 * 10 ** (-float(noise) / 10))
            for source, group in selected.groupby("source_wav_id", sort=False):
                snrs.append({"noise": noise, "source_wav_id": source,
                    "q_kW_m2": float(group.heat_flux.iloc[0]) / 1000, "n": len(group),
                    "realized_snr_median_db": float(group.realized_snr_db.median()),
                    "realized_snr_min_db": float(group.realized_snr_db.min()),
                    "realized_snr_max_db": float(group.realized_snr_db.max())})
        source_hashes.append({"path": str(data_manifest.relative_to(ROOT)), "sha256": sha(data_manifest)})
        y = frame.y_true.to_numpy()
        assert np.isfinite(frame[models].to_numpy()).all()
        for key, mapping in weights.items():
            reconstructed = sum(frame[k].to_numpy() * v for k, v in mapping.items())
            diff = float(np.max(np.abs(reconstructed - frame[key].to_numpy())))
            max_reconstruction = max(max_reconstruction, diff)
            assert diff < 1e-6
        computed = source_day_metric_rows(y, {k: frame[k].to_numpy() for k in models},
            frame.to_dict("records"), thresholds, 1)
        saved = pd.read_csv(paths[5]).set_index(["source_day", "model_key"])
        for row in computed:
            original = saved.loc[(row["source_day"], row["model_key"])]
            for metric, value in row.items():
                if isinstance(value, (int, float)) and metric in original.index and np.isfinite(value):
                    diff = abs(float(original[metric]) - value)
                    max_metric_difference = max(max_metric_difference, diff)
                    assert np.isclose(original[metric], value, rtol=1e-10, atol=1e-7), metric
            rows.append({"seed": 42, "noise": noise, "source_result": str(folder.relative_to(ROOT)), **row})
        actual = y >= threshold
        near = np.abs(y - threshold) <= .1 * threshold
        base_positive = frame[CORE].to_numpy() >= threshold
        five_positive = frame[FIVE].to_numpy() >= threshold
        all_negative = (frame[MEMBERS].to_numpy() < threshold).all(axis=1)
        lost_et = actual & (frame.extra_trees.to_numpy() >= threshold) & ~five_positive
        cases_mask = actual & ~five_positive
        for i in np.flatnonzero(cases_mask):
            cases.append({"noise": noise, **frame.iloc[i].to_dict(),
                "all_five_negative": bool(all_negative[i]), "et_positive_lost": bool(lost_et[i])})
        for baseline in [CORE, ET4, HGB4, "extra_trees", "hgb"]:
            before = frame[baseline].to_numpy() >= threshold
            changes.append({"noise": noise, "baseline": baseline, "method": FIVE,
                "corrected_fn": int(np.sum(actual & ~before & five_positive)),
                "added_fn": int(np.sum(actual & before & ~five_positive)),
                "corrected_fp": int(np.sum(~actual & before & ~five_positive)),
                "added_fp": int(np.sum(~actual & ~before & five_positive))})
        for key in models:
            prediction = frame[key].to_numpy()
            missed = actual & (prediction < threshold)
            maximum_negative = float(np.max(prediction[~actual]))
            minimum_positive = float(np.min(prediction[actual]))
            endpoints.append({"noise": noise, "model_key": key,
                "last_fn_q_kW_m2": float(np.max(y[missed])) / 1000 if missed.any() else np.nan,
                "all_five_common_fn": int(np.sum(actual & all_negative)),
                "et_positive_lost_by_five": int(lost_et.sum()),
                "max_pre_onb_prediction_kW_m2": maximum_negative / 1000,
                "min_post_onb_prediction_kW_m2": minimum_positive / 1000,
                "ranking_gap_kW_m2": (minimum_positive - maximum_negative) / 1000,
                "calibration_margin_at_onb_kW_m2": (minimum_positive - threshold) / 1000})
            for source, group in frame.groupby("source_wav_id", sort=False):
                values = group[key].to_numpy() / 1000
                target = float(group.y_true.iloc[0]) / 1000
                positive = values >= threshold / 1000
                stage_rows.append({"noise": noise, "model_key": key, "source_wav_id": source,
                    "q_kW_m2": target, "n": len(group), "positive": int(positive.sum()),
                    "fn": int((~positive).sum()) if target >= threshold / 1000 else 0,
                    "fp": int(positive.sum()) if target < threshold / 1000 else 0,
                    "prediction_mean_kW_m2": float(values.mean()),
                    "prediction_min_kW_m2": float(values.min()), "prediction_max_kW_m2": float(values.max()),
                    "rmse_kW_m2": float(np.sqrt(np.mean((values - target) ** 2))),
                    "bias_kW_m2": float(values.mean() - target),
                    "sse_kW_m2_squared": float(np.sum((values - target) ** 2))})
        for region, mask in [("all", np.ones(len(y), bool)), ("pre_onb", ~actual),
                ("near_onb", near), ("post_onb", actual)]:
            e4 = (frame[ET4].to_numpy() - y)[mask] / 1000
            delta = (frame[FIVE].to_numpy() - frame[ET4].to_numpy())[mask] / 1000
            cross = float(2 * np.mean(e4 * delta))
            square = float(np.mean(delta ** 2))
            difference = float(np.mean((e4 + delta) ** 2) - np.mean(e4 ** 2))
            assert np.isclose(difference, cross + square, atol=1e-8)
            losses.append({"noise": noise, "region": region, "comparison": "five_minus_et4",
                "n": int(mask.sum()), "cross_term": cross, "squared_adjustment": square,
                "mse_change_kW_m2_squared": difference})
            for i, key in enumerate(MEMBERS):
                for other in MEMBERS[i + 1:]:
                    first = (frame[key].to_numpy() - y)[mask] / 1000
                    second = (frame[other].to_numpy() - y)[mask] / 1000
                    correlations.append({"noise": noise, "region": region, "first": key, "second": other,
                        "residual_correlation": float(np.corrcoef(first, second)[0, 1]),
                        "same_error_sign_fraction": float(np.mean(first * second > 0))})
        completed.append({"noise": noise, **completion})
        source_hashes.extend({"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in paths)
    assert len(epochs) == 8
    for row in epochs:
        assert row["epochs_completed"] == row["requested_epochs"] == 200
        assert row["actual_batch_size"] == row["requested_batch_size"]
        assert not row["stopped_by_memory_error"]
    protocol = {"model_keys": MEMBERS, "ensemble_names": [k.removeprefix("ensemble__") for k in weights],
        "comparison_baselines": [CORE, ET4, "extra_trees"], "noise_conditions": NOISES}
    tables = comparison_tables(rows, protocol)
    for name, table in tables.items():
        save_csv(name, table)
    save_csv("stage_metrics.csv", stage_rows)
    save_csv("paired_detection_changes.csv", changes)
    save_csv("endpoints_and_ranking.csv", endpoints)
    save_csv("five_false_negative_cases.csv", cases)
    save_csv("loss_decomposition.csv", losses)
    save_csv("residual_complementarity.csv", correlations)
    save_csv("fit_epoch_audit.csv", epochs)
    save_csv("saved_reload_checks.csv", reloads)
    save_csv("realized_snr_by_stage.csv", snrs)
    # OOF margins describe the same training-side predictions used to fit weights.
    oof_rows = []
    for key in models:
        p = oof[key].to_numpy() if key in MEMBERS else sum(oof[k].to_numpy() * v for k, v in weights[key].items())
        actual = oof.y_true.to_numpy() >= threshold
        oof_rows.append({"model_key": key, "n": len(oof),
            "fn": int(np.sum(actual & (p < threshold))), "fp": int(np.sum(~actual & (p >= threshold))),
            "rmse_kW_m2": float(np.sqrt(np.mean((p - oof.y_true.to_numpy()) ** 2))) / 1000,
            "max_pre_onb_prediction_kW_m2": float(p[~actual].max()) / 1000,
            "min_post_onb_prediction_kW_m2": float(p[actual].min()) / 1000})
    save_csv("oof_fit_diagnostics.csv", oof_rows)
    conditions = {"analysis_date_jst": "2026-10-14", "run_directory": str(RUN.relative_to(ROOT)),
        "run_hash": manifest["run_hash"], "fit_id": artifacts["fit_id"], "learning_context": context,
        "input": {"chunk_seconds": 1, "max_frequency_hz": 3000, "selection": "none"},
        "threshold_W_m2": threshold, "training_chunks": 495, "evaluation_chunks_per_condition": 165,
        "independent_source_wavs": 11, "training_and_evaluation_wavs_shared": True,
        "evaluation_positive_negative_near": [135, 30, 15], "noise_conditions": NOISES,
        "noise_reference_rms": .011917774650702801,
        "evaluation_scope": "new_experiment_within_known_source_wav_unseen_chunks",
        "saved_old_fit_transfer_performed": False, "analysis_fitting_or_model_inference": False,
        "weights": weights}
    (OUT / "conditions.json").write_text(json.dumps(conditions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verification = {"status": "passed", "completed_conditions": completed, "same_evaluation_ids": True,
        "same_clean_fit_and_weights": True, "oof_coverage_once": True,
        "cnn_fits_completed_200_epochs": 8, "saved_model_hashes_match": artifact_hashes,
        "original_full_reload_prediction_count": 5775,
        "original_full_reload_max_difference_W_m2": max(r["max_abs_prediction_difference_w_m2"] for r in reloads),
        "recomputed_metric_rows": len(rows), "max_metric_difference_original_W_m2": max_metric_difference,
        "max_ensemble_reconstruction_difference_W_m2": max_reconstruction,
        "source_hashes": source_hashes + [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in base_inputs],
        "scope": "Saved metadata/prediction checks and analysis only. Source model reload logs checked, not rerun."}
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(frames, pd.DataFrame(tables["main_metrics.csv"]), threshold / 1000)
    table = pd.DataFrame(tables["main_metrics.csv"])
    table = table[table.source_day == DAY]
    display = table[table.noise.isin(["clean", "-20"])][["noise", "model_key", "rmse_all_kW_m2", "rmse_onb_kW_m2", "fn", "fp", "q100_kW_m2"]]
    print(display.round(3).to_string(index=False))
    print(json.dumps({"verification": "passed", "metric_rows": len(rows), "epochs": len(epochs)}, ensure_ascii=False))


def plot(frames, metrics, threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {CORE: "tab:gray", ET4: "tab:orange", FIVE: "tab:blue", "extra_trees": "tab:green"}
    names = {CORE: "Original 3", ET4: "+ ExtraTrees (4)", FIVE: "+ HGB + ExtraTrees (5)", "extra_trees": "ExtraTrees"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    current = metrics[metrics.source_day == DAY]
    for key in colors:
        subset = current[current.model_key == key].set_index("noise").loc[NOISES]
        axes[0].plot(range(7), subset.rmse_all_kW_m2, "o-", color=colors[key], label=names[key])
        axes[1].plot(range(7), subset.fpr * 100, "o-", color=colors[key], label=names[key])
        axes[2].plot(range(7), subset.recall * 100, "o-", color=colors[key], label=names[key])
    for ax, title in zip(axes, ["Regression RMSE (kW/m2)", "False positive rate (%)", "Recall (%)"]):
        ax.set_title(title)
        ax.set_xticks(range(7), NOISES)
        ax.set_xlabel("Reference SNR (dB); clean = no added noise")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("New experiment trained within WAVs: 495 train / 165 test; one fixed clean fit")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(OUT / ("noise_and_detection." + extension), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, noise in zip(axes, ["clean", "-20"]):
        frame = frames[noise]
        for key in colors:
            grouped = frame.groupby("y_true")[key].agg(["mean", "min", "max"]) / 1000
            ax.plot(grouped.index / 1000, grouped["mean"], "o-", color=colors[key], label=names[key], markersize=4)
        ax.plot([230, 900], [230, 900], "k--", linewidth=1, label="Exact regression")
        ax.axhline(threshold, color="red", linestyle=":", label="Observed ONB threshold")
        ax.axvline(threshold, color="red", linestyle=":")
        ax.set_title("Stage mean predictions: " + noise)
        ax.set_xlabel("Measured stage heat flux (kW/m2)")
        ax.set_ylabel("Predicted heat flux (kW/m2)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(OUT / ("stage_predictions." + extension), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
