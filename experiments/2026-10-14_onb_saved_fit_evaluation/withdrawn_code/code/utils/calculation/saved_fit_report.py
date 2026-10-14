"""Descriptive diagnostics of frozen predictions; never select or fit models."""
import csv
from collections import defaultdict
from itertools import combinations

import numpy as np

from utils.calculation.onb_comparison_report import _csv
from utils.calculation.source_day_metrics import detection_regression_metrics, source_experiment
from utils.dataloading.acoustic_summary_features import AcousticFrequency34
from utils.experiment.run_helpers import open_text


BASELINE = "ensemble__original3_mse"
PRIMARY = "ensemble__original3_hgb_extra_trees"


def transfer_diagnostics(y, predictions, metadata, thresholds, training_metadata, band_frac=.1):
    y = np.asarray(y, float)
    days = np.asarray([source_experiment(r) for r in metadata])
    threshold = np.asarray([thresholds[d] for d in days], float)
    base_keys = [k for k in predictions if not k.startswith("ensemble__")]
    actual = y >= threshold
    positive = {k: np.asarray(p, float) >= threshold for k, p in predictions.items()}
    primary, baseline = positive[PRIMARY], positive[BASELINE]
    all_negative = ~np.any([positive[k] for k in base_keys], axis=0)
    train_q = np.asarray([float(r["heat_flux"]) for r in training_metadata])
    low, high = float(train_q.min()), float(train_q.max())
    stages, cases, paired = [], [], []
    groups = defaultdict(list)
    for i, row in enumerate(metadata):
        groups[(days[i], row["source_wav_id"], float(y[i]))].append(i)
        if actual[i] and not primary[i]:
            category = "all_base_models_negative" if all_negative[i] else "positive_member_lost_in_ensemble"
        elif not actual[i] and primary[i]:
            category = "false_positive"
        else:
            category = "true_positive" if actual[i] else "true_negative"
        cases.append({"source_day": days[i], "source_wav_id": row["source_wav_id"],
            "chunk_index": int(row["chunk_index"]), "sample_filename": row["sample_filename"],
            "heat_flux_W_m2": float(y[i]), "onb_threshold_W_m2": float(threshold[i]),
            "q_over_onb": float(y[i]/threshold[i]),
            "training_range": "below" if y[i] < low else "above" if y[i] > high else "inside",
            "primary_outcome": category,
            "corrected_fn_vs_original3": bool(actual[i] and not baseline[i] and primary[i]),
            "added_fn_vs_original3": bool(actual[i] and baseline[i] and not primary[i]),
            "corrected_fp_vs_original3": bool(not actual[i] and baseline[i] and not primary[i]),
            "added_fp_vs_original3": bool(not actual[i] and not baseline[i] and primary[i]),
            **{k+"_W_m2": float(p[i]) for k, p in predictions.items()}})
    for (day, wav, q), indices in sorted(groups.items()):
        for key, values in predictions.items():
            p = np.asarray(values)[indices]
            stages.append({"source_day": day, "source_wav_id": wav, "heat_flux_W_m2": q,
                "onb_threshold_W_m2": thresholds[day], "q_over_onb": q/thresholds[day], "model_key": key,
                "prediction_mean_W_m2": float(p.mean()), "prediction_median_W_m2": float(np.median(p)),
                "prediction_p05_W_m2": float(np.quantile(p, .05)), "prediction_p95_W_m2": float(np.quantile(p, .95)),
                **detection_regression_metrics(y[indices], p, threshold[indices], band_frac)})
    for day in ["pooled_source_day_thresholds", *sorted(set(days))]:
        mask = np.ones(len(y), bool) if day == "pooled_source_day_thresholds" else days == day
        for key, p in positive.items():
            if key == BASELINE:
                continue
            paired.append({"source_day": day, "baseline_key": BASELINE, "model_key": key,
                "n": int(mask.sum()), "n_source_wavs": len({metadata[i]["source_wav_id"] for i in np.flatnonzero(mask)}),
                "corrected_fn": int(np.sum(mask & actual & ~baseline & p)),
                "added_fn": int(np.sum(mask & actual & baseline & ~p)),
                "corrected_fp": int(np.sum(mask & ~actual & baseline & ~p)),
                "added_fp": int(np.sum(mask & ~actual & ~baseline & p))})
    return stages, cases, paired


def write_transfer_diagnostics(directory, y, predictions, metadata, thresholds, training_metadata, band_frac):
    stages, cases, paired = transfer_diagnostics(y, predictions, metadata, thresholds, training_metadata, band_frac)
    _csv(directory / "metrics_by_stage.csv", stages)
    _csv(directory / "chunk_error_diagnostics.csv", cases)
    _csv(directory / "paired_detection_changes.csv", paired)
    residual_rows, range_rows = [], []
    y = np.asarray(y, float)
    threshold = np.asarray([thresholds[source_experiment(r)] for r in metadata])
    actual = y >= threshold
    ranges = np.asarray([r["training_range"] for r in cases])
    for name in ["below", "inside", "above"]:
        mask = ranges == name
        if mask.any():
            for key, p in predictions.items():
                range_rows.append({"training_range": name, "model_key": key,
                    "n_source_wavs": len({metadata[i]["source_wav_id"] for i in np.flatnonzero(mask)}),
                    **detection_regression_metrics(y[mask], np.asarray(p)[mask], threshold[mask], band_frac)})
    for region, mask in [("all", np.ones(len(y), bool)), ("pre_onb", ~actual),
                         ("onb_near", abs(y-threshold) <= abs(threshold)*band_frac), ("onb_or_above", actual)]:
        if not mask.any():
            continue
        for a, b in combinations([k for k in predictions if not k.startswith("ensemble__")], 2):
            pa, pb = np.asarray(predictions[a])[mask], np.asarray(predictions[b])[mask]
            ea, eb = pa-y[mask], pb-y[mask]
            positive_a, positive_b = pa >= threshold[mask], pb >= threshold[mask]
            truth = actual[mask]
            residual_rows.append({"region": region, "model_a": a, "model_b": b, "n_chunks": int(mask.sum()),
                "n_source_wavs": len({metadata[i]["source_wav_id"] for i in np.flatnonzero(mask)}),
                "residual_correlation": float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else np.nan,
                "mean_residual_product_W2_m4": float(np.mean(ea*eb)),
                "bias_a_W_m2": float(ea.mean()), "bias_b_W_m2": float(eb.mean()),
                "shared_fn": int(np.sum(truth & ~positive_a & ~positive_b)),
                "a_corrects_b_fn": int(np.sum(truth & positive_a & ~positive_b)),
                "b_corrects_a_fn": int(np.sum(truth & ~positive_a & positive_b)),
                "shared_fp": int(np.sum(~truth & positive_a & positive_b))})
    _csv(directory / "metrics_by_training_range.csv", range_rows)
    _csv(directory / "residual_complementarity.csv", residual_rows)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, key in zip(axes.ravel(), [BASELINE, "ensemble__original3_hgb", "ensemble__original3_extra_trees", PRIMARY]):
        rows = sorted([r for r in stages if r["model_key"] == key], key=lambda r: r["heat_flux_W_m2"])
        q = np.asarray([r["heat_flux_W_m2"] for r in rows])/1000
        mean = np.asarray([r["prediction_mean_W_m2"] for r in rows])/1000
        lower = np.asarray([r["prediction_p05_W_m2"] for r in rows])/1000
        upper = np.asarray([r["prediction_p95_W_m2"] for r in rows])/1000
        ax.plot(q, q, "k--", alpha=.5)
        ax.plot(q, mean, "o-", label="mean")
        ax.fill_between(q, lower, upper, alpha=.2, label="5-95% chunks")
        ax.axhline(rows[0]["onb_threshold_W_m2"]/1000, color="red", ls=":", label="target-day ONB")
        ax.axvline(rows[0]["onb_threshold_W_m2"]/1000, color="red", ls=":")
        ax.set(title=key.removeprefix("ensemble__"), xlabel="Measured heat flux (kW/m2)", ylabel="Predicted heat flux (kW/m2)")
        ax.legend(fontsize=8)
    with open_text(directory / "prediction_by_stage.png", "wb") as stream:
        fig.savefig(stream, dpi=160)
    plt.close(fig)


def write_feature_comparison(output, prepared, clean, thresholds, batch_size):
    """Export existing 34 features on exact old fit chunks and all new clean chunks.

    No normalization or fitting is performed. A WAV-level summary retains the
    recording unit; feature variation is descriptive, not causal identification.
    """
    from utils.experiment.saved_fit_evaluation import input_batches
    transformer = AcousticFrequency34()
    names = list(transformer.get_feature_names_out())
    source_thresholds = {**prepared["manifest"]["validation_config"]["thresholds"]["by_experiment"], **thresholds}
    summary = []
    with open_text(output / "features_by_chunk.csv", "w", encoding="utf-8-sig", newline="") as stream:
        fields = ["role", "source_day", "source_wav_id", "chunk_index", "heat_flux_W_m2", "onb_threshold_W_m2", "q_over_onb", *names]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role, data_path, metadata in [
            ("source_training", prepared["source_data_path"], prepared["training_metadata"]),
            ("target_clean", clean["job"]["data_path"], clean["metadata"]),
        ]:
            by_wav = defaultdict(list)
            for rows, x in input_batches(data_path, metadata, batch_size):
                for row, values in zip(rows, transformer.transform(x)):
                    day = source_experiment(row)
                    q, threshold = float(row["heat_flux"]), float(source_thresholds[day])
                    context = {"role": role, "source_day": day, "source_wav_id": row["source_wav_id"],
                               "heat_flux_W_m2": q, "onb_threshold_W_m2": threshold, "q_over_onb": q/threshold}
                    writer.writerow({**context, "chunk_index": int(row["chunk_index"]), **dict(zip(names, values))})
                    by_wav[(day, row["source_wav_id"], q)].append(values)
            for (day, wav, q), values in sorted(by_wav.items()):
                values = np.asarray(values)
                summary.append({"role": role, "source_day": day, "source_wav_id": wav, "heat_flux_W_m2": q,
                    "onb_threshold_W_m2": source_thresholds[day], "q_over_onb": q/source_thresholds[day], "n_chunks": len(values),
                    **{name+"_mean": float(values[:, i].mean()) for i, name in enumerate(names)},
                    **{name+"_std": float(values[:, i].std()) for i, name in enumerate(names)}})
    _csv(output / "features_by_wav.csv", summary)
    import matplotlib.pyplot as plt
    features = ["log_power_total", "log_ratio_2100_2500_to_1000_2000", "temporal_power_cv"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for day in sorted({r["source_day"] for r in summary}):
        rows = [r for r in summary if r["source_day"] == day]
        for j, feature in enumerate(features):
            for i, xkey in enumerate(["heat_flux_W_m2", "q_over_onb"]):
                x = [r[xkey]/1000 if i == 0 else r[xkey] for r in rows]
                axes[i, j].scatter(x, [r[feature+"_mean"] for r in rows], label=day, s=22)
                axes[i, j].set(xlabel="Heat flux (kW/m2)" if i == 0 else "Heat flux / day ONB", ylabel=feature)
    axes[0, 0].legend(fontsize=7)
    with open_text(output / "features_across_days.png", "wb") as stream:
        fig.savefig(stream, dpi=160)
    plt.close(fig)
