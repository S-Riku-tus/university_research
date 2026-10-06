"""Supplementary ONB metrics by source-day thresholds, including pooled counts.

The scalar combined-experiment threshold used in historical ONB summaries is
unchanged. These rows explicitly record a different, source-day label basis.
"""
import numpy as np
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


def source_experiment(row):
    return str(row.get("source_experiment_name") or row["experiment_name"])


def detection_regression_metrics(y, prediction, thresholds, band_frac=.1):
    y, prediction = np.asarray(y, float).ravel(), np.asarray(prediction, float).ravel()
    threshold = np.broadcast_to(np.asarray(thresholds, float), y.shape)
    if (y.shape != prediction.shape or not len(y) or not np.isfinite(y).all()
            or not np.isfinite(prediction).all() or not np.isfinite(threshold).all()):
        raise ValueError("Invalid aligned predictions, targets or ONB thresholds")
    actual, positive = y >= threshold, prediction >= threshold
    near = abs(y-threshold) <= abs(threshold)*band_frac
    tp, fp = int(np.sum(actual & positive)), int(np.sum(~actual & positive))
    fn, tn = int(np.sum(actual & ~positive)), int(np.sum(~actual & ~positive))

    def error(mask, kind="rmse"):
        if not mask.any():
            return np.nan
        residual = prediction[mask]-y[mask]
        return float(np.sqrt(np.mean(residual**2)) if kind == "rmse" else np.mean(abs(residual)))

    both = np.ones(len(y), bool)
    return {"n": len(y), "n_pre_onb": tn+fp, "n_post_onb": tp+fn, "n_onb": int(near.sum()),
        "r2": float(r2_score(y, prediction)) if len(y)>1 and np.var(y)>0 else np.nan,
        "rmse_all": error(both), "mae_all": error(both, "mae"),
        "bias_all": float(np.mean(prediction-y)),
        "bias_pre_onb": float(np.mean((prediction-y)[~actual])) if (~actual).any() else np.nan,
        "bias_onb": float(np.mean((prediction-y)[near])) if near.any() else np.nan,
        "bias_high": float(np.mean((prediction-y)[actual])) if actual.any() else np.nan,
        "rmse_pre_onb": error(~actual), "rmse_high": error(actual), "mae_high": error(actual, "mae"),
        "r2_high": float(r2_score(y[actual], prediction[actual])) if actual.sum()>1 and np.var(y[actual])>0 else np.nan,
        "rmse_onb": error(near), "mae_onb": error(near, "mae"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp+tn)/len(y),
        "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "fpr": fp/(tn+fp) if tn+fp else np.nan,
        "roc_auc_cont": float(roc_auc_score(actual, prediction)) if len(np.unique(actual))==2 else np.nan,
        "pr_auc_cont": float(average_precision_score(actual, prediction)) if actual.any() else np.nan}


def source_day_metric_rows(y, predictions, metadata, thresholds, fold, labels=None, band_frac=.1):
    y = np.asarray(y, float).ravel()
    if len(metadata) != len(y):
        raise ValueError("Evaluation metadata and targets must align")
    days = np.asarray([source_experiment(r) for r in metadata])
    missing = set(days)-set(thresholds)
    if missing:
        raise ValueError(f"Missing source-day ONB thresholds: {sorted(missing)}")
    sample_thresholds = np.asarray([thresholds[d] for d in days], float)
    rows = []
    for key, prediction in predictions.items():
        prediction = np.asarray(prediction, float).ravel()
        if prediction.shape != y.shape:
            raise ValueError("Evaluation predictions and targets must align")
        for day in ["pooled_source_day_thresholds", *sorted(set(days))]:
            mask = np.ones(len(y), bool) if day == "pooled_source_day_thresholds" else days == day
            current_y, current_p = y[mask], prediction[mask]
            q100, g100 = np.nan, np.nan
            if day != "pooled_source_day_thresholds":
                reached = [v for v in np.unique(current_y) if np.all(current_p[current_y>=v]>=thresholds[day])]
                if reached:
                    q100, g100 = float(reached[0]), float(reached[0]-thresholds[day])
            rows.append({"fold": int(fold), "model_key": key, "model_label": (labels or {}).get(key, key),
                "source_day": day, "threshold_basis": "source-day", "heat_flux_unit": "W/m2",
                "onb_threshold": np.nan if day == "pooled_source_day_thresholds" else float(thresholds[day]),
                "n_source_wavs": len({(source_experiment(r), r["source_wav_id"]) for i, r in enumerate(metadata) if mask[i]}),
                **detection_regression_metrics(current_y, current_p, sample_thresholds[mask], band_frac),
                "q100": q100, "g100": g100})
    return rows


def metric_delta_rows(rows, baseline_key="ensemble__original3_mse"):
    """Compare identical source-day/fold units; changes are new minus baseline."""
    lookup = {(r["fold"], r["source_day"]): r for r in rows if r["model_key"] == baseline_key}
    deltas = []
    metrics = ["r2", "rmse_all", "mae_all", "rmse_high", "rmse_onb", "roc_auc_cont", "pr_auc_cont",
               "accuracy", "precision", "recall", "f1", "fpr", "fp", "fn", "q100", "g100"]
    rates = {"accuracy", "precision", "recall", "f1", "fpr"}
    for row in rows:
        before = lookup.get((row["fold"], row["source_day"]))
        if before is None or row["model_key"] == baseline_key:
            continue
        for metric in metrics:
            change = float(row[metric]-before[metric])
            deltas.append({"fold": row["fold"], "source_day": row["source_day"],
                "baseline_key": baseline_key, "model_key": row["model_key"], "metric": metric,
                "baseline": before[metric], "value": row[metric], "change": change,
                "percentage_point_change": change*100 if metric in rates else np.nan,
                "threshold_basis": row["threshold_basis"]})
    return deltas
