import itertools
import re

import numpy as np


VALID_STRATEGIES = {
    "simple", "max", "inner_holdout", "val_fold_legacy",
}
HIGHER_IS_BETTER = {
    "r2", "r2_high", "auc_binary", "roc_auc_cont", "pr_auc_cont",
    "accuracy", "precision", "recall", "f1",
}
LOWER_IS_BETTER = {
    "rmse_all", "mae_all", "rmse_high", "mae_high", "rmse_onb", "mae_onb",
}


def _safe_name(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text[:40] or "ensemble"


def normalize_strategy_plan(config):
    """Normalize configured ensemble strategies without touching model training.

    Disabled entries are retained in the configuration file for discoverability
    but are not returned.  Validation-fold weighting is rejected unless it is
    explicitly unlocked as a reproduction-only diagnostic.
    """
    configured = config.get("strategies")
    if not configured:
        configured = [{
            "name": config.get("weight_strategy", "simple"),
            "strategy": config.get("weight_strategy", "simple"),
        }]

    allow_leaky = bool(config.get("allow_leaky_strategies", False))
    plan = []
    names = set()
    result_keys = set()
    for index, raw in enumerate(configured, start=1):
        if isinstance(raw, str):
            raw = {"name": raw, "strategy": raw}
        if not raw.get("enabled", True):
            continue

        strategy = str(raw.get("strategy", "")).strip().lower()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown ensemble strategy {strategy!r}; expected one of "
                f"{sorted(VALID_STRATEGIES)}."
            )
        if strategy == "val_fold_legacy" and not allow_leaky:
            raise ValueError(
                "val_fold_legacy uses outer validation labels and is disabled. "
                "Set allow_leaky_strategies=True only for reproduction diagnostics."
            )

        name = _safe_name(raw.get("name") or f"{strategy}_{index}")
        result_key = f"ensemble__{name}"
        if name in names or result_key in result_keys:
            raise ValueError(f"Duplicate ensemble strategy name: {name}")

        item = {
            "name": name,
            "result_key": result_key,
            "label": str(raw.get("label") or f"Ensemble {name}"),
            "strategy": strategy,
            "claim_safe": strategy != "val_fold_legacy",
        }
        plan.append(item)
        names.add(name)
        result_keys.add(result_key)

    if config.get("enabled", False) and not plan:
        raise ValueError("Ensemble is enabled but no ensemble strategies are enabled.")
    return plan


def strategy_plan_requires_inner_holdout(strategy_plan):
    return any(item["strategy"] == "inner_holdout" for item in strategy_plan)


def compute_strategy_outputs(
    weighting,
    strategy_plan,
    run_specs,
    val_preds,
    combine,
    inner_errors=None,
    legacy_errors=None,
):
    outputs = {}
    inner_errors = inner_errors or {}
    legacy_errors = legacy_errors or {}
    for item in strategy_plan:
        strategy = item["strategy"]
        if strategy == "inner_holdout":
            errors = inner_errors
        elif strategy == "val_fold_legacy":
            errors = legacy_errors
        else:
            errors = {}
        weight_strategy = "simple" if strategy == "max" else strategy
        weights = weighting.compute_weights(
            weight_strategy,
            run_specs,
            errors,
        )
        combine_method = "max" if strategy == "max" else combine
        prediction = weighting.combine_predictions(
            val_preds, weights, combine_method
        )
        outputs[item["result_key"]] = {
            "strategy": item,
            "weights": weights,
            "prediction": np.asarray(prediction, dtype=float),
        }
    return outputs


def metric_comparison_rows(store, model_keys, strategy_plan, mean_se, metric_names):
    """Return long-form ensemble deltas where positive always means improvement."""
    rows = []
    for item in strategy_plan:
        result_key = item["result_key"]
        for metric_name in metric_names:
            if metric_name not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
                continue
            ensemble_mean, ensemble_se = mean_se(store[result_key][metric_name])
            candidates = []
            for model_key in model_keys:
                model_mean, model_se = mean_se(store[model_key][metric_name])
                if np.isfinite(model_mean):
                    candidates.append((model_key, model_mean, model_se))
            if not candidates or not np.isfinite(ensemble_mean):
                best_key = ""
                best_mean = best_se = improvement = np.nan
                paired_se = np.nan
                folds_improved = folds_total = 0
            elif metric_name in HIGHER_IS_BETTER:
                best_key, best_mean, best_se = max(candidates, key=lambda row: row[1])
                improvement = ensemble_mean - best_mean
                fold_improvements = (
                    np.asarray(store[result_key][metric_name], dtype=float)
                    - np.asarray(store[best_key][metric_name], dtype=float)
                )
            else:
                best_key, best_mean, best_se = min(candidates, key=lambda row: row[1])
                improvement = best_mean - ensemble_mean
                fold_improvements = (
                    np.asarray(store[best_key][metric_name], dtype=float)
                    - np.asarray(store[result_key][metric_name], dtype=float)
                )
            if candidates and np.isfinite(ensemble_mean):
                finite = fold_improvements[np.isfinite(fold_improvements)]
                folds_total = int(finite.size)
                folds_improved = int(np.sum(finite > 0))
                paired_se = (
                    float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                    if finite.size > 1 else np.nan
                )
            rows.append([
                item["name"],
                item["strategy"],
                int(item["claim_safe"]),
                metric_name,
                "higher" if metric_name in HIGHER_IS_BETTER else "lower",
                ensemble_mean,
                ensemble_se,
                best_key,
                best_mean,
                best_se,
                improvement,
                int(np.isfinite(improvement) and improvement > 0),
                paired_se,
                folds_improved,
                folds_total,
            ])
    return rows


def _safe_corr(left, right):
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.size < 2 or right.size != left.size:
        return np.nan
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def pairwise_diversity_rows(y_true, predictions, threshold=None, onb_band_frac=0.10):
    y_true = np.asarray(y_true, dtype=float).ravel()
    rows = []
    for left_key, right_key in itertools.combinations(predictions, 2):
        left = np.asarray(predictions[left_key], dtype=float).ravel()
        right = np.asarray(predictions[right_key], dtype=float).ravel()
        left_error = np.abs(left - y_true)
        right_error = np.abs(right - y_true)
        onb_mask = np.zeros_like(y_true, dtype=bool)
        if threshold is not None and np.isfinite(float(threshold)):
            threshold_value = float(threshold)
            onb_mask = np.abs(y_true - threshold_value) <= abs(threshold_value) * float(
                onb_band_frac
            )
        rows.append([
            left_key,
            right_key,
            len(y_true),
            _safe_corr(left, right),
            _safe_corr(left - y_true, right - y_true),
            int(np.sum(left_error < right_error)),
            int(np.sum(right_error < left_error)),
            int(np.sum(np.isclose(left_error, right_error))),
            int(np.sum(onb_mask)),
            _safe_corr((left - y_true)[onb_mask], (right - y_true)[onb_mask])
            if np.any(onb_mask) else np.nan,
        ])
    return rows


def correction_diagnostic_row(
    strategy_item,
    y_true,
    reference_key,
    reference_pred,
    ensemble_pred,
    threshold=None,
    onb_band_frac=0.10,
):
    y_true = np.asarray(y_true, dtype=float).ravel()
    reference_pred = np.asarray(reference_pred, dtype=float).ravel()
    ensemble_pred = np.asarray(ensemble_pred, dtype=float).ravel()
    reference_error = np.abs(reference_pred - y_true)
    ensemble_error = np.abs(ensemble_pred - y_true)
    error_delta = ensemble_error - reference_error

    onb_mask = np.zeros_like(y_true, dtype=bool)
    recovered_false_negatives = new_false_negatives = 0
    reference_false_negatives = ensemble_false_negatives = 0
    if threshold is not None and np.isfinite(float(threshold)):
        threshold_value = float(threshold)
        onb_mask = np.abs(y_true - threshold_value) <= abs(threshold_value) * float(
            onb_band_frac
        )
        true_positive = y_true >= threshold_value
        reference_positive = reference_pred >= threshold_value
        ensemble_positive = ensemble_pred >= threshold_value
        reference_fn_mask = true_positive & ~reference_positive
        ensemble_fn_mask = true_positive & ~ensemble_positive
        reference_false_negatives = int(np.sum(reference_fn_mask))
        ensemble_false_negatives = int(np.sum(ensemble_fn_mask))
        recovered_false_negatives = int(np.sum(reference_fn_mask & ensemble_positive))
        new_false_negatives = int(np.sum(ensemble_fn_mask & reference_positive))

    return [
        strategy_item["name"],
        strategy_item["strategy"],
        int(strategy_item["claim_safe"]),
        reference_key,
        len(y_true),
        int(np.sum(error_delta < -1e-9)),
        int(np.sum(error_delta > 1e-9)),
        int(np.sum(np.abs(error_delta) <= 1e-9)),
        float(np.mean(error_delta)),
        int(np.sum(onb_mask)),
        float(np.mean(error_delta[onb_mask])) if np.any(onb_mask) else np.nan,
        reference_false_negatives,
        ensemble_false_negatives,
        ensemble_false_negatives - reference_false_negatives,
        recovered_false_negatives,
        new_false_negatives,
    ]


def aggregate_correction_rows(correction_rows):
    """Aggregate fold-wise correction diagnostics for presentation tables."""
    grouped = {}
    for row in correction_rows:
        key = tuple(row[1:5])
        grouped.setdefault(key, []).append(row)

    output = []
    for (name, strategy, claim_safe, reference_key), rows in grouped.items():
        n_samples = sum(int(row[5]) for row in rows)
        n_onb = sum(int(row[10]) for row in rows)
        mean_error_delta = (
            sum(float(row[9]) * int(row[5]) for row in rows) / n_samples
            if n_samples else np.nan
        )
        finite_onb_rows = [
            row for row in rows
            if int(row[10]) > 0 and np.isfinite(float(row[11]))
        ]
        finite_onb_count = sum(int(row[10]) for row in finite_onb_rows)
        mean_onb_error_delta = (
            sum(float(row[11]) * int(row[10]) for row in finite_onb_rows)
            / finite_onb_count
            if finite_onb_count else np.nan
        )
        n_better = sum(int(row[6]) for row in rows)
        n_worse = sum(int(row[7]) for row in rows)
        output.append([
            name,
            strategy,
            claim_safe,
            reference_key,
            len(rows),
            n_samples,
            n_better,
            n_worse,
            sum(int(row[8]) for row in rows),
            n_better / n_samples if n_samples else np.nan,
            n_worse / n_samples if n_samples else np.nan,
            mean_error_delta,
            n_onb,
            mean_onb_error_delta,
            sum(int(row[12]) for row in rows),
            sum(int(row[13]) for row in rows),
            sum(int(row[14]) for row in rows),
            sum(int(row[15]) for row in rows),
            sum(int(row[16]) for row in rows),
        ])
    return output
