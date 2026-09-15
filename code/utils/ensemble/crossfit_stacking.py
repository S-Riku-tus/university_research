"""Small-sample stacking with the same chunk -> WAV reduction as evaluation.

Only inner, source-WAV-disjoint OOF predictions may be passed for fitting.
The median objective is not convex; retain all feasible subset starts as well
as successful SLSQP solutions. No outer-test improvement is guaranteed.
"""

from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold


CROSSFIT_STRATEGIES = frozenset({
    "subset_equal_cv", "crossfit_wav_stack", "crossfit_shrinkage_stack",
})
CROSSFIT_DEFAULTS = {
    "inner_folds": 4,
    "aggregation": "median",
    "regularization": 0.0,
    "maxiter": 300,
    "ftol": 1e-10,
    "max_models": 8,
}


def validate_crossfit_options(options):
    if options["aggregation"] not in {"mean", "median"}:
        raise ValueError("Crossfit aggregation must be mean or median.")
    for key, minimum in (("inner_folds", 2), ("maxiter", 1), ("max_models", 2)):
        if isinstance(options[key], bool) or int(options[key]) != options[key] or options[key] < minimum:
            raise ValueError(f"Crossfit {key} must be an integer >= {minimum}.")
    if not np.isfinite(options["regularization"]) or options["regularization"] < 0:
        raise ValueError("Crossfit regularization must be finite and nonnegative.")
    if not np.isfinite(options["ftol"]) or options["ftol"] <= 0:
        raise ValueError("Crossfit ftol must be finite and positive.")


def group_crossfit_splits(groups, n_splits=4, random_state=42):
    """Shuffle unique WAV IDs, never chunks; cover each WAV exactly once."""
    if groups is None:
        raise ValueError("Crossfit requires source_wav_id groups.")
    groups = np.asarray(groups).ravel()
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Crossfit requires at least three source WAV groups.")
    if isinstance(n_splits, bool) or int(n_splits) != n_splits or n_splits < 2:
        raise ValueError("Crossfit n_splits must be an integer >= 2.")
    splitter = KFold(n_splits=min(int(n_splits), len(unique)), shuffle=True,
                     random_state=int(random_state))
    splits = []
    for fit, held in splitter.split(unique):
        fit_index = np.flatnonzero(np.isin(groups, unique[fit]))
        held_index = np.flatnonzero(np.isin(groups, unique[held]))
        if len(fit) < 2:
            raise ValueError("Each crossfit training partition needs at least two WAVs.")
        splits.append((fit_index, held_index))
    return splits


def _validated_data(y_true, predictions, groups, model_keys):
    y = np.asarray(y_true, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if not model_keys or len(set(model_keys)) != len(model_keys):
        raise ValueError("Model keys must be nonempty and unique.")
    if set(predictions) != set(model_keys):
        raise ValueError("OOF predictions must contain exactly the configured models.")
    columns = [np.asarray(predictions[key], dtype=float).ravel() for key in model_keys]
    if not len(y) or len(groups) != len(y) or any(len(p) != len(y) for p in columns):
        raise ValueError("OOF targets, predictions and groups must have equal nonzero lengths.")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all() or not np.isfinite(y).all():
        raise ValueError("OOF targets/predictions must be finite and fully covered.")
    indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for index in indices:
        values = y[index]
        if not np.allclose(values, values[0], rtol=0, atol=max(1e-6, np.max(np.abs(values)) * 1e-9)):
            raise ValueError("A source WAV has multiple targets.")
    return y, matrix, indices


def _subset_weights(n_models):
    # Ties prefer more models; no outcome-dependent tie tolerance.
    for size in range(n_models, 0, -1):
        for subset in combinations(range(n_models), size):
            weights = np.zeros(n_models)
            weights[list(subset)] = 1.0 / size
            yield weights


def fit_crossfit_strategy(y_true, predictions, groups, model_keys, strategy, options=None):
    """Fit fixed convex weights; score every WAV equally after chunk mixing.

    regularization is fixed in the catalog, not tuned on evaluation labels.
    Loss = WAV MSE / mean(single-model WAV MSE) + lambda * ||w - uniform||².
    Inner OOF scores are fitting diagnostics, not unbiased ensemble estimates.
    """
    if strategy not in CROSSFIT_STRATEGIES:
        raise ValueError(f"Unknown crossfit strategy: {strategy}")
    options = {**CROSSFIT_DEFAULTS, **(options or {})}
    validate_crossfit_options(options)
    if strategy != "crossfit_shrinkage_stack" and options["regularization"] != 0:
        raise ValueError("Only crossfit_shrinkage_stack accepts regularization.")
    if len(model_keys) > options["max_models"]:
        raise ValueError("Too many models for exhaustive subset starts.")
    y, matrix, indices = _validated_data(y_true, predictions, groups, model_keys)
    if len(indices) < 3:
        raise ValueError("Crossfit weight fitting requires at least three WAVs.")
    target = np.asarray([y[index[0]] for index in indices])
    reducer = np.median if options["aggregation"] == "median" else np.mean
    n_models = len(model_keys)
    uniform = np.full(n_models, 1.0 / n_models)

    def grouped_prediction(weights):
        mixed = matrix @ weights
        return np.asarray([reducer(mixed[index]) for index in indices])

    def mse(weights):
        return float(np.mean(np.square(grouped_prediction(weights) - target)))

    single_mse = np.asarray([mse(vertex) for vertex in np.eye(n_models)])
    # Translation-independent scale; exact predictions can have zero residuals.
    scale = max(float(np.mean(single_mse)), np.finfo(float).tiny)
    strength = float(options["regularization"])

    def objective(weights):
        return mse(weights) / scale + strength * float(np.sum((weights - uniform) ** 2))

    starts = list(_subset_weights(n_models))
    candidate_weights = list(starts)
    solver_attempts = []
    if strategy != "subset_equal_cv" and np.any(single_mse > 0):
        for start in starts:
            result = minimize(objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * n_models,
                              constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
                              options={"maxiter": int(options["maxiter"]), "ftol": options["ftol"]})
            feasible = (np.isfinite(result.x).all() and np.min(result.x) >= -1e-7
                        and abs(result.x.sum() - 1.0) <= 1e-7)
            accepted = bool(result.success and feasible)
            solver_attempts.append({"success": accepted, "message": str(result.message),
                                    "iterations": int(result.nit)})
            if accepted:
                weights = np.clip(result.x, 0.0, 1.0)
                candidate_weights.append(weights / weights.sum())
    scores = [objective(w) for w in candidate_weights]
    chosen = int(np.argmin(scores))
    weights = candidate_weights[chosen]
    group_single = np.column_stack([grouped_prediction(w) for w in np.eye(n_models)])
    residual = group_single - target[:, None]
    correlation = []
    for left in range(n_models):
        row = []
        for right in range(n_models):
            if np.std(residual[:, left]) == 0 or np.std(residual[:, right]) == 0:
                row.append(None)
            else:
                row.append(float(np.corrcoef(residual[:, left], residual[:, right])[0, 1]))
        correlation.append(row)
    diagnostics = {
        "strategy": strategy, "model_keys": list(model_keys), "options": options,
        "n_wavs": len(indices), "n_chunks": len(y),
        "score_role": "inner_oof_weight_fit_only_not_outer_performance",
        "prediction_order": "chunk_weighted_mean_then_wav_" + options["aggregation"],
        "mse_normalization": scale, "single_wav_mse": dict(zip(model_keys, single_mse.tolist())),
        "equal_wav_mse": mse(uniform), "selected_wav_mse": mse(weights),
        "selected_objective": scores[chosen], "residual_correlation": correlation,
        "subset_candidates": [{"weights": w.tolist(), "wav_mse": mse(w)} for w in starts],
        "solver_attempts": solver_attempts,
        "solver_fallback": bool(solver_attempts and not any(a["success"] for a in solver_attempts)),
        "selected_from": "subset_start" if chosen < len(starts) else "slsqp",
    }
    return dict(zip(model_keys, weights.tolist())), diagnostics
