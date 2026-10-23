"""Training-OOF integration preserving the original three and their ratios."""
import numpy as np
from scipy.optimize import minimize


FIXED_CORE_STRATEGY = "fixed_core_oof"
CORE_KEYS = ["randomforest", "conformer", "alexnet"]
FIXED_CORE_DEFAULTS = {
    "core_keys": CORE_KEYS,
    "core_rule": "mse",
    # Numerical membership floor: 1e-6 of the original-three block, not a
    # claim that the RF has a scientifically established minimum utility.
    "core_member_floor": 1e-6,
    "minimum_core_total": .25,
    "minimum_added_member": .05,
    "added_keys": [],
}


def validate_options(options):
    core, added = list(options["core_keys"]), list(options["added_keys"])
    if core != CORE_KEYS or len(added) != len(set(added)) or set(core) & set(added):
        raise ValueError("Fixed-core integration requires the original three and distinct additions")
    if options["core_rule"] not in {"mse", "performance"}:
        raise ValueError("core_rule must be mse or performance")
    values = [options[k] for k in ["core_member_floor", "minimum_core_total", "minimum_added_member"]]
    if not np.isfinite(values).all() or not (0 < values[0] < 1/3 and 0 < values[1] < 1 and values[2] > 0):
        raise ValueError("Invalid fixed-core weight floors")
    if values[1] + len(added)*values[2] >= 1:
        raise ValueError("Fixed-core lower bounds leave no free weight")


def mse_simplex(matrix, targets, lower):
    matrix, targets, lower = np.asarray(matrix, float), np.asarray(targets, float).ravel(), np.asarray(lower, float)
    if (matrix.ndim != 2 or matrix.shape != (len(targets), len(lower)) or len(targets) < 2
            or not np.isfinite(matrix).all() or not np.isfinite(targets).all()
            or not np.isfinite(lower).all() or np.any(lower < 0) or lower.sum() >= 1
            or np.var(targets) <= 0):
        raise ValueError("Invalid aligned OOF matrix, targets or weight bounds")
    scale = float(np.var(targets))
    initial = lower + (1-lower.sum())/len(lower)
    result = minimize(lambda w: np.mean((matrix@w-targets)**2)/scale, initial,
        jac=lambda w: 2*matrix.T@(matrix@w-targets)/len(targets)/scale,
        method="SLSQP", bounds=[(float(v), 1.) for v in lower],
        constraints={"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(len(w))},
        options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"OOF weight fitting failed: {result.message}")
    # Project sub-epsilon roundoff to valid bounds without changing the rule.
    free = np.maximum(result.x-lower, 0)
    weight = lower + free/free.sum()*(1-lower.sum())
    if not np.all(weight >= lower) or not np.isclose(weight.sum(), 1., rtol=0, atol=1e-10):
        raise RuntimeError("Invalid fitted OOF weights")
    return weight


def fit_fixed_core_strategy(oof_predictions, targets, model_keys, options):
    """This API accepts no outer inputs, labels, noise conditions or SNR."""
    validate_options(options)
    model_keys = list(model_keys)
    if set(oof_predictions) != set(model_keys):
        raise ValueError("OOF predictions and configured models differ")
    core_keys, added_keys = options["core_keys"], options["added_keys"]
    if set(core_keys+added_keys)-set(model_keys):
        raise ValueError("Required fixed-core models are missing")
    y = np.asarray(targets, float).ravel()
    core_matrix = np.column_stack([oof_predictions[k] for k in core_keys])
    if options["core_rule"] == "mse":
        core = mse_simplex(core_matrix, y, [options["core_member_floor"]]*len(core_keys))
    else:
        if not np.isfinite(core_matrix).all() or np.var(y) <= 0:
            raise ValueError("Invalid performance OOF data")
        errors = np.mean((core_matrix-y[:, None])**2, axis=0)/np.var(y)
        core = 1/np.maximum(errors, 1e-6)
        core /= core.sum()
    core_prediction = core_matrix@core
    if added_keys:
        blocks = np.column_stack([core_prediction, *[oof_predictions[k] for k in added_keys]])
        lower = [options["minimum_core_total"]]+[options["minimum_added_member"]]*len(added_keys)
        block_weights = mse_simplex(blocks, y, lower)
    else:
        block_weights = np.asarray([1.])
    weights = {key: 0. for key in model_keys}
    weights.update({key: float(w*block_weights[0]) for key, w in zip(core_keys, core)})
    weights.update({key: float(w) for key, w in zip(added_keys, block_weights[1:])})
    fitted = sum(np.asarray(oof_predictions[k])*w for k, w in weights.items())
    diagnostic = {"core_weights": {k: float(w) for k, w in zip(core_keys, core)},
        "block_weights": {"original_three": float(block_weights[0]),
                          **{k: float(w) for k, w in zip(added_keys, block_weights[1:])}},
        "active_members": sum(w > 0 for w in weights.values()),
        "oof_fit_rmse_w_m2": float(np.sqrt(np.mean((fitted-y)**2))),
        "selection_scope": "training OOF integration fit diagnostic, not outer performance",
        "outer_labels_used": False, "options": options}
    return weights, diagnostic
