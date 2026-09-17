from copy import deepcopy

from utils.ensemble.crossfit_stacking import CROSSFIT_DEFAULTS


# Stable strategy definitions live here so experiment scripts only choose names.
# Add or revise a reusable ensemble profile in this catalog, not in a run script.
ENSEMBLE_STRATEGY_CATALOG = {
    "simple_equal": {
        "label": "Ensemble simple equal",
        "strategy": "simple",
    },
    "inner_holdout": {
        "label": "Ensemble",
        "strategy": "inner_holdout",
    },
    "performance_kfold": {
        "label": "Ensemble individual-performance KFold",
        "strategy": "performance_kfold",
    },
    "subset_equal_cv": {
        "label": "Ensemble crossfit subset equal",
        "strategy": "subset_equal_cv",
        "crossfit": {**CROSSFIT_DEFAULTS},
    },
    "crossfit_wav_stack": {
        "label": "Ensemble crossfit WAV stack",
        "strategy": "crossfit_wav_stack",
        "crossfit": {**CROSSFIT_DEFAULTS},
    },
    "crossfit_shrinkage_stack": {
        "label": "Ensemble crossfit shrinkage stack",
        "strategy": "crossfit_shrinkage_stack",
        "crossfit": {**CROSSFIT_DEFAULTS, "regularization": 0.1},
    },
    "val_fold_legacy": {
        "label": "Ensemble validation-fold legacy",
        "strategy": "val_fold_legacy",
    },
}


# These mechanics are shared by experiments and are deliberately not exposed in
# the ONB run configuration. The resolved values are still saved in manifests.
ENSEMBLE_RUNTIME_DEFAULTS = {
    "reference_model": "randomforest",
    "inner_holdout_frac": 0.20,
    "combine": "mean",
}


def available_ensemble_strategy_names():
    return list(ENSEMBLE_STRATEGY_CATALOG)


def is_supported_result_key(model_key):
    """Keep single models and supported ensembles when reading historical results."""
    prefix = "ensemble__"
    return (not model_key.startswith(prefix)
            or model_key[len(prefix):] in ENSEMBLE_STRATEGY_CATALOG)


def resolve_ensemble_selection(selection_config):
    """Resolve a run script's strategy-name selection into full mechanics."""
    selection_config = dict(selection_config or {})
    selected_names = selection_config.get("enabled_strategy_names", [])
    if not isinstance(selected_names, (list, tuple)):
        raise TypeError("ensemble.enabled_strategy_names must be a list or tuple.")
    selected_names = [str(name) for name in selected_names]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("ensemble.enabled_strategy_names contains duplicates.")

    unknown = [
        name for name in selected_names
        if name not in ENSEMBLE_STRATEGY_CATALOG
    ]
    if unknown:
        raise ValueError(
            f"Unknown ensemble strategy names: {unknown}. Available: "
            f"{available_ensemble_strategy_names()}"
        )

    strategies = []
    for name in selected_names:
        item = deepcopy(ENSEMBLE_STRATEGY_CATALOG[name])
        item["name"] = name
        strategies.append(item)

    resolved = {
        "enabled": bool(selected_names),
        "strategies": strategies,
        **deepcopy(ENSEMBLE_RUNTIME_DEFAULTS),
    }
    # Selecting the explicitly named legacy diagnostic is itself the opt-in.
    resolved["allow_leaky_strategies"] = "val_fold_legacy" in selected_names
    return resolved
