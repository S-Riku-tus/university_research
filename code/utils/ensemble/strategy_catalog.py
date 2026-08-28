from copy import deepcopy


# Stable strategy definitions live here so experiment scripts only choose names.
# Add or revise a reusable ensemble profile in this catalog, not in a run script.
ENSEMBLE_STRATEGY_CATALOG = {
    "simple_equal": {
        "label": "Ensemble simple equal",
        "strategy": "simple",
    },
    "fixed_rf98_even": {
        "label": "Ensemble RF98 + 1 + 1",
        "strategy": "fixed",
        "fixed_weights": {
            "rf": 0.98,
            "cnntf_v2_gap": 0.01,
            "alexnet": 0.01,
        },
    },
    "prediction_max": {
        "label": "Ensemble prediction max",
        "strategy": "max",
    },
    "inner_holdout": {
        "label": "Ensemble inner holdout",
        "strategy": "inner_holdout",
    },
    "val_fold_legacy": {
        "label": "Ensemble validation-fold legacy",
        "strategy": "val_fold_legacy",
    },
}


# These mechanics are shared by experiments and are deliberately not exposed in
# the ONB run configuration. The resolved values are still saved in manifests.
ENSEMBLE_RUNTIME_DEFAULTS = {
    "reference_model": "rf",
    "inner_holdout_frac": 0.20,
    "combine": "mean",
}


def available_ensemble_strategy_names():
    return list(ENSEMBLE_STRATEGY_CATALOG)


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

    primary_name = selection_config.get("primary_strategy_name")
    if selected_names:
        primary_name = primary_name or selected_names[0]
        if primary_name not in selected_names:
            raise ValueError(
                "ensemble.primary_strategy_name must be included in "
                f"enabled_strategy_names; got {primary_name!r}."
            )
    elif primary_name is not None:
        raise ValueError(
            "ensemble.primary_strategy_name must be None when no strategy is enabled."
        )

    strategies = []
    for name in selected_names:
        item = deepcopy(ENSEMBLE_STRATEGY_CATALOG[name])
        item["name"] = name
        strategies.append(item)

    resolved = {
        "enabled": bool(selected_names),
        "primary_strategy": primary_name,
        "strategies": strategies,
        **deepcopy(ENSEMBLE_RUNTIME_DEFAULTS),
    }
    # Selecting the explicitly named legacy diagnostic is itself the opt-in.
    resolved["allow_leaky_strategies"] = "val_fold_legacy" in selected_names
    return resolved
