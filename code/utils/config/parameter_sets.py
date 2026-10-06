import hashlib
import json
from itertools import product


def _param_tag(value):
    if isinstance(value, float):
        text = f"{value:.8f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def format_param_value(value):
    return _param_tag(value)


KERAS_TRAINING_PARAM_KEYS = {
    "lr",
    "batch_size",
    "fit_verbose",
    "progress_interval_epochs",
    "min_batch_size",
    "accept_partial_min_epochs",
}


MODEL_TAG_ALIASES = {
    "randomforest": "rf",
    "conformer": "ctf",
    "alexnet": "alex",
}


PARAMETER_TAG_ALIASES = {
    "learning_rate": "lr",
    "batch_size": "bs",
    "n_estimators": "ne",
    "max_depth": "md",
    "subsample": "ss",
    "colsample_bynode": "cs",
}


def _grid_values(model_key, parameter_name, values):
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"model_grids['{model_key}']['{parameter_name}'] must be a list or tuple."
        )
    if not values:
        raise ValueError(
            f"model_grids['{model_key}']['{parameter_name}'] must not be empty."
        )
    return list(values)


def _independent_grid_tag(model_key, parameter_names, params):
    """Build a readable, collision-resistant tag that stays short on Windows."""
    payload = json.dumps(
        {"model_key": model_key, "params": params},
        sort_keys=True,
        ensure_ascii=True,
        default=repr,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:5]
    model_tag = MODEL_TAG_ALIASES.get(model_key, model_key[:4])
    readable_parts = [model_tag]
    for name in parameter_names[:2]:
        alias = PARAMETER_TAG_ALIASES.get(name, name[:2])
        readable_parts.append(f"{alias}{_param_tag(params[name])}")
    # run_dir_name currently keeps 24 characters for this tag. Reserve the
    # suffix for the digest so even long/custom parameter names remain unique.
    readable = "_".join(readable_parts)[:18].rstrip("_")
    return f"{readable}_{digest}"


def _active_grid_tag(active_model_keys, model_params):
    """Build a short unique tag for a joint run of all active models."""
    payload = json.dumps(
        {"active_model_keys": active_model_keys, "models": model_params},
        sort_keys=True,
        ensure_ascii=True,
        default=repr,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:7]
    model_tag = "-".join(
        MODEL_TAG_ALIASES.get(key, key[:4]) for key in active_model_keys
    )
    readable = f"active_{model_tag}"[:22].rstrip("-_")
    return f"{readable}_{digest}"


def build_active_model_grid_parameter_sets(
    model_grids,
    active_model_keys,
    default_keras=None,
):
    """Expand parameter candidates for the globally active models only.

    Every returned set contains parameters for every active model, so all
    active models are trained together and can be ensembled. Inactive model
    grids are deliberately ignored. Singleton candidate lists therefore
    produce one fixed-parameter run; multiple candidates produce a Cartesian
    tuning grid over the active models.
    """
    if not isinstance(model_grids, dict) or not model_grids:
        raise ValueError("model_grids must be a non-empty dict.")

    active_model_keys = list(active_model_keys or [])
    if not active_model_keys:
        raise ValueError("active_model_keys must select at least one model.")
    if len(active_model_keys) != len(set(active_model_keys)):
        raise ValueError(f"active_model_keys contains duplicates: {active_model_keys}")

    missing_grids = [key for key in active_model_keys if key not in model_grids]
    if missing_grids:
        raise ValueError(
            "model_grids must define every active model; missing "
            f"{missing_grids}."
        )

    candidates_by_model = []
    for model_key in active_model_keys:
        grid = model_grids[model_key]
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"model_grids['{model_key}'] must be a non-empty dict.")
        parameter_names = list(grid)
        value_lists = [
            _grid_values(model_key, name, grid[name])
            for name in parameter_names
        ]
        candidates_by_model.append([
            dict(zip(parameter_names, values))
            for values in product(*value_lists)
        ])

    default_keras = dict(default_keras or {})
    parameter_sets = []
    seen_tags = set()
    for model_candidates in product(*candidates_by_model):
        model_params = {
            key: dict(params)
            for key, params in zip(active_model_keys, model_candidates)
        }
        name_parts = []
        for model_key in active_model_keys:
            params = model_params[model_key]
            param_text = "__".join(
                f"{key}={value}" for key, value in params.items()
            )
            name_parts.append(f"{model_key}__{param_text}")
        name = "___".join(name_parts)
        tag = _active_grid_tag(active_model_keys, model_params)
        if tag in seen_tags:
            raise ValueError(f"Generated duplicate parameter-set tag: {tag}")
        seen_tags.add(tag)
        parameter_set = {
            "name": name,
            "tag": tag,
            "models": model_params,
        }
        if default_keras:
            parameter_set["default_keras"] = dict(default_keras)
        parameter_sets.append(parameter_set)

    return parameter_sets


def build_independent_model_grid_parameter_sets(model_grids, default_keras=None):
    """Expand one Cartesian grid per model without multiplying models together.

    Each returned parameter set selects exactly one model. This prevents a
    three-model joint grid from exploding combinatorially and makes every row
    in the tuning summary attributable to one model's hyperparameters.
    """
    if not isinstance(model_grids, dict) or not model_grids:
        raise ValueError("model_grids must be a non-empty dict.")

    default_keras = dict(default_keras or {})
    parameter_sets = []
    seen_tags = set()
    for model_key, grid in model_grids.items():
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"model_grids['{model_key}'] must be a non-empty dict.")
        parameter_names = list(grid)
        value_lists = [
            _grid_values(model_key, name, grid[name])
            for name in parameter_names
        ]
        for values in product(*value_lists):
            params = dict(zip(parameter_names, values))
            name = model_key + "__" + "__".join(
                f"{key}={value}" for key, value in params.items()
            )
            tag = _independent_grid_tag(model_key, parameter_names, params)
            if tag in seen_tags:
                raise ValueError(f"Generated duplicate parameter-set tag: {tag}")
            seen_tags.add(tag)
            parameter_set = {
                "name": name,
                "tag": tag,
                "model_key": model_key,
                "models": {model_key: params},
            }
            if default_keras:
                parameter_set["default_keras"] = dict(default_keras)
            parameter_sets.append(parameter_set)

    return parameter_sets


def build_keras_grid_parameter_sets(
    lrs,
    batch_sizes,
    model_keys,
    default_keras=None,
    models=None,
):
    default_keras = dict(default_keras or {})
    base_models = {}
    for model_key, params in (models or {}).items():
        if params is None:
            base_models[model_key] = {}
        elif isinstance(params, dict):
            base_models[model_key] = dict(params)
        else:
            raise TypeError(f"models['{model_key}'] must be a dict or None.")

    parameter_sets = []

    for lr in lrs:
        for batch_size in batch_sizes:
            set_models = {
                model_key: dict(params)
                for model_key, params in base_models.items()
            }
            for model_key in model_keys:
                params = dict(set_models.get(model_key, {}))
                params.update({"lr": lr, "batch_size": batch_size})
                set_models[model_key] = params

            parameter_set = {
                "name": f"lr{_param_tag(lr)}_bs{batch_size}",
                "models": set_models,
            }
            if default_keras:
                parameter_set["default_keras"] = dict(default_keras)
            parameter_sets.append(parameter_set)

    return parameter_sets


def expand_parameter_sets(parameter_sets_config, active_model_keys=None):
    if isinstance(parameter_sets_config, list):
        return parameter_sets_config

    if not isinstance(parameter_sets_config, dict):
        raise TypeError("parameter_sets must be a list or a parameter-set config dict.")

    config_type = parameter_sets_config.get("type")
    if config_type == "active_model_grid":
        return build_active_model_grid_parameter_sets(
            model_grids=parameter_sets_config["model_grids"],
            active_model_keys=active_model_keys,
            default_keras=parameter_sets_config.get("default_keras"),
        )

    if config_type == "keras_grid":
        return build_keras_grid_parameter_sets(
            lrs=parameter_sets_config["lrs"],
            batch_sizes=parameter_sets_config["batch_sizes"],
            model_keys=parameter_sets_config["model_keys"],
            default_keras=parameter_sets_config.get("default_keras"),
            models=parameter_sets_config.get("models"),
        )

    if config_type == "independent_model_grid":
        return build_independent_model_grid_parameter_sets(
            model_grids=parameter_sets_config["model_grids"],
            default_keras=parameter_sets_config.get("default_keras"),
        )

    raise ValueError(f"Unknown parameter_sets config type: {config_type}")


def build_parameter_execution_plan(parameter_sets_config, active_model_keys):
    """Choose a fixed run or model-wise search from the candidate-list lengths.

    ``active_model_grid`` is the single user-facing configuration for both
    modes.  When every active-model candidate list is a singleton, one joint
    parameter set is returned for the normal ONB experiment.  As soon as any
    list creates more than one joint combination, candidates are instead
    expanded independently per model for training-only OOF selection.  This
    avoids multiplying RF, Conformer, and AlexNet candidates together.
    """
    if not isinstance(parameter_sets_config, dict):
        raise TypeError(
            "Automatic parameter execution requires parameter_sets to be a config dict."
        )
    if parameter_sets_config.get("type") != "active_model_grid":
        raise ValueError(
            "Automatic parameter execution requires "
            "parameter_sets.type='active_model_grid'."
        )

    active_model_keys = list(active_model_keys or [])
    if not active_model_keys:
        raise ValueError("active_model_keys must select at least one model.")
    if len(active_model_keys) != len(set(active_model_keys)):
        raise ValueError(f"active_model_keys contains duplicates: {active_model_keys}")
    model_grids = parameter_sets_config["model_grids"]
    missing_grids = [key for key in active_model_keys if key not in model_grids]
    if missing_grids:
        raise ValueError(
            "model_grids must define every active model; missing "
            f"{missing_grids}."
        )
    active_model_grids = {
        model_key: model_grids[model_key]
        for model_key in active_model_keys
    }
    search_config = {
        "type": "independent_model_grid",
        "model_grids": active_model_grids,
    }
    if parameter_sets_config.get("default_keras"):
        search_config["default_keras"] = dict(
            parameter_sets_config["default_keras"]
        )
    independent_parameter_sets = expand_parameter_sets(search_config)
    search_enabled = len(independent_parameter_sets) > len(active_model_keys)
    if search_enabled:
        return independent_parameter_sets, True
    return expand_parameter_sets(
        parameter_sets_config,
        active_model_keys=active_model_keys,
    ), False


def build_frequency_parameter_execution_plans(
    parameter_sets_config,
    active_model_keys,
    max_freq_hz_list,
):
    """Resolve one fixed-run or tuning plan for each selected max frequency.

    ``max_freq_active_model_grid`` keeps the existing candidate-list behavior,
    but scopes ``model_grids`` by the input-frequency directory.  Unselected
    frequency entries may remain in the config, making it possible to switch
    frequencies only by editing ``data.max_freq_hz_list``.

    The historical ``active_model_grid`` form remains supported and applies
    the same grid to every selected frequency.
    """
    max_freq_hz_list = list(max_freq_hz_list or [])
    if not max_freq_hz_list:
        raise ValueError("max_freq_hz_list must select at least one frequency.")
    if len(max_freq_hz_list) != len(set(max_freq_hz_list)):
        raise ValueError(
            f"max_freq_hz_list contains duplicates: {max_freq_hz_list}"
        )
    if not isinstance(parameter_sets_config, dict):
        raise TypeError(
            "Frequency-aware parameter execution requires parameter_sets "
            "to be a config dict."
        )

    config_type = parameter_sets_config.get("type")
    if config_type == "active_model_grid":
        parameter_sets, search_enabled = build_parameter_execution_plan(
            parameter_sets_config,
            active_model_keys,
        )
        return {
            max_freq_hz: {
                "parameter_sets": parameter_sets,
                "parameter_search_enabled": search_enabled,
            }
            for max_freq_hz in max_freq_hz_list
        }

    if config_type != "max_freq_active_model_grid":
        raise ValueError(
            "Frequency-aware parameter execution requires parameter_sets.type "
            "to be 'max_freq_active_model_grid' or 'active_model_grid'."
        )

    grids_by_frequency = parameter_sets_config.get("by_max_freq_hz")
    if not isinstance(grids_by_frequency, dict) or not grids_by_frequency:
        raise ValueError(
            "parameter_sets.by_max_freq_hz must be a non-empty dict."
        )
    missing_frequencies = [
        max_freq_hz
        for max_freq_hz in max_freq_hz_list
        if max_freq_hz not in grids_by_frequency
    ]
    if missing_frequencies:
        raise ValueError(
            "parameter_sets.by_max_freq_hz is missing selected frequencies: "
            f"{missing_frequencies}."
        )

    default_keras = parameter_sets_config.get("default_keras")
    plans = {}
    for max_freq_hz in max_freq_hz_list:
        model_grids = grids_by_frequency[max_freq_hz]
        if not isinstance(model_grids, dict) or not model_grids:
            raise ValueError(
                f"parameter_sets.by_max_freq_hz['{max_freq_hz}'] must be "
                "a non-empty model-grid dict."
            )
        frequency_config = {
            "type": "active_model_grid",
            "model_grids": model_grids,
        }
        if default_keras:
            frequency_config["default_keras"] = dict(default_keras)
        parameter_sets, search_enabled = build_parameter_execution_plan(
            frequency_config,
            active_model_keys,
        )
        plans[max_freq_hz] = {
            "parameter_sets": parameter_sets,
            "parameter_search_enabled": search_enabled,
        }
    return plans


def resolve_parameter_set(enabled_specs, parameter_set):
    resolved = []
    default_keras = parameter_set.get("default_keras", {})
    per_model = parameter_set.get("models", {})
    requested_key = parameter_set.get("model_key")
    if requested_key is None:
        selected_specs = list(enabled_specs)
    else:
        spec_by_key = {spec["key"]: spec for spec in enabled_specs}
        if requested_key not in spec_by_key:
            raise ValueError(
                "parameter_set.model_key contains an unknown model: "
                f"{requested_key}"
            )
        selected_specs = [spec_by_key[requested_key]]
    for spec in selected_specs:
        resolved_spec = dict(spec)
        if resolved_spec["kind"] == "keras":
            params = dict(default_keras)
            params.update(per_model.get(resolved_spec["key"], {}))
            missing = [name for name in ("lr", "batch_size") if name not in params]
            if missing:
                raise ValueError(
                    f"PARAMETER_SETS entry '{parameter_set.get('name', '<unnamed>')}' "
                    f"does not define {missing} for {resolved_spec['key']}."
                )
            builder_params = dict(resolved_spec.get("builder_params", {}))
            for name, value in params.items():
                if name in KERAS_TRAINING_PARAM_KEYS:
                    resolved_spec[name] = value
                else:
                    builder_params[name] = value
            if builder_params:
                resolved_spec["builder_params"] = builder_params
        elif resolved_spec["kind"] in {"sklearn", "sklearn_summary"}:
            params = {}
            params.update(per_model.get(resolved_spec["key"], {}))
            if params:
                resolved_spec["builder_params"] = params
        resolved.append(resolved_spec)
    return resolved


def parameter_set_tag(parameter_set, resolved_specs, safe_tag_func):
    if parameter_set.get("tag"):
        return safe_tag_func(parameter_set["tag"])
    if parameter_set.get("name"):
        return safe_tag_func(parameter_set["name"])
    parts = []
    for spec in resolved_specs:
        if spec["kind"] == "keras":
            parts.append(
                f"{spec['key']}_lr{format_param_value(spec['lr'])}"
                f"_bs{format_param_value(spec['batch_size'])}"
            )
    return safe_tag_func("__".join(parts))
