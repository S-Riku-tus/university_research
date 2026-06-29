def _param_tag(value):
    if isinstance(value, float):
        text = f"{value:.8f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


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


def expand_parameter_sets(parameter_sets_config):
    if isinstance(parameter_sets_config, list):
        return parameter_sets_config

    if not isinstance(parameter_sets_config, dict):
        raise TypeError("parameter_sets must be a list or a parameter-set config dict.")

    config_type = parameter_sets_config.get("type")
    if config_type == "keras_grid":
        return build_keras_grid_parameter_sets(
            lrs=parameter_sets_config["lrs"],
            batch_sizes=parameter_sets_config["batch_sizes"],
            model_keys=parameter_sets_config["model_keys"],
            default_keras=parameter_sets_config.get("default_keras"),
            models=parameter_sets_config.get("models"),
        )

    raise ValueError(f"Unknown parameter_sets config type: {config_type}")
