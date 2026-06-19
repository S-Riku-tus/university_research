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
    name_prefix="k",
    default_keras=None,
):
    default_keras = dict(default_keras or {"early_stopping": True})
    parameter_sets = []

    for lr in lrs:
        for batch_size in batch_sizes:
            parameter_sets.append(
                {
                    "name": f"{name_prefix}_lr{_param_tag(lr)}_bs{batch_size}",
                    "default_keras": dict(default_keras),
                    "models": {
                        model_key: {"lr": lr, "batch_size": batch_size}
                        for model_key in model_keys
                    },
                }
            )

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
            name_prefix=parameter_sets_config.get("name_prefix", "k"),
            default_keras=parameter_sets_config.get("default_keras"),
        )

    raise ValueError(f"Unknown parameter_sets config type: {config_type}")
