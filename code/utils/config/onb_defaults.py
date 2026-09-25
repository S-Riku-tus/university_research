"""Stable defaults for the ONB regression pipeline.

The main run script should contain the experimental choices that are expected
to change between runs.  Fixed acoustic-selection mechanics, output mechanics,
XAI diagnostics, and the model registry live here so
they do not obscure those choices.  Resolved defaults are still written to
every run manifest.
"""

from copy import deepcopy
from datetime import datetime


def _build_random_forest(model_maker, **params):
    return model_maker.random_forest(**params)


def _build_cnn_transformer(model_maker, **params):
    return model_maker.cnn_transformer_v2(**params)


def _build_alexnet(model_maker, **params):
    return model_maker.alexnet(**params)


def onb_model_specs():
    """Return a fresh registry for the fixed production model structures."""
    return [
        {
            "key": "randomforest",
            "label": "RandomForest",
            "kind": "sklearn",
            "builder": _build_random_forest,
            "random_state_from_run": True,
        },
        {
            "key": "conformer",
            "label": "Conformer",
            "kind": "keras",
            "builder": _build_cnn_transformer,
            "input_axes_assumption": [
                "time_frame", "frequency_bin", "channel"
            ],
            "architecture": {
                "front_end": "alexnet_like_cnn",
                "input_transform": "log1p(power / 1e-12)",
                "sequence_length_after_cnn": 7,
                "model_dim": 64,
                "num_heads": 4,
                "attention_key_dim_per_head": 16,
                "ff_dim": 256,
                "num_transformer_blocks": 2,
                "dropout": 0.1,
                "encoder": "transformer_encoder",
                "pooling": "GlobalAveragePooling1D",
            },
        },
        {
            "key": "alexnet",
            "label": "AlexNet",
            "kind": "keras",
            "builder": _build_alexnet,
            "architecture": {
                "input_transform": "log1p(power / 1e-12)",
                "regression_head": "Flatten-Dense4096-Dense4096",
            },
        },
    ]


def default_output_config(now=None):
    now = now or datetime.now()
    save_date = now.strftime("%Y%m%d")
    return {
        "save_date": save_date,
        # Group every analysis performed on the same day without mixing the
        # date into the study name. The final onb_* name is derived later.
        # 実行条件を表す短い系列名はscoped_result_job()でonb_*として付ける。
        "result_date_dir": f"{save_date}/onb",
        "save_fold_predictions": True,
        "save_tuning_summary": True,
        # Final fitted state is large (especially AlexNet), so historical runs
        # keep the previous lightweight behavior unless a condition opts in.
        "save_fitted_artifacts": False,
        "verify_reloaded_artifacts": True,
        "resume_completed_runs": True,
        "noise_trend_plots": {
            "enabled": True,
            "ensemble_strategy_names": "all",
            "metrics": ["r2", "roc_auc_cont"],
            "formats": ["png", "pdf"],
        },
    }


DEFAULT_ACOUSTIC_SELECTION_CONFIG = {
    # Only peak_height_threshold is an experiment-by-experiment choice in the
    # main script.  None means that selection is disabled.
    "mode": "peak_height",
    "features_csv": "experiments/2026-09-16_peak_height_selection/peak_features.csv",
    "feature": "peak_2100_2500_psd",
    "peak_height_threshold": None,
    "peak_height_threshold_by_experiment": {},
    "apply_max_heat_flux_by_experiment": {},
}


DEFAULT_EXPLAINABILITY_CONFIG = {
    "enabled": True,
    "max_samples_per_fold": 5,
    "ig_steps": 64,
    "ig_max_steps": 4096,
    "ig_batch_size": 8,
    # The neural models learn after LogPowerCompression.  Integrating in that
    # feature space avoids the near-zero raw-power singularity while retaining
    # an input-resolution attribution map.  raw_power remains available for
    # reproducing historical outputs.
    "ig_path_space": "log_power",
    "ig_rtol": 1e-3,
    "ig_atol": 1e-6,
    "ig_map_rtol": 1e-2,
    "methods_by_model": {
        "randomforest": ["tree_shap_pca", "group_occlusion"],
        "conformer": ["integrated_gradients", "group_occlusion"],
        "alexnet": [
            "integrated_gradients", "grad_cam", "group_occlusion"
        ],
    },
    "frequency_bands_hz": [
        [0, 256],
        [256, 512],
        [512, 1000],
        [1000, 2000],
        [2000, 5000],
        [5000, 10000],
        [10000, 15000],
        [15000, 22000],
    ],
    "time_extent_seconds": 1.0,
    "onb_band_frac": 0.10,
    "baseline_value": 0.0,
    "curve_fractions": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
    "stability": {
        "enabled": True,
        "methods": ["integrated_gradients"],
        "repeats": 2,
        "noise_fraction": 0.01,
        "clip_nonnegative": True,
        "random_seed": 42,
    },
    "sanity_check": {
        "enabled": True,
        "methods": ["integrated_gradients"],
        "random_seed": 42,
    },
    "retrain_completed_runs_for_xai": False,
}


def _merge_dict(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def apply_onb_defaults(config, now=None):
    """Resolve stable defaults while allowing explicit, local overrides."""
    defaults = {
        "acoustic_selection": deepcopy(DEFAULT_ACOUSTIC_SELECTION_CONFIG),
        "output": default_output_config(now),
        "explainability": deepcopy(DEFAULT_EXPLAINABILITY_CONFIG),
    }
    resolved = _merge_dict(defaults, deepcopy(config))
    selection_override = config.get("acoustic_selection", {})
    if "enabled" not in selection_override:
        resolved["acoustic_selection"]["enabled"] = (
            resolved["acoustic_selection"].get("peak_height_threshold") is not None
        )
    return resolved
