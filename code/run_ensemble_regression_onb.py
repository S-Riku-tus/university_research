"""
run_ensemble_regression_onb.py

荳ｭ蠢・せ繧ｯ繝ｪ繝励ヨ 3.run_ensemble_ROC_100%_analysis.py 縺ｮ菴懊ｊ逶ｴ縺礼沿縲・
2026-06-12 縺ｮ遐皮ｩｶ險育判 (遐皮ｩｶ騾ｲ謐怜ｱ蜻・2026/612/莉雁ｾ後・遐皮ｩｶ險育判_2026-06-12.md) 縺ｮ
Phase 0縲瑚ｩ穂ｾ｡霆ｸ縺ｨ繧ｳ繝ｼ繝峨・謨ｴ逅・阪↓蟇ｾ蠢懊☆繧九・

譌ｧ繧ｹ繧ｯ繝ｪ繝励ヨ縺ｫ蟇ｾ縺励※縲∵ｬ｡縺ｮ3轤ｹ繧堤峩縺励◆縺・∴縺ｧ菴懊ｊ逶ｴ縺励※縺・ｋ縲・

  竭 繝｢繝・Ν蜷阪・蜿悶ｊ驕輔∴繧偵↑縺上☆
     譌ｧ繧ｳ繝ｼ繝峨・ RandomForest 縺ｮ莠域ｸｬ繧・alexnet_pred 縺ｨ縺・≧螟画焚縺ｫ蜈･繧後∽ｻ･髯阪・
     R2/AUC/繧ｰ繝ｩ繝・txt 縺吶∋縺ｦ縺ｫ "AlexNet" 縺ｨ縺・≧繝ｩ繝吶Ν縺ｧ險倬鹸縺励※縺・◆縲・
     譛ｬ繧ｹ繧ｯ繝ｪ繝励ヨ縺ｧ縺ｯ MODEL_SPECS 縺ｨ縺・≧繝ｬ繧ｸ繧ｹ繝医Μ縺ｧ蜷・Δ繝・Ν繧貞ｮ夂ｾｩ縺励・
     繝ｩ繝吶Ν縺悟ｿ・★螳滉ｽ薙・繝｢繝・Ν縺ｫ霑ｽ蠕薙☆繧九ｈ縺・↓縺励◆縲ゅΔ繝・Ν縺ｮ蟾ｮ縺玲崛縺医・
     譛牙柑/辟｡蜉ｹ蛹悶・ MODEL_SPECS 繧堤ｷｨ髮・☆繧九□縺代〒貂医・縲・

  竭｡ AUC 繧偵碁｣邯壹せ繧ｳ繧｢迚医阪→縲御ｺ悟､蛹門ｾ後・蛻・｡樊欠讓吶阪↓蛻・￠繧・
     譌ｧ繧ｳ繝ｼ繝峨・莠域ｸｬ繧帝明蛟､縺ｧ 0/1 蛹悶＠縺ｦ縺九ｉ ROC 繧定ｨ育ｮ励＠縺ｦ縺・◆縺溘ａ縲ヽOC 縺・
     2轤ｹ縺励°謖√◆縺・AUC 縺悟ｮ溯ｳｪ繝舌Λ繝ｳ繧ｹ邊ｾ蠎ｦ縺ｫ縺ｪ縺｣縺ｦ縺・◆縲よ悽繧ｹ繧ｯ繝ｪ繝励ヨ縺ｧ縺ｯ
       - 騾｣邯壹せ繧ｳ繧｢迚・AUC: 莠域ｸｬ辭ｱ豬∵據繧偵◎縺ｮ縺ｾ縺ｾ繧ｹ繧ｳ繧｢縺ｫ縺励◆ ROC-AUC / PR-AUC
       - 莠悟､蛹門ｾ後・蛻・｡樊欠讓・ Accuracy / Precision / Recall / F1
     繧貞・縺代※邂怜・縺吶ｋ縲よ立譚･縺ｮ莠悟､蛹・AUC 繧ょｾ梧婿豈碑ｼ・畑縺ｫ谿九＠縺ｦ縺ゅｋ縲・

  竭｢ 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν驥阪∩縺ｮ豎ｺ繧∵婿繧帝∈謚槫ｼ上↓縺吶ｋ (繝ｪ繝ｼ繧ｯ蟇ｾ遲・
     譌ｧ繧ｳ繝ｼ繝峨・隧穂ｾ｡蟇ｾ雎｡縺ｧ縺ゅｋ讀懆ｨｼ fold (y_val) 縺ｮ隱､蟾ｮ縺九ｉ驥阪∩繧呈ｱｺ繧√※縺翫ｊ縲・
     Ensemble methods are selected by catalog name in VALIDATION_CONFIG.
     Their weights, leakage rules, and execution mechanics live in utils/ensemble/.
     繧貞・繧頑崛縺医ｉ繧後ｋ縲・

  竭｣ 繝・・繧ｿ繝代せ縺ｯ蛻･繝槭す繝ｳ驕狗畑縺ｮ縺溘ａ縺昴・縺ｾ縺ｾ (譌ｧ繧ｳ繝ｼ繝峨→蜷後§繝上・繝峨さ繝ｼ繝・

譌ｧ繧ｹ繧ｯ繝ｪ繝励ヨ (code/3.run_ensemble_ROC_100%_analysis.py) 縺ｯ蜀咲樟諤ｧ縺ｮ縺溘ａ谿九☆縲・

蜀榊茜逕ｨ蜿ｯ閭ｽ縺ｪ蜃ｦ逅・(謖・ｨ呵ｨ育ｮ励・蟄ｦ鄙・莠域ｸｬ繝ｻ驥阪∩莉倥￠繝ｻ菴懷峙) 縺ｯ縲∵里蟄倥・ utils 譁ｹ驥昴↓
蜷医ｏ縺帙※逕ｨ騾泌挨縺ｮ繧ｯ繝ｩ繧ｹ縺ｫ蛻・屬縺励※縺ゅｋ縲よ悽繝輔ぃ繧､繝ｫ縺ｫ縺ｯ縺薙・螳滄ｨ灘崋譛峨・險ｭ螳壹→
main() 縺ｮ繧ｪ繝ｼ繧ｱ繧ｹ繝医Ξ繝ｼ繧ｷ繝ｧ繝ｳ縺縺代ｒ鄂ｮ縺上・
    - 謖・ｨ呵ｨ育ｮ・   : utils/calculation/regression_detection_metrics.py
    - 蟄ｦ鄙・莠域ｸｬ   : utils/training/model_training.py
    - 驥阪∩莉倥￠    : utils/ensemble/ensemble_weighting.py
    - 菴懷峙        : utils/plotting/regression_plots.py
"""

import os
import gc
import time
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

# Avoid grabbing most of the GPU memory before the first model fit. This also
# makes OOM recovery by smaller batch sizes more reliable on Windows/TensorFlow.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import backend as K

from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.training.model_training import ModelTrainer
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.plotting.regression_plots import RegressionPlotter
from utils.config.parameter_sets import (
    expand_parameter_sets,
    parameter_set_tag,
    resolve_parameter_set,
)
from utils.explainability.training_integration import (
    aggregate_group_mask_comparison,
    explainability_condition_selected,
    explainability_outputs_complete,
    maybe_explain_trained_model,
    resolve_explainability_scope,
)
from utils.experiment.dataset_jobs import build_dataset_jobs as make_dataset_jobs
from utils.experiment.run_helpers import (
    append_tuning_summary,
    has_threshold,
    is_completed_run,
    makedirs as _makedirs,
    model_param_summary,
    open_text as _open_text,
    run_config_digest,
    run_dir_name,
    safe_tag,
    set_global_seed,
    write_run_manifest,
)


#######################################################################
#                              螟画焚縺ｮ謖・ｮ・
#######################################################################
# Validation controls: edit this block first.
# Edit this block first when changing an experiment.
#
# Current default:
# - purpose: compare predeclared ensemble strategies from the same fold models
# - data: 3 experiments x representative stress frequencies/noise conditions
# - models: RF + CNN/Transformer v2 GAP + AlexNet
# - ensemble: strategies selected by their reusable catalog names
# - explainability: model-specific methods on selected 22 kHz conditions only
#
# VALIDATION_CONFIG["ensemble"] only selects strategy names and a primary
# strategy. Reusable definitions and mechanics live in utils/ensemble/.

VALIDATION_CONFIG = {
    "run": {
        "smoke_test": False,
        "epochs": 200,
        "folds": 3,
        "smoke_epochs": 2,
        "smoke_folds": 2,
        "color_channel": 1,
        "random_seed": 42,
        "loop_parameter_sets": True,
    },
    "data": {
        "experiment_root_parts": ["Pool_boiling", "Subcooling_20_degrees", "0.3"],
        "noise_source": "waterflow",  # "waterflow" or "whitenoise"
        "chunk_seconds": 1,
        "experiment_names": [
            "2025.06.18_0.3_3",
            # "2025.07.09_0.3_1",
            # "2025.06.11_0.3_2",
        ],
        "max_freq_hz_list": [
            "maxfreq=3kHz",
            "maxfreq=5kHz",
            "maxfreq=10kHz",
            "maxfreq=15kHz",
            "maxfreq=22kHz",
        ],
        "noise_dir_names": [
            "heatflux_no_noise",
            # "heatflux_reference_SNR=0",
            # "heatflux_reference_SNR=-4",
            # "heatflux_reference_SNR=-8",
            # "heatflux_reference_SNR=-12",
            # "heatflux_reference_SNR=-16",
            # "heatflux_reference_SNR=-20",
        ],
        "data_source_dir_by_experiment": {
            "2025.06.11_0.3_2": "waterflow_20260817_1s",
            "2025.06.18_0.3_3": "waterflow_20260817_1s",
            "2025.07.09_0.3_1": "waterflow_20260817_1s",
        },
        "skip_missing_datasets": False,
    },
    "thresholds": {
        "by_experiment": {
            "2025.07.09_0.3_1": 275174.6640882674,
            "2025.06.11_0.3_2": 266907.6965,
            "2025.06.18_0.3_3": 271677.6816,
        },
        # False lets parameter tuning run before the ONB threshold is fixed.
        # Threshold-dependent metrics become NaN until the experiment threshold
        # is added above. Set True again when making ONB claims.
        "require_experiment_threshold": False,
        "onb_band_frac": 0.10,
    },
    "models": {
        "active_model_keys": ["cnntf_v2_gap", "alexnet"],
        # Only active_model_keys are executed. Inactive grids may remain below
        # as reusable settings and are ignored. Singleton lists mean one fixed
        # run; multiple values expand a Cartesian tuning grid over active models.
        "parameter_sets": {
            "type": "active_model_grid",
            "model_grids": {
                "rf": {
                    # XGBRFRegressor: tune model capacity first; keep sampling
                    # ratios fixed during this first-stage search.
                    "n_estimators": [300],
                    "max_depth": [4],
                    "subsample": [0.6],
                    "colsample_bynode": [0.6],
                },
                "cnntf_v2_gap": {
                    "lr": [0.001],
                    "batch_size": [16],
                    "variant": ["balanced_axis_log"],
                    "num_transformer_blocks": [2],
                    "num_heads": [4],
                    "ff_dim": [256],
                    "model_dim": [64],
                    "attention_key_dim": [16],
                    "dropout": [0.1],
                    "tokenization": ["time_axis"],
                    "input_transform": ["log_power"],
                    "log_scale": [1e-12],
                },
                "alexnet": {
                    "lr": [0.001],
                    "batch_size": [16],
                    "variant": ["legacy_log"],
                    "log_scale": [1e-12],
                },
            },
            "default_keras": {
                "fit_verbose": 1,
            },
        },
    },
    "ensemble": {
        # Select reusable strategies by name. Labels, weights, leakage rules,
        # holdout fraction, and combination mechanics live in utils/ensemble/.
        "enabled_strategy_names": [
            "simple_equal",
            "prediction_max",
            "inner_holdout",
            "val_fold_legacy",
        ],
        "primary_strategy_name": "val_fold_legacy",
    },
    "features": {
        "pca_components": 100,
    },
    "output": {
        "save_date": datetime.now().strftime("%Y%m%d"),
        "result_date_dir": datetime.now().strftime("%Y%m%d") + "_selected_log_architecture",
        "save_fold_predictions": True,
        "save_tuning_summary": True,
        "resume_completed_runs": True,
    },
    "explainability": {
        # Explanations are extra validation analyses for the trained fold model.
        # They are saved under each run folder:
        #   <SAVE_PATH>/explainability/fold{n}/{model_key}/
        #
        # Dataset conditions, model keys, folds, and map saving are inherited
        # from data/models/run/enabled so they have a single source of truth.
        "enabled": True,
        "max_samples_per_fold": 5,
        "ig_steps": 64,
        # Use methods that match each architecture.  RF TreeSHAP is retained in
        # PCA space for model auditing; physical RF claims use grouped masks.
        "methods_by_model": {
            "rf": [
                "tree_shap_pca",
                "group_occlusion"
            ],
            "cnntf_v2_gap": [
                "integrated_gradients",
                "group_occlusion"
            ],
            "alexnet": [
                "integrated_gradients",
                "grad_cam",
                "group_occlusion",
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
        "time_groups": 4,
        "time_extent_seconds": 1.0,
        "onb_band_frac": 0.10,
        "baseline_value": 0.0,
        "curve_fractions": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
        # Small non-negative perturbations test whether the primary IG maps are
        # locally stable.  This is separate from noise-condition robustness.
        "stability": {
            "enabled": True,
            "methods": ["integrated_gradients"],
            "repeats": 2,
            "noise_fraction": 0.01,
            "clip_nonnegative": True,
            "random_seed": 42,
        },
        # A lightweight Adebayo-style screening test.  Only the final trainable
        # layer is randomized, so report it as a partial sanity check rather
        # than a full cascading randomization test.
        "sanity_check": {
            "enabled": True,
            "methods": ["integrated_gradients"],
            "random_seed": 42,
        },
        # Never silently repeat a completed training run just because old XAI
        # files are missing.  Set True only after intentionally choosing that cost.
        "retrain_completed_runs_for_xai": False,
    },
}


def _cfg(section, key):
    return VALIDATION_CONFIG[section][key]


def _noise_source_prefix(noise_source):
    if noise_source in (0, "0", "whitenoise"):
        return "whitenoise"
    if noise_source in (1, "1", "waterflow"):
        return "waterflow"
    raise ValueError("noise_source must be 'waterflow' or 'whitenoise'.")


SMOKE_TEST = _cfg("run", "smoke_test")
EPOCH_NUM = _cfg("run", "smoke_epochs" if SMOKE_TEST else "epochs")
DIVISIONS = _cfg("run", "smoke_folds" if SMOKE_TEST else "folds")
COLOR_CHANNEL = _cfg("run", "color_channel")
RANDOM_SEED = _cfg("run", "random_seed")
FLG_ROOP = _cfg("run", "loop_parameter_sets")

NOISE_SOURCE_PREFIX = _noise_source_prefix(_cfg("data", "noise_source"))
CHUNK = _cfg("data", "chunk_seconds")
EXPERIMENT_DIR_NAMES = _cfg("data", "experiment_names")
MAX_FREQ_HZ_LIST = _cfg("data", "max_freq_hz_list")
NOISE_DIR_NAMES = _cfg("data", "noise_dir_names")
DATA_SOURCE_DIR_BY_EXPERIMENT = _cfg("data", "data_source_dir_by_experiment")
SKIP_MISSING_DATASETS = _cfg("data", "skip_missing_datasets")

THRESHOLD_BY_EXPERIMENT = _cfg("thresholds", "by_experiment")
REQUIRE_EXPERIMENT_THRESHOLD = _cfg("thresholds", "require_experiment_threshold")
ONB_BAND_FRAC = _cfg("thresholds", "onb_band_frac")

ACTIVE_MODEL_KEYS = _cfg("models", "active_model_keys")
PARAMETER_SETS = expand_parameter_sets(
    _cfg("models", "parameter_sets"),
    active_model_keys=ACTIVE_MODEL_KEYS,
)

ENSEMBLE_MANAGER = EnsembleManager(
    VALIDATION_CONFIG.get("ensemble", {}),
    ACTIVE_MODEL_KEYS,
    random_seed=RANDOM_SEED,
)
ENSEMBLE_ENABLED = ENSEMBLE_MANAGER.enabled and len(ACTIVE_MODEL_KEYS) >= 2
RESULT_MODEL_GROUP = (
    "ensemble" if ENSEMBLE_ENABLED
    else "rf" if ACTIVE_MODEL_KEYS == ["rf"]
    else "cnntf_v2_gap" if ACTIVE_MODEL_KEYS == ["cnntf_v2_gap"]
    else "alexnet" if ACTIVE_MODEL_KEYS == ["alexnet"]
    else "single_model"
)

PCA_COMPONENTS = _cfg("features", "pca_components")

SAVE_DATE = _cfg("output", "save_date")
RESULT_DATE_DIR = _cfg("output", "result_date_dir") or SAVE_DATE
SAVE_FOLD_PREDICTIONS = _cfg("output", "save_fold_predictions")
SAVE_TUNING_SUMMARY = _cfg("output", "save_tuning_summary")
RESUME_COMPLETED_RUNS = _cfg("output", "resume_completed_runs")
RUN_INSTANCE_ID = os.environ.get("RUN_ID", datetime.now().strftime("%H%M%S"))
FOLD_PREDICTIONS_DIR_NAME = "fold_pred"
EXPLAINABILITY_CONFIG = resolve_explainability_scope(
    VALIDATION_CONFIG.get("explainability", {}),
    experiment_names=EXPERIMENT_DIR_NAMES,
    max_freq_hz_list=MAX_FREQ_HZ_LIST,
    noise_dir_names=NOISE_DIR_NAMES,
    model_keys=ACTIVE_MODEL_KEYS,
    fold_count=DIVISIONS,
)
EXPLAINABILITY_ENABLED = EXPLAINABILITY_CONFIG.get("enabled", False)

# Settings are defined in VALIDATION_CONFIG above.
# The constants below are derived values used by the run loop; do not edit
# them directly unless you are changing the script mechanics.


# Model registry.
# Keep this block because it maps model keys to the actual builder functions.
# Fixed numeric parameters belong in VALIDATION_CONFIG["models"]["parameter_sets"].


MODEL_SPECS = [
    {
        "key": "rf",
        "label": "RandomForest",
        "kind": "sklearn",
        "builder": lambda mm, **params: mm.random_forest(**params),
    },
    {
        "key": "cnntf_v2_gap",
        "label": "CNN+Tf v2 GAP",
        "kind": "keras",
        "builder": lambda mm, **params: mm.cnn_transformer_v2(pooling="gap", **params),
        "input_axes_assumption": ["time_frame", "frequency_bin", "channel"],
        "architecture": {
            "front_end": "alexnet_like_cnn",
            "sequence_length_after_cnn": 7,
            "encoder": "transformer_encoder",
            "pooling": "GlobalAveragePooling1D",
        },
    },
    {
        "key": "alexnet",
        "label": "AlexNet",
        "kind": "keras",
        "builder": lambda mm, **params: mm.alexnet(**params),
    },
]




#### 繝・・繧ｿ繝輔か繝ｫ繝縺ｮ險ｭ螳・####
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT.joinpath(*_cfg("data", "experiment_root_parts"))

# matplotlib 縺ｮ險ｭ螳・
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#######################################################################
#                                螳溯｡碁Κ
#######################################################################

def build_dataset_jobs():
    return make_dataset_jobs(
        experiment_root=EXPERIMENT_ROOT,
        experiment_names=EXPERIMENT_DIR_NAMES,
        max_freq_hz_list=MAX_FREQ_HZ_LIST,
        noise_dir_names=NOISE_DIR_NAMES,
        data_source_dir_by_experiment=DATA_SOURCE_DIR_BY_EXPERIMENT,
        noise_source_prefix=NOISE_SOURCE_PREFIX,
        chunk_seconds=CHUNK,
        threshold_by_experiment=THRESHOLD_BY_EXPERIMENT,
        result_model_group=RESULT_MODEL_GROUP,
        result_date_dir=RESULT_DATE_DIR,
        color_channel=COLOR_CHANNEL,
        require_experiment_threshold=REQUIRE_EXPERIMENT_THRESHOLD,
        skip_missing_datasets=SKIP_MISSING_DATASETS,
    )


def validation_config_snapshot():
    return {
        "run": {
            "smoke_test": SMOKE_TEST,
            "epochs": EPOCH_NUM,
            "folds": DIVISIONS,
            "color_channel": COLOR_CHANNEL,
            "random_seed": RANDOM_SEED,
            "loop_parameter_sets": FLG_ROOP,
        },
        "data": {
            "experiment_root": str(EXPERIMENT_ROOT),
            "result_model_group": RESULT_MODEL_GROUP,
            "noise_source": NOISE_SOURCE_PREFIX,
            "chunk_seconds": CHUNK,
            "experiment_names": EXPERIMENT_DIR_NAMES,
            "max_freq_hz_list": MAX_FREQ_HZ_LIST,
            "noise_dir_names": NOISE_DIR_NAMES,
            "data_source_dir_by_experiment": DATA_SOURCE_DIR_BY_EXPERIMENT,
            "skip_missing_datasets": SKIP_MISSING_DATASETS,
        },
        "thresholds": {
            "by_experiment": THRESHOLD_BY_EXPERIMENT,
            "require_experiment_threshold": REQUIRE_EXPERIMENT_THRESHOLD,
            "onb_band_frac": ONB_BAND_FRAC,
        },
        "models": {
            "active_model_keys": ACTIVE_MODEL_KEYS,
            "parameter_sets": PARAMETER_SETS,
        },
        "ensemble": ENSEMBLE_MANAGER.snapshot(),
        "features": {
            "pca_components": PCA_COMPONENTS,
        },
        "output": {
            "save_date": SAVE_DATE,
            "result_date_dir": RESULT_DATE_DIR,
            "run_instance_id": RUN_INSTANCE_ID,
            "save_fold_predictions": SAVE_FOLD_PREDICTIONS,
            "save_tuning_summary": SAVE_TUNING_SUMMARY,
            "resume_completed_runs": RESUME_COMPLETED_RUNS,
        },
        "explainability": EXPLAINABILITY_CONFIG,
    }


def validation_config_text():
    return pformat(validation_config_snapshot(), sort_dicts=False)


def validate_validation_config(enabled_specs):
    if DIVISIONS < 2:
        raise ValueError("folds must be at least 2.")
    if not PARAMETER_SETS:
        raise ValueError("VALIDATION_CONFIG['models']['parameter_sets'] must not be empty.")
    if int(PCA_COMPONENTS) <= 0:
        raise ValueError("pca_components must be a positive integer.")

    model_keys = [spec["key"] for spec in enabled_specs]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError(f"Duplicate active model keys: {model_keys}")

    for parameter_set in PARAMETER_SETS:
        if not isinstance(parameter_set, dict):
            raise TypeError("Each expanded parameter set must be a dict.")
        # This also verifies that every active Keras model has lr/batch_size.
        # Parameters belonging to inactive models are intentionally ignored.
        resolve_parameter_set(enabled_specs, parameter_set)

    ENSEMBLE_MANAGER.validate(enabled_specs)

    if REQUIRE_EXPERIMENT_THRESHOLD:
        missing_thresholds = [
            name for name in EXPERIMENT_DIR_NAMES
            if THRESHOLD_BY_EXPERIMENT.get(name) is None
        ]
        if missing_thresholds:
            raise ValueError(
                "Missing ONB threshold for experiments: "
                f"{missing_thresholds}. Add them to VALIDATION_CONFIG['thresholds']['by_experiment']."
            )

    if EXPLAINABILITY_ENABLED:
        requested_models = set(EXPLAINABILITY_CONFIG.get("model_keys") or model_keys)
        unknown_xai_models = requested_models - set(model_keys)
        if unknown_xai_models:
            raise ValueError(
                "Explainability model_keys must be active models, got: "
                f"{sorted(unknown_xai_models)}")

        target_folds = EXPLAINABILITY_CONFIG.get("target_folds") or []
        invalid_folds = [
            int(fold) for fold in target_folds
            if not 1 <= int(fold) <= int(DIVISIONS)
        ]
        if invalid_folds:
            raise ValueError(
                f"Explainability target_folds must be within 1..{DIVISIONS}: "
                f"{invalid_folds}")
        if int(EXPLAINABILITY_CONFIG.get("max_samples_per_fold", 0)) <= 0:
            raise ValueError("Explainability max_samples_per_fold must be positive.")
        if int(EXPLAINABILITY_CONFIG.get("ig_steps", 0)) <= 0:
            raise ValueError("Explainability ig_steps must be positive.")

        fractions = [
            float(value)
            for value in EXPLAINABILITY_CONFIG.get("curve_fractions", [])
        ]
        if fractions and (
                fractions != sorted(set(fractions))
                or fractions[0] != 0.0
                or fractions[-1] != 1.0):
            raise ValueError(
                "Explainability curve_fractions must be unique, sorted, and "
                "include endpoints 0.0 and 1.0.")

        if not np.isclose(
                float(EXPLAINABILITY_CONFIG.get("time_extent_seconds", CHUNK)),
                float(CHUNK)):
            raise ValueError(
                "Explainability time_extent_seconds must match data chunk_seconds.")
        if not np.isclose(
                float(EXPLAINABILITY_CONFIG.get("onb_band_frac", ONB_BAND_FRAC)),
                float(ONB_BAND_FRAC)):
            raise ValueError(
                "Explainability onb_band_frac must match thresholds.onb_band_frac.")

        known_methods = {
            "tree_shap_pca", "treeshap", "integrated_gradients",
            "grad_cam", "group_occlusion", "occlusion",
        }
        methods_by_model = EXPLAINABILITY_CONFIG.get("methods_by_model") or {}
        known_model_keys = {spec["key"] for spec in MODEL_SPECS}
        unknown_method_models = set(methods_by_model) - known_model_keys
        if unknown_method_models:
            raise ValueError(
                "Explainability methods_by_model contains unknown models: "
                f"{sorted(unknown_method_models)}")
        missing_method_models = requested_models - set(methods_by_model)
        if missing_method_models:
            raise ValueError(
                "Explainability methods_by_model must explicitly define every "
                f"requested model: {sorted(missing_method_models)}")
        unknown_methods = {
            method
            for methods_for_model in methods_by_model.values()
            for method in methods_for_model
            if method not in known_methods
        }
        if unknown_methods:
            raise ValueError(
                f"Unknown explainability methods: {sorted(unknown_methods)}")

        condition_filter = EXPLAINABILITY_CONFIG.get("condition_filter") or {}
        available_by_filter = {
            "experiment_names": set(EXPERIMENT_DIR_NAMES),
            "max_freq_hz_list": set(MAX_FREQ_HZ_LIST),
            "noise_dir_names": set(NOISE_DIR_NAMES),
        }
        for filter_key, available in available_by_filter.items():
            requested_values = set(condition_filter.get(filter_key) or [])
            selected_values = requested_values & available if requested_values else available
            if not selected_values:
                raise ValueError(
                    f"Explainability condition_filter.{filter_key} does not match "
                    f"any configured data value. requested={sorted(requested_values)}, "
                    f"available={sorted(available)}")

        requested_experiments = set(condition_filter.get("experiment_names") or [])
        selected_experiments = (
            requested_experiments & set(EXPERIMENT_DIR_NAMES)
            if requested_experiments else set(EXPERIMENT_DIR_NAMES)
        )
        missing_xai_thresholds = [
            name for name in selected_experiments
            if THRESHOLD_BY_EXPERIMENT.get(name) is None
        ]
        if missing_xai_thresholds:
            raise ValueError(
                "Explainability requires ONB thresholds for every selected "
                f"experiment: {sorted(missing_xai_thresholds)}")

def main():
    set_global_seed(RANDOM_SEED)
    # Shared helpers for metrics, training, weighting, and plots.
    metrics = RegressionDetectionMetrics()
    trainer = ModelTrainer(random_seed=RANDOM_SEED)
    plotter = RegressionPlotter()

    # Resolve the model keys selected in VALIDATION_CONFIG.
    spec_by_key = {s["key"]: s for s in MODEL_SPECS}
    unknown = [k for k in ACTIVE_MODEL_KEYS if k not in spec_by_key]
    if unknown:
        raise ValueError(f"ACTIVE_MODEL_KEYS has unknown keys: {unknown} "
                         f"(defined: {list(spec_by_key)})")
    enabled_specs = [spec_by_key[k] for k in ACTIVE_MODEL_KEYS]
    if not enabled_specs:
        raise ValueError("ACTIVE_MODEL_KEYS must select at least one model.")

    # Keep output folder names short enough for Windows paths.
    validate_validation_config(enabled_specs)

    configured_model_keys = [s["key"] for s in enabled_specs]
    configured_model_tag = "-".join(configured_model_keys)
    single_model_run = len(configured_model_keys) == 1

    print("#" * 60)
    if SMOKE_TEST:
        print("### SMOKE_TEST = True ###")
        print(f"###   epoch={EPOCH_NUM} / fold={DIVISIONS}")
        print("###   Use only for quick checks; set SMOKE_TEST=False for real runs.")
    else:
        print("### FULL_RUN mode (SMOKE_TEST = False) ###")
    print(
        f"configured models: {[s['label'] for s in enabled_specs]}  "
        f"(model_tag={configured_model_tag})"
    )
    parameter_mode = (
        "fixed parameters (one expanded set)"
        if len(PARAMETER_SETS) == 1
        else f"grid tuning ({len(PARAMETER_SETS)} expanded sets)"
    )
    print(f"parameter mode: {parameter_mode}")
    if single_model_run:
        print(
            "execution mode: single active model; ensemble strategies are skipped "
            f"| epoch={EPOCH_NUM} | fold={DIVISIONS}"
        )
    else:
        print(
            f"ensemble: {ENSEMBLE_MANAGER.description()} "
            f"| epoch={EPOCH_NUM} | fold={DIVISIONS}"
        )
    print(
        "explainability: "
        f"{'enabled' if EXPLAINABILITY_ENABLED else 'disabled'} | "
        f"target_folds={EXPLAINABILITY_CONFIG.get('target_folds')} | "
        f"max_samples={EXPLAINABILITY_CONFIG.get('max_samples_per_fold')} | "
        f"condition_filter={EXPLAINABILITY_CONFIG.get('condition_filter')}"
    )
    print("validation_config:")
    print(validation_config_text())
    if not single_model_run and ENSEMBLE_MANAGER.has_leaky_strategy:
        print("WARNING: val_fold_legacy uses validation-fold labels for weights; use only for reproduction.")
    print("#" * 60)

    dataset_jobs = build_dataset_jobs()
    if not dataset_jobs:
        raise FileNotFoundError("No datasets were found for the requested experiment/maxfreq/noise plan.")

    for job_i, job in enumerate(dataset_jobs, start=1):
        K.clear_session()
        gc.collect()

        data_path = job["data_path"]
        snr_value = job["snr_value"]
        noise_dir_name = job["noise_dir_name"]
        max_freq_name = job["max_freq_hz"]
        base_save_path = job["save_base_path"]
        threshold = job["threshold"]
        xai_for_job = bool(
            EXPLAINABILITY_ENABLED
            and explainability_condition_selected(
                EXPLAINABILITY_CONFIG,
                job["experiment_name"],
                max_freq_name,
                noise_dir_name,
            )
        )

        print(
            f"\n{'='*40}\n"
            f"dataset {job_i}/{len(dataset_jobs)} | "
            f"{job['experiment_name']} | {max_freq_name} | {noise_dir_name}"
        )
        print(f"data_path={data_path}")
        print(f"threshold={threshold}")
        print(f"explainability_selected={xai_for_job}")

        start_time = time.time()
        data_loading = DataLoadingConversion()
        sample_groups = None
        if COLOR_CHANNEL == 1:
            x, y, sample_metadata = data_loading.load_npy_data(
                data_path, return_metadata=True
            )
            sample_groups = np.asarray(
                [row.get("source_wav_id", "") for row in sample_metadata],
                dtype=str,
            )
            if np.any(sample_groups == ""):
                raise RuntimeError(
                    "source_wav_id is missing. The corrected dataset requires "
                    "chunk_manifest.csv so validation can use GroupKFold."
                )
            if len(np.unique(sample_groups)) < DIVISIONS:
                raise ValueError(
                    "GroupKFold requires at least "
                    f"{DIVISIONS} source WAVs, got "
                    f"{len(np.unique(sample_groups))}."
                )
        else:
            x, y = data_loading.load_image_data(data_path)
        print(f"x shape: {x.shape} | y shape: {y.shape} | "
              f"load_time={time.time() - start_time:.2f}s")

        for parameter_set in PARAMETER_SETS:
                run_specs = resolve_parameter_set(enabled_specs, parameter_set)
                model_keys = [spec["key"] for spec in run_specs]
                model_tag = "-".join(model_keys)
                if SMOKE_TEST:
                    model_tag = "s_" + model_tag
                use_sklearn = any(spec["kind"] == "sklearn" for spec in run_specs)
                ensemble_run = ENSEMBLE_MANAGER.create_run(run_specs)
                include_ensemble = ensemble_run.enabled
                all_keys = list(model_keys)
                all_keys.extend(ensemble_run.result_keys)
                label_of = {spec["key"]: spec["label"] for spec in run_specs}
                label_of.update(ensemble_run.labels)
                param_tag = parameter_set_tag(parameter_set, run_specs, safe_tag)
                param_summary = model_param_summary(run_specs)
                run_hash = run_config_digest(
                    validation_config_snapshot(), parameter_set, run_specs,
                    model_tag, SAVE_FOLD_PREDICTIONS)
                run_dir = run_dir_name(
                    EPOCH_NUM, param_tag, model_tag,
                    include_ensemble,
                    ensemble_run.strategy_tag,
                )
                print(f"parameter_set={parameter_set.get('name', param_tag)} | model_params={param_summary}")
                print(f"run_dir={run_dir}")
                SAVE_PATH = os.path.join(
                    base_save_path, noise_dir_name, max_freq_name,
                    run_dir)
                tuning_summary_path = os.path.join(base_save_path, "tuning_summary.csv")
                append_tuning_summary_this_run = SAVE_TUNING_SUMMARY
                completed_run = is_completed_run(
                    tuning_summary_path, run_dir, SAVE_PATH, snr_value,
                    RESUME_COMPLETED_RUNS, SAVE_TUNING_SUMMARY,
                    run_hash=run_hash)
                if completed_run:
                    xai_complete = explainability_outputs_complete(
                        SAVE_PATH, EXPLAINABILITY_CONFIG, model_keys, DIVISIONS,
                        experiment_name=job["experiment_name"],
                        max_freq_name=max_freq_name,
                        noise_dir_name=noise_dir_name)
                    if xai_complete:
                        print(f"[resume skip] completed run found: {run_dir}")
                        continue
                    if not EXPLAINABILITY_CONFIG.get(
                            "retrain_completed_runs_for_xai", False):
                        print(
                            "[resume skip] metrics are complete but selected XAI "
                            "outputs are missing. No retraining was started; set "
                            "explainability.retrain_completed_runs_for_xai=True "
                            "only if the extra training cost is intentional."
                        )
                        continue
                    print(
                        "[resume xai] metrics are complete, but explainability "
                        "outputs are missing; explicit retraining is enabled."
                    )
                    append_tuning_summary_this_run = False

                _makedirs(SAVE_PATH)
                write_run_manifest(
                    SAVE_PATH, job, parameter_set, run_specs,
                    param_tag, model_tag, run_hash, run_dir,
                    RUN_INSTANCE_ID, validation_config_snapshot())

                if sample_groups is None:
                    kf = KFold(
                        n_splits=DIVISIONS,
                        shuffle=True,
                        random_state=RANDOM_SEED,
                    )
                    split_indices = kf.split(x)
                    split_description = "KFold(sample-level fallback)"
                else:
                    kf = GroupKFold(n_splits=DIVISIONS)
                    split_indices = kf.split(x, y, groups=sample_groups)
                    split_description = "GroupKFold(source_wav_id)"

                # 謖・ｨ吶・菫晏ｭ伜・ (key -> metric -> [fold 縺斐→縺ｮ蛟､])
                store = {k: defaultdict(list) for k in all_keys}
                train_meta = {k: defaultdict(list) for k in model_keys}

                output_file = os.path.join(SAVE_PATH, f'validation_results_{snr_value}.txt')
                with _open_text(output_file, 'w', encoding='utf-8') as f:
                    f.write("K-fold Cross-Validation Results\n")
                    f.write("validation_config:\n")
                    f.write(validation_config_text() + "\n")
                    f.write(f"experiment={job['experiment_name']}\n")
                    f.write(f"data_source={job['source_dir']}\n")
                    f.write(f"data_path={data_path}\n")
                    f.write(f"max_freq={max_freq_name}\n")
                    f.write(f"noise_dir={noise_dir_name}\n")
                    f.write(f"threshold={threshold}\n")
                    f.write(f"result_date_dir={RESULT_DATE_DIR}\n")
                    f.write(f"models={[s['label'] for s in run_specs]}\n")
                    f.write(f"parameter_set={parameter_set.get('name', param_tag)}\n")
                    f.write(f"run_dir={run_dir}\n")
                    f.write(f"run_hash={run_hash}\n")
                    f.write(f"run_instance_id={RUN_INSTANCE_ID}\n")
                    f.write(f"validation_split={split_description}\n")
                    f.write(f"model_params={param_summary}\n")
                    f.write(ensemble_run.description() + "\n")
                    f.write("=" * 30 + "\n")

                    fold = 1
                    for train_index, val_index in split_indices:
                        x_train, x_val = x[train_index], x[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        inner_errors = ensemble_run.fit_inner_holdout_errors(
                            trainer=trainer,
                            x_train=x_train,
                            y_train=y_train,
                            pca_components=PCA_COMPONENTS,
                            input_shape=(224, 224, COLOR_CHANNEL),
                            epochs=EPOCH_NUM,
                            fold=fold,
                            total_folds=DIVISIONS,
                        )

                        # Final outer-fold model preprocessing uses all training
                        # samples and never sees the outer validation labels.
                        scaler = MinMaxScaler()
                        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

                        # sklearn 邉ｻ繝｢繝・Ν逕ｨ縺ｮ PCA (蟄ｦ鄙偵ョ繝ｼ繧ｿ縺ｮ縺ｿ縺ｧ fit)
                        pca_model = None
                        if use_sklearn:
                            if xai_for_job:
                                x_train_pca, (x_val_pca,), pca_model = trainer.make_pca(
                                    x_train, [x_val], PCA_COMPONENTS, return_pca=True)
                            else:
                                x_train_pca, (x_val_pca,) = trainer.make_pca(
                                    x_train, [x_val], PCA_COMPONENTS)
                        else:
                            x_train_pca = x_val_pca = None

                        mm = RegressionModelMaker((224, 224, COLOR_CHANNEL))

                        # --- 蜷・Δ繝・Ν縺ｮ蟄ｦ鄙偵→莠域ｸｬ ---
                        val_preds = {}        # key -> 讀懆ｨｼ fold 縺ｸ縺ｮ莠域ｸｬ (蜈・せ繧ｱ繝ｼ繝ｫ)
                        for spec in run_specs:
                            # Reset before every model so architecture comparisons
                            # do not depend on parameter-set or model execution order.
                            set_global_seed(RANDOM_SEED + fold)
                            if spec["kind"] == "keras":
                                print(f"  params for {spec['key']}: lr={spec['lr']}, batch_size={spec['batch_size']}")
                            print(f"[{spec['label']}] Fold {fold}/{DIVISIONS} training start")
                            model, history = trainer.train_one_model(
                                spec, mm, x_train, y_train_scaled, x_train_pca,
                                EPOCH_NUM)
                            if history is not None:
                                actual_batch_size = history.params.get("actual_batch_size")
                                requested_batch_size = history.params.get("requested_batch_size")
                                stopped_by_memory_error = history.params.get("stopped_by_memory_error")
                                epochs_completed = history.params.get("epochs_completed")
                                train_meta[spec["key"]]["actual_batch_size"].append(actual_batch_size)
                                train_meta[spec["key"]]["requested_batch_size"].append(requested_batch_size)
                                train_meta[spec["key"]]["stopped_by_memory_error"].append(bool(stopped_by_memory_error))
                                train_meta[spec["key"]]["epochs_completed"].append(epochs_completed)
                                if actual_batch_size and requested_batch_size and actual_batch_size != requested_batch_size:
                                    msg = (
                                        f"  [OOM retry used] {spec['key']}: "
                                        f"batch_size {requested_batch_size} -> {actual_batch_size}"
                                    )
                                    print(msg)
                                    f.write(msg + "\n")
                                if stopped_by_memory_error:
                                    msg = (
                                        f"  [OOM accepted] {spec['key']}: "
                                        f"epochs_completed={epochs_completed}, current weights used"
                                    )
                                    print(msg)
                                    f.write(msg + "\n")
                            plotter.plot_loss_history(history, EPOCH_NUM, spec["label"],
                                                      fold, SAVE_PATH, snr_value)

                            # 讀懆ｨｼ fold 縺ｸ縺ｮ莠域ｸｬ
                            val_preds[spec["key"]] = trainer.predict_one_model(
                                spec, model, x_val, x_val_pca, scaler)

                            maybe_explain_trained_model(
                                spec, model, scaler, x_val, y_val,
                                val_preds[spec["key"]], threshold, SAVE_PATH,
                                fold, max_freq_name, EXPLAINABILITY_CONFIG,
                                pca=pca_model,
                                experiment_name=job["experiment_name"],
                                noise_dir_name=noise_dir_name)

                            ensemble_run.record_validation_error(
                                spec["key"],
                                y_val,
                                val_preds[spec["key"]],
                            )

                            del model
                            del history
                            K.clear_session()
                            gc.collect()

                        # Evaluate every configured ensemble from the same
                        # outer-fold predictions. No model is retrained between
                        # ensemble strategy profiles.
                        ensemble_outputs = ensemble_run.combine_predictions(
                            val_preds,
                            inner_errors,
                            fold,
                        )

                        # --- 謖・ｨ吶・邂怜・ (竭｡ 3 遞ｮ鬘槭↓蛻・屬) ---
                        preds_all = ensemble_run.merge_predictions(
                            val_preds,
                            ensemble_outputs,
                        )
                        if SAVE_FOLD_PREDICTIONS:
                            pred_dir = os.path.join(SAVE_PATH, FOLD_PREDICTIONS_DIR_NAME)
                            _makedirs(pred_dir)
                            pred_csv = os.path.join(pred_dir, f"pred_f{fold}_{snr_value}.csv")
                            with _open_text(pred_csv, "w", newline="", encoding="utf-8") as pf:
                                writer = csv.writer(pf)
                                writer.writerow(["sample_index", "y_true"] + all_keys)
                                for row_i, sample_idx in enumerate(val_index):
                                    writer.writerow(
                                        [int(sample_idx), f"{float(y_val[row_i]):.10g}"]
                                        + [f"{float(preds_all[key][row_i]):.10g}" for key in all_keys]
                                    )
                        for key in all_keys:
                            pred = preds_all[key]
                            reg = metrics.regression_metrics(y_val, pred, threshold, ONB_BAND_FRAC)
                            det_c = metrics.detection_metrics_continuous(y_val, pred, threshold)
                            det_b = metrics.detection_metrics_binary(y_val, pred, threshold)
                            for d in (reg, det_c, det_b):
                                for mk, mv in d.items():
                                    store[key][mk].append(mv)

                        ensemble_run.record_diagnostics(
                            fold,
                            y_val,
                            val_preds,
                            ensemble_outputs,
                            threshold,
                            ONB_BAND_FRAC,
                        )

                        # --- 菴懷峙 (繧｢繝ｳ繧ｵ繝ｳ繝悶Ν縺ｮ謨｣蟶・峙) ---
                        if include_ensemble and has_threshold(threshold):
                            primary_pred = ensemble_outputs[
                                ensemble_run.primary_result_key
                            ]["prediction"]
                            ens_fold_metrics = {
                                mk: store[ensemble_run.primary_result_key][mk][-1]
                                for mk in store[ensemble_run.primary_result_key]
                            }
                            plotter.plot_regression_scatter(
                                y_val, primary_pred, y, ens_fold_metrics,
                                threshold, SAVE_PATH, snr_value, fold)

                        # --- fold 邨先棡繧・txt 縺ｫ霑ｽ險・---
                        f.write(f"Recorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                        f.write(f"Fold {fold} Results\n")
                        ensemble_run.write_fold_weights(f, ensemble_outputs)
                        for key in all_keys:
                            m = {mk: store[key][mk][-1] for mk in store[key]}
                            f.write(
                                f"  [{label_of[key]}] "
                                f"R2={m.get('r2', float('nan')):.4f} "
                                f"RMSE={m.get('rmse_all', float('nan')):.1f} "
                                f"MAE={m.get('mae_all', float('nan')):.1f} | "
                                f"R2_high={m.get('r2_high', float('nan')):.4f} "
                                f"RMSE_onb={m.get('rmse_onb', float('nan')):.1f} "
                                f"(n_onb={m.get('n_onb', 0)}) | "
                                f"AUC_bin={m.get('auc_binary', float('nan')):.4f} "
                                f"ROC_cont={m.get('roc_auc_cont', float('nan')):.4f} "
                                f"PR_cont={m.get('pr_auc_cont', float('nan')):.4f} | "
                                f"Acc={m.get('accuracy', float('nan')):.4f} "
                                f"Prec={m.get('precision', float('nan')):.4f} "
                                f"Rec={m.get('recall', float('nan')):.4f} "
                                f"F1={m.get('f1', float('nan')):.4f}\n")
                        f.write("-" * 30 + "\n")

                        fold += 1
                        del x_train, x_val, y_train, y_val
                        del pca_model
                        del x_train_pca, x_val_pca, y_train_scaled
                        del val_preds, preds_all, ensemble_outputs
                        K.clear_session()
                        gc.collect()

                    # --- 蟷ｳ蝮・ｵ先棡 (mean ﾂｱ SE) ---
                    f.write(f"\nRecorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    f.write("Average Results (mean ﾂｱ SE):\n")
                    summary_metrics = [
                        "r2", "rmse_all", "mae_all", "r2_high", "rmse_high", "mae_high",
                        "rmse_onb", "mae_onb",
                        "auc_binary", "roc_auc_cont", "pr_auc_cont",
                        "accuracy", "precision", "recall", "f1",
                    ]
                    for key in all_keys:
                        f.write(f"  [{label_of[key]}]\n")
                        for mk in summary_metrics:
                            mean, se = metrics.mean_se(store[key][mk])
                            f.write(f"    {mk:14s}: {mean:.4f} ﾂｱ {se:.4f}\n")
                    f.write("=" * 30 + "\n\n")

                if xai_for_job:
                    aggregated_xai = aggregate_group_mask_comparison(
                        SAVE_PATH,
                        EXPLAINABILITY_CONFIG,
                        model_keys,
                        DIVISIONS,
                    )
                    print(
                        "XAI model comparison: "
                        f"{'saved' if aggregated_xai else 'no group-mask rows found'}"
                    )

                # --- 譽偵げ繝ｩ繝・(繝｢繝・Ν蛻･: R2 縺ｨ譌ｧ繧ｳ繝ｼ繝我ｺ呈鋤縺ｮ莠悟､蛹門ｾ・AUC) ---
                # Keep the conventional bar plot readable: individual models
                # plus the predeclared primary ensemble only. All strategies
                # are shown in the signed-delta comparison below.
                plot_keys = list(model_keys)
                if include_ensemble:
                    plot_keys.append(ensemble_run.primary_result_key)
                labels = [label_of[k] for k in plot_keys]
                r2_means = [metrics.mean_se(store[k]["r2"])[0] for k in plot_keys]
                r2_ses = [metrics.mean_se(store[k]["r2"])[1] for k in plot_keys]
                auc_bin_means = [metrics.mean_se(store[k]["auc_binary"])[0] for k in plot_keys]
                auc_bin_ses = [metrics.mean_se(store[k]["auc_binary"])[1] for k in plot_keys]
                plotter.plot_bar("R2 Score", labels, r2_means, r2_ses, EPOCH_NUM, SAVE_PATH, snr_value)
                if has_threshold(threshold):
                    plotter.plot_bar("AUC (binary legacy)", labels, auc_bin_means, auc_bin_ses,
                                     EPOCH_NUM, SAVE_PATH, snr_value)

                # --- 謖・ｨ・CSV (fold 蟷ｳ蝮・ｒ繝｢繝・Ν蛻･縺ｫ菫晏ｭ倥ょｾ後〒豈碑ｼ・＠繧・☆縺上☆繧・ ---
                csv_path = os.path.join(SAVE_PATH, f'metrics_summary_{snr_value}.csv')
                with _open_text(csv_path, 'w', encoding='utf-8') as cf:
                    header = ["model"] + [f"{mk}_mean" for mk in summary_metrics] \
                                       + [f"{mk}_se" for mk in summary_metrics]
                    cf.write(",".join(header) + "\n")
                    for key in all_keys:
                        means = [f"{metrics.mean_se(store[key][mk])[0]:.6f}" for mk in summary_metrics]
                        ses = [f"{metrics.mean_se(store[key][mk])[1]:.6f}" for mk in summary_metrics]
                        cf.write(",".join([label_of[key]] + means + ses) + "\n")
                print(f"謖・ｨ・CSV 繧剃ｿ晏ｭ・ {csv_path}")

                ensemble_run.save_reports(
                    save_path=SAVE_PATH,
                    base_save_path=base_save_path,
                    snr_value=snr_value,
                    store=store,
                    metrics=metrics,
                    summary_metrics=summary_metrics,
                    plotter=plotter,
                    run_instance_id=RUN_INSTANCE_ID,
                    run_hash=run_hash,
                    run_dir=run_dir,
                    job=job,
                )

                append_tuning_summary(
                    tuning_summary_path, job, parameter_set, run_specs,
                    store, train_meta, summary_metrics, metrics,
                    param_tag, run_dir, run_hash, SAVE_PATH, all_keys,
                    append_tuning_summary_this_run, RUN_INSTANCE_ID)
                if append_tuning_summary_this_run:
                    print(f"tuning summary saved: {tuning_summary_path}")
                else:
                    print("tuning summary append skipped to avoid duplicate rows.")

                if not FLG_ROOP:
                    break

        del x, y
        gc.collect()


if __name__ == '__main__':
    main()
