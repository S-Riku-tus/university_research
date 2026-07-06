"""
run_ensemble_regression_onb.py

中心スクリプト 3.run_ensemble_ROC_100%_analysis.py の作り直し版。
2026-06-12 の研究計画 (研究進捗報告/2026/612/今後の研究計画_2026-06-12.md) の
Phase 0「評価軸とコードの整理」に対応する。

旧スクリプトに対して、次の3点を直したうえで作り直している。

  ① モデル名の取り違えをなくす
     旧コードは RandomForest の予測を alexnet_pred という変数に入れ、以降の
     R2/AUC/グラフ/txt すべてに "AlexNet" というラベルで記録していた。
     本スクリプトでは MODEL_SPECS というレジストリで各モデルを定義し、
     ラベルが必ず実体のモデルに追従するようにした。モデルの差し替え・
     有効/無効化は MODEL_SPECS を編集するだけで済む。

  ② AUC を「連続スコア版」と「二値化後の分類指標」に分ける
     旧コードは予測を閾値で 0/1 化してから ROC を計算していたため、ROC が
     2点しか持たず AUC が実質バランス精度になっていた。本スクリプトでは
       - 連続スコア版 AUC: 予測熱流束をそのままスコアにした ROC-AUC / PR-AUC
       - 二値化後の分類指標: Accuracy / Precision / Recall / F1
     を分けて算出する。旧来の二値化 AUC も後方比較用に残してある。

  ③ アンサンブル重みの決め方を選択式にする (リーク対策)
     旧コードは評価対象である検証 fold (y_val) の誤差から重みを決めており、
     データリークになっていた。本スクリプトでは WEIGHT_STRATEGY で
       - simple          : 単純平均 (重みなし)
       - fixed           : 固定重み (FIXED_WEIGHTS で指定)
       - inner_holdout   : 学習 fold 内 holdout の誤差から重み (リークなし)
       - val_fold_legacy : 旧来どおり検証 fold 誤差から重み (リークあり/再現用)
     を切り替えられる。

  ④ データパスは別マシン運用のためそのまま (旧コードと同じハードコード)

旧スクリプト (code/3.run_ensemble_ROC_100%_analysis.py) は再現性のため残す。

再利用可能な処理 (指標計算・学習/予測・重み付け・作図) は、既存の utils 方針に
合わせて用途別のクラスに分離してある。本ファイルにはこの実験固有の設定と
main() のオーケストレーションだけを置く。
    - 指標計算    : utils/calculation/regression_detection_metrics.py
    - 学習/予測   : utils/training/model_training.py
    - 重み付け    : utils/ensemble/ensemble_weighting.py
    - 作図        : utils/plotting/regression_plots.py
"""

import os
import gc
import time
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

# Avoid grabbing most of the GPU memory before the first model fit. This also
# makes OOM recovery by smaller batch sizes more reliable on Windows/TensorFlow.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras import backend as K

from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.training.model_training import ModelTrainer
from utils.ensemble.ensemble_weighting import EnsembleWeighting
from utils.plotting.regression_plots import RegressionPlotter
from utils.config.parameter_sets import expand_parameter_sets
from utils.explainability.training_integration import maybe_explain_trained_model


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


#######################################################################
#                              変数の指定
#######################################################################
# Validation controls: edit this block first.
# Edit this block first when changing an experiment.
#
# Current default:
# - data: 3 experiments x 6 max-frequency settings x 7 noise conditions
# - models: RF + CNN/Transformer v2 GAP + AlexNet
# - ensemble: controlled by VALIDATION_CONFIG["ensemble"] below
# - explainability: integrated into training, but disabled for full-grid runs
#
# Ensemble modes are configured in VALIDATION_CONFIG["ensemble"]:
# - "simple": equal-weight mean. No leakage, useful as a neutral baseline.
# - "fixed": user-defined weights in fixed_weights. No leakage; recommended
#   for final-report comparisons.
# - "inner_holdout": uses a holdout split inside the training fold to choose
#   weights. No validation-fold leakage, but slower and less stable.
# - "val_fold_legacy": chooses weights from the validation fold. Reproduction
#   only; do not use for final claims because it leaks evaluation labels.

VALIDATION_CONFIG = {
    "run": {
        "smoke_test": False,
        "epochs": 300,
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
            "2025.07.09_0.3_1",
            "2025.06.11_0.3_2",
        ],
        "max_freq_hz_list": [
            "maxfreq=2kHz",
            "maxfreq=3kHz",
            "maxfreq=5kHz",
            "maxfreq=10kHz",
            "maxfreq=15kHz",
            "maxfreq=22kHz",
        ],
        "noise_dir_names": [
            "heatflux_no_noise",
            "heatflux_SNR=0",
            "heatflux_SNR=-4",
            "heatflux_SNR=-8",
            "heatflux_SNR=-12",
            "heatflux_SNR=-16",
            "heatflux_SNR=-20",
        ],
        "data_source_dir_by_experiment": {
            "2025.06.11_0.3_2": "waterflow_20260629_1s",
            "2025.06.18_0.3_3": "waterflow_20260622_1s",
            "2025.07.09_0.3_1": "waterflow_20251219_1s",
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
        "active_model_keys": ["rf", "cnntf_v2_gap", "alexnet"],
        "parameter_sets": [
            {
                "name": "rf_v2_alex",
                "models": {
                    "rf": {
                        "n_estimators": 300,
                        "max_depth": 8,
                        "subsample": 0.8,
                        "colsample_bynode": 0.6,
                    },
                    "cnntf_v2_gap": {"lr": 0.0005, "batch_size": 32},
                    "alexnet": {"lr": 0.005, "batch_size": 32},
                },
                "default_keras": {
                    "fit_verbose": 1,
                },
            },
        ],
    },
    "ensemble": {
        # How to combine model predictions.
        #
        # Recommended final-report setting:
        #   enabled=True, weight_strategy="fixed", combine="mean"
        #
        # Other valid weight_strategy values:
        #   "simple"          equal weights across active models
        #   "inner_holdout"   choose weights using only training-fold holdout
        #   "val_fold_legacy" reproduce old validation-fold weighting; leaks
        #                     validation labels and should not support claims
        "enabled": True,
        "weight_strategy": "val_fold_legacy",
        # Used only when weight_strategy == "fixed".
        "fixed_weights": {
            "rf": 0.90,
            "cnntf_v2_gap": 0.05,
            "alexnet": 0.05,
        },
        # "mean" is a weighted average. "min" keeps the minimum model prediction
        # for each sample and is mainly a legacy/diagnostic option.
        "combine": "mean",
    },
    "features": {
        "pca_components": 100,
    },
    "output": {
        "save_date": datetime.now().strftime("%Y%m%d"),
        "result_date_dir": datetime.now().strftime("%Y%m%d") + "_fixed_ensemble_full_grid",
        "save_fold_predictions": True,
        "save_tuning_summary": True,
        "resume_completed_runs": True,
    },
    "explainability": {
        # Set True when you want explanations to be saved during training.
        # For the full grid, start with target_folds=[1] and a small sample count.
        "enabled": False,
        "model_keys": ["rf", "cnntf_v2_gap", "alexnet"],
        "target_folds": [1],
        "max_samples_per_fold": 3,
        "ig_steps": 32,
        "methods": ["integrated_gradients", "grad_cam", "occlusion"],
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

PARAMETER_SETS = expand_parameter_sets(_cfg("models", "parameter_sets"))
ACTIVE_MODEL_KEYS = _cfg("models", "active_model_keys")

ENSEMBLE_CONFIG = VALIDATION_CONFIG.get("ensemble", {})
ENSEMBLE_ENABLED = ENSEMBLE_CONFIG.get("enabled", False)
WEIGHT_STRATEGY = ENSEMBLE_CONFIG.get("weight_strategy", "simple")
FIXED_WEIGHTS = ENSEMBLE_CONFIG.get(
    "fixed_weights", {"rf": 0.90, "cnntf_v2_gap": 0.05, "alexnet": 0.05}
)
INNER_HOLDOUT_FRAC = ENSEMBLE_CONFIG.get("inner_holdout_frac", 0.2)
ENSEMBLE_COMBINE = ENSEMBLE_CONFIG.get("combine", "mean")
RESULT_MODEL_GROUP = (
    "ensemble" if ENSEMBLE_ENABLED
    else "rf" if ACTIVE_MODEL_KEYS == ["rf"]
    else "cnntf_v2_gap" if ACTIVE_MODEL_KEYS == ["cnntf_v2_gap"]
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
EXPLAINABILITY_CONFIG = VALIDATION_CONFIG.get("explainability", {})
EXPLAINABILITY_ENABLED = EXPLAINABILITY_CONFIG.get("enabled", False)

WEIGHT_STRATEGY_TAGS = {
    "simple": "simp",
    "fixed": "fix",
    "inner_holdout": "ih",
    "val_fold_legacy": "vleg",
}

KERAS_TRAINING_PARAM_KEYS = {
    "lr",
    "batch_size",
    "fit_verbose",
    "min_batch_size",
    "accept_partial_min_epochs",
}


def format_param_value(value):
    if isinstance(value, float):
        text = f"{value:.8f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


# Settings are defined in VALIDATION_CONFIG above.
# The constants below are derived values used by the run loop; do not edit
# them directly unless you are changing the script mechanics.


# ===================================================================
# ① モデルレジストリ
#    各モデルを 1 つの辞書で定義する。label が必ず実体に追従するので、
#    旧コードのような「RF の予測を alexnet と呼ぶ」取り違えが起きない。
#    モデルの追加・差し替え・無効化は、このリストを編集するだけでよい。
#
#    kind:
#       "keras"   ... 画像 (224,224,C) をそのまま入力する深層モデル
#       "sklearn" ... 平坦化 + PCA した特徴量を入力する非深層モデル
#                     (RandomForest / XGBRF など)
#    builder: RegressionModelMaker のインスタンスを受け取りモデルを返す関数
#
#    どのモデルを実行するかは下の ACTIVE_MODEL_KEYS で選ぶ。研究計画 (2026-06-12) の
#    「まず RandomForest 単体で回帰が成立する状態を作り、その後 3 モデルを同条件で
#    比較する」という段取りを、ここの 1 行だけで切り替えられるようにしている。
#       RF 単体の動作確認 : ACTIVE_MODEL_KEYS = ["rf"]
#       3 モデル同条件     : ACTIVE_MODEL_KEYS = ["rf", "cnntf_v2_gap", "alexnet"]
#    None にすると各 spec の "enabled" フラグに従う (従来動作)。
# ===================================================================
# Active model keys are derived from VALIDATION_CONFIG above.

MODEL_SPECS = [
    {
        "key": "rf",
        "label": "RandomForest",
        "kind": "sklearn",
        "builder": lambda mm, **params: mm.random_forest(**params),
        "enabled": True,
    },
    {
        "key": "cnntf_v2_gap",
        "label": "CNN+Tf v2 GAP",
        "kind": "keras",
        "builder": lambda mm, **params: mm.cnn_transformer_v2(pooling="gap", **params),
        "builder_params": {
            "num_transformer_blocks": 4,
            "head_size": 256,
            "num_heads": 4,
            "ff_dim": 2048,
            "model_dim": 32,
            "dropout": 0.2,
        },
        "input_axes_assumption": ["time_frame", "frequency_bin", "channel"],
        "architecture": {
            "front_end": "alexnet_like_cnn",
            "sequence_length_after_cnn": 7,
            "encoder": "transformer_encoder",
            "pooling": "GlobalAveragePooling1D",
        },
        "enabled": True,
    },
    {
        "key": "alexnet",
        "label": "AlexNet",
        "kind": "keras",
        "builder": lambda mm, **params: mm.alexnet(**params),
        "enabled": True,
    },
]




#### データフォルダの設定 ####
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT.joinpath(*_cfg("data", "experiment_root_parts"))

# matplotlib の設定
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#######################################################################
#                                実行部
#######################################################################

def safe_tag(text, max_len=32):
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")[:max_len] or "params"


def chunk_tag():
    if isinstance(CHUNK, (int, float)):
        return f"{CHUNK:g}s"
    return f"{CHUNK}s"


def snr_value_from_noise_dir(noise_dir_name):
    if noise_dir_name == "heatflux_no_noise":
        return "no_noise"
    if "SNR=" in noise_dir_name:
        return noise_dir_name.split("SNR=", 1)[1]
    return safe_tag(noise_dir_name)


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def has_threshold(threshold):
    if threshold is None:
        return False
    try:
        return np.isfinite(float(threshold))
    except (TypeError, ValueError):
        return False


def short_digest(payload, length=8):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=json_default)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def compact_weight_strategy_tag():
    return WEIGHT_STRATEGY_TAGS.get(WEIGHT_STRATEGY, safe_tag(WEIGHT_STRATEGY, max_len=8))


def run_config_digest(parameter_set, run_specs, model_tag):
    config = validation_config_snapshot()
    config["output"] = {
        "save_fold_predictions": SAVE_FOLD_PREDICTIONS,
    }
    return short_digest({
        "validation_config": config,
        "parameter_set": parameter_set,
        "model_tag": model_tag,
        "model_params": model_param_summary(run_specs),
    }, length=6)


def run_dir_name(param_tag, model_tag):
    parts = [
        f"e{EPOCH_NUM}",
        safe_tag(param_tag, max_len=24),
        safe_tag(model_tag, max_len=12),
    ]
    if ENSEMBLE_ENABLED:
        parts.append(compact_weight_strategy_tag())
    return "_".join(parts)


def has_input_files(data_path):
    extension = "*.npy" if COLOR_CHANNEL == 1 else "*.png"
    return data_path.is_dir() and any(data_path.glob(extension))


def find_data_source_dir(experiment_root, experiment_name):
    npy_root = experiment_root / "data" / "npy"
    configured = DATA_SOURCE_DIR_BY_EXPERIMENT.get(experiment_name)
    if configured:
        configured_path = npy_root / configured
        if configured_path.is_dir():
            return configured_path

    candidates = sorted(npy_root.glob(f"{NOISE_SOURCE_PREFIX}_*_{chunk_tag()}"))
    if not candidates:
        return None
    return candidates[-1]


def build_dataset_jobs():
    jobs = []
    missing = []
    for experiment_name in EXPERIMENT_DIR_NAMES:
        experiment_root = EXPERIMENT_ROOT / experiment_name
        source_dir = find_data_source_dir(experiment_root, experiment_name)
        threshold = THRESHOLD_BY_EXPERIMENT.get(experiment_name)
        for max_freq_name in MAX_FREQ_HZ_LIST:
            for noise_dir_name in NOISE_DIR_NAMES:
                data_path = None if source_dir is None else source_dir / max_freq_name / noise_dir_name
                job = {
                    "experiment_name": experiment_name,
                    "experiment_root": experiment_root,
                    "source_dir": source_dir,
                    "threshold": threshold,
                    "max_freq_hz": max_freq_name,
                    "noise_dir_name": noise_dir_name,
                    "snr_value": snr_value_from_noise_dir(noise_dir_name),
                    "data_path": data_path,
                    "save_base_path": (
                        experiment_root / "regression_result" / "npy" / RESULT_MODEL_GROUP / RESULT_DATE_DIR
                    ),
                }
                if REQUIRE_EXPERIMENT_THRESHOLD and threshold is None:
                    missing.append({**job, "missing_reason": "threshold"})
                elif data_path is not None and has_input_files(data_path):
                    jobs.append(job)
                else:
                    missing.append({**job, "missing_reason": "data"})

    print(f"dataset plan: existing={len(jobs)} / intended={len(EXPERIMENT_DIR_NAMES) * len(MAX_FREQ_HZ_LIST) * len(NOISE_DIR_NAMES)}")
    if missing:
        print("missing datasets:")
        for job in missing:
            missing_path = job["data_path"] if job["data_path"] is not None else job["experiment_root"] / "data" / "npy"
            reason = job.get("missing_reason", "data")
            print(f"  - {reason} | {job['experiment_name']} | {job['max_freq_hz']} | {job['noise_dir_name']} | {missing_path}")
        if not SKIP_MISSING_DATASETS:
            raise FileNotFoundError("Some intended datasets are missing. Set SKIP_MISSING_DATASETS=True to continue.")
    return jobs


def resolve_parameter_set(enabled_specs, parameter_set):
    resolved = []
    default_keras = parameter_set.get("default_keras", {})
    per_model = parameter_set.get("models", {})
    for spec in enabled_specs:
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
        elif resolved_spec["kind"] == "sklearn":
            params = {}
            params.update(per_model.get(resolved_spec["key"], {}))
            if params:
                resolved_spec["builder_params"] = params
        resolved.append(resolved_spec)
    return resolved


def parameter_set_tag(parameter_set, resolved_specs):
    if parameter_set.get("name"):
        return safe_tag(parameter_set["name"])
    parts = []
    for spec in resolved_specs:
        if spec["kind"] == "keras":
            parts.append(
                f"{spec['key']}_lr{format_param_value(spec['lr'])}"
                f"_bs{format_param_value(spec['batch_size'])}"
            )
    return safe_tag("__".join(parts))


def model_param_summary(resolved_specs):
    summary = {}
    for spec in resolved_specs:
        if spec["kind"] == "keras":
            item = {
                "lr": spec["lr"],
                "batch_size": spec["batch_size"],
            }
            if spec.get("builder_params"):
                item["builder_params"] = spec["builder_params"]
            if spec.get("input_axes_assumption"):
                item["input_axes_assumption"] = spec["input_axes_assumption"]
            if spec.get("actual_npy_axes"):
                item["actual_npy_axes"] = spec["actual_npy_axes"]
            if spec.get("architecture"):
                item["architecture"] = spec["architecture"]
            if spec.get("note"):
                item["note"] = spec["note"]
            summary[spec["key"]] = item
        else:
            summary[spec["key"]] = {
                "kind": spec["kind"],
                "params": spec.get("builder_params", {}),
            }
    return summary


def serializable_run_specs(run_specs):
    clean_specs = []
    for spec in run_specs:
        clean_specs.append({
            key: value
            for key, value in spec.items()
            if key not in {"builder"}
        })
    return clean_specs


def write_run_manifest(save_path, job, parameter_set, run_specs,
                       param_tag, model_tag, run_hash, run_dir):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_instance_id": RUN_INSTANCE_ID,
        "run_hash": run_hash,
        "run_dir": run_dir,
        "folder_naming": {
            "scheme": "e{epochs}_{param}_{models}[_{weight_when_ensemble_enabled}]",
            "reason": "Keep Windows paths short while keeping parameter folders readable.",
            "details": "Full conditions are stored in this manifest and validation_results_*.txt.",
        },
        "dataset": {
            "experiment_name": job["experiment_name"],
            "source_dir": str(job["source_dir"]),
            "data_path": str(job["data_path"]),
            "max_freq_hz": job["max_freq_hz"],
            "noise_dir_name": job["noise_dir_name"],
            "snr_value": job["snr_value"],
            "threshold": job["threshold"],
        },
        "parameter_set_name": parameter_set.get("name"),
        "parameter_set_tag": param_tag,
        "model_tag": model_tag,
        "model_params": model_param_summary(run_specs),
        "run_specs": serializable_run_specs(run_specs),
        "validation_config": validation_config_snapshot(),
    }
    manifest_path = os.path.join(save_path, "run_manifest.json")
    with _open_text(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2, default=json_default)


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
        "ensemble": {
            "enabled": ENSEMBLE_ENABLED,
            "weight_strategy": WEIGHT_STRATEGY,
            "fixed_weights": FIXED_WEIGHTS,
            "inner_holdout_frac": INNER_HOLDOUT_FRAC,
            "combine": ENSEMBLE_COMBINE,
        },
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
    if WEIGHT_STRATEGY not in {"simple", "fixed", "inner_holdout", "val_fold_legacy"}:
        raise ValueError(f"Unknown weight_strategy: {WEIGHT_STRATEGY}")
    if ENSEMBLE_COMBINE not in {"mean", "min"}:
        raise ValueError(f"Unknown ensemble combine method: {ENSEMBLE_COMBINE}")
    if not 0 < float(INNER_HOLDOUT_FRAC) < 1:
        raise ValueError("inner_holdout_frac must be between 0 and 1.")
    if int(PCA_COMPONENTS) <= 0:
        raise ValueError("pca_components must be a positive integer.")

    model_keys = [spec["key"] for spec in enabled_specs]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError(f"Duplicate active model keys: {model_keys}")

    if WEIGHT_STRATEGY == "fixed":
        missing_weights = [key for key in model_keys if key not in FIXED_WEIGHTS]
        if missing_weights:
            raise ValueError(f"FIXED_WEIGHTS does not define weights for: {missing_weights}")

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


def set_global_seed(seed):
    """Keep KFold, sklearn, and Keras runs as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _windows_long_path(path):
    path = os.path.abspath(path)
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + path
    return path


def _makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        os.makedirs(_windows_long_path(path), exist_ok=True)


def _path_exists(path):
    return os.path.exists(path) or os.path.exists(_windows_long_path(path))


def _open_text(path, mode, **kwargs):
    try:
        return open(path, mode, **kwargs)
    except OSError:
        return open(_windows_long_path(path), mode, **kwargs)


def _csv_metric(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(value):
        return ""
    return f"{value:.10g}"


def _join_unique(values):
    clean = [str(v) for v in values if v not in (None, "")]
    return "|".join(dict.fromkeys(clean))


def append_tuning_summary(summary_path, job, parameter_set, run_specs,
                          store, train_meta, summary_metrics, metrics,
                          param_tag, run_dir, run_hash, save_path,
                          model_keys):
    if not SAVE_TUNING_SUMMARY:
        return

    _makedirs(os.path.dirname(summary_path))
    header = [
        "created_at", "run_instance_id", "run_hash", "run_dir", "save_path",
        "experiment_name", "data_source_dir", "max_freq_hz", "noise_dir_name",
        "snr_value", "threshold_available", "threshold",
        "parameter_set", "model_key", "model_label",
        "lr", "batch_size",
        "requested_batch_size", "actual_batch_sizes", "min_actual_batch_size",
        "epochs_completed", "stopped_by_memory_error",
    ]
    for metric_name in summary_metrics:
        header.extend([f"{metric_name}_mean", f"{metric_name}_se"])

    spec_by_key = {spec["key"]: spec for spec in run_specs}
    file_exists = _path_exists(summary_path)
    with _open_text(summary_path, "a", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        if not file_exists:
            writer.writerow(header)
        for key in model_keys:
            spec = spec_by_key.get(key, {})
            meta = train_meta.get(key, {})
            actual_batch_sizes = meta.get("actual_batch_size", [])
            min_actual_batch_size = min(actual_batch_sizes) if actual_batch_sizes else ""
            stopped_flags = meta.get("stopped_by_memory_error", [])
            row = [
                datetime.now().isoformat(timespec="seconds"),
                RUN_INSTANCE_ID,
                run_hash,
                run_dir,
                save_path,
                job["experiment_name"],
                job["source_dir"],
                job["max_freq_hz"],
                job["noise_dir_name"],
                job["snr_value"],
                int(has_threshold(job["threshold"])),
                job["threshold"] if has_threshold(job["threshold"]) else "",
                parameter_set.get("name", param_tag),
                key,
                spec.get("label", key),
                spec.get("lr", ""),
                spec.get("batch_size", ""),
                _join_unique(meta.get("requested_batch_size", [])),
                _join_unique(actual_batch_sizes),
                min_actual_batch_size,
                _join_unique(meta.get("epochs_completed", [])),
                int(any(bool(flag) for flag in stopped_flags)),
            ]
            for metric_name in summary_metrics:
                mean, se = metrics.mean_se(store[key][metric_name])
                row.extend([_csv_metric(mean), _csv_metric(se)])
            writer.writerow(row)


def is_completed_run(summary_path, run_dir, save_path, snr_value):
    """Return True only when the per-run metrics and tuning summary both exist."""
    if not RESUME_COMPLETED_RUNS:
        return False

    metrics_path = os.path.join(save_path, f"metrics_summary_{snr_value}.csv")
    if not _path_exists(metrics_path):
        return False

    if not SAVE_TUNING_SUMMARY:
        return True
    if not _path_exists(summary_path):
        return False

    try:
        with _open_text(summary_path, "r", newline="", encoding="utf-8") as sf:
            for row in csv.DictReader(sf):
                if row.get("run_dir") == run_dir:
                    return True
    except (OSError, csv.Error):
        return False
    return False


def main():
    set_global_seed(RANDOM_SEED)
    # 再利用ヘルパー (用途別の utils クラス) を用意する
    metrics = RegressionDetectionMetrics()
    trainer = ModelTrainer(random_seed=RANDOM_SEED)
    weighting = EnsembleWeighting()
    plotter = RegressionPlotter()

    # 実行するモデルを決める。ACTIVE_MODEL_KEYS が指定されていればそれを優先し、
    # None のときだけ各 spec の "enabled" フラグに従う。
    if ACTIVE_MODEL_KEYS is None:
        enabled_specs = [s for s in MODEL_SPECS if s.get("enabled", True)]
    else:
        spec_by_key = {s["key"]: s for s in MODEL_SPECS}
        unknown = [k for k in ACTIVE_MODEL_KEYS if k not in spec_by_key]
        if unknown:
            raise ValueError(f"ACTIVE_MODEL_KEYS に未定義の key があります: {unknown} "
                             f"(定義済み: {list(spec_by_key)})")
        enabled_specs = [spec_by_key[k] for k in ACTIVE_MODEL_KEYS]
    if not enabled_specs:
        raise ValueError("実行するモデルが 0 個です。ACTIVE_MODEL_KEYS を確認してください。")

    # 出力先を混ぜないためのモデルセットのタグ。
    # Windows のパス長制限に当たりやすいため、代表的な3モデル構成は短いタグにする。
    validate_validation_config(enabled_specs)

    model_keys = [s["key"] for s in enabled_specs]
    model_tag = "-".join(model_keys)
    if SMOKE_TEST:
        model_tag = "s_" + model_tag

    use_sklearn = any(s["kind"] == "sklearn" for s in enabled_specs)
    include_ensemble = bool(ENSEMBLE_ENABLED and len(enabled_specs) >= 2)
    all_keys = [s["key"] for s in enabled_specs]
    if include_ensemble:
        all_keys.append("ensemble")
    label_of = {s["key"]: s["label"] for s in enabled_specs}
    if include_ensemble:
        label_of["ensemble"] = "Ensemble"

    print("#" * 60)
    if SMOKE_TEST:
        print("### SMOKE_TEST = True : 動作確認モードです ###")
        print(f"###   epoch={EPOCH_NUM} / fold={DIVISIONS} に縮小しています。")
        print("###   この結果は本番評価には使えません。本番では SMOKE_TEST = False に戻してください。")
    else:
        print("### 本番モード (SMOKE_TEST = False) ###")
    print(f"有効なモデル: {[s['label'] for s in enabled_specs]}  (model_tag={model_tag})")
    print(f"重み戦略: {WEIGHT_STRATEGY} | 統合: {ENSEMBLE_COMBINE} | epoch={EPOCH_NUM} | fold={DIVISIONS}")
    print("validation_config:")
    print(validation_config_text())
    if include_ensemble and WEIGHT_STRATEGY == "val_fold_legacy":
        print("【警告】val_fold_legacy は検証 fold の正解から重みを決めるリークあり方式です。"
              "旧結果の再現用にのみ使用してください。")
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

        print(
            f"\n{'='*40}\n"
            f"dataset {job_i}/{len(dataset_jobs)} | "
            f"{job['experiment_name']} | {max_freq_name} | {noise_dir_name}"
        )
        print(f"data_path={data_path}")
        print(f"threshold={threshold}")

        start_time = time.time()
        data_loading = DataLoadingConversion()
        if COLOR_CHANNEL == 1:
            x, y = data_loading.load_npy_data(data_path)
        else:
            x, y = data_loading.load_image_data(data_path)
        print(f"x shape: {x.shape} | y shape: {y.shape} | "
              f"読み込み {time.time() - start_time:.2f} 秒")

        for parameter_set in PARAMETER_SETS:
                run_specs = resolve_parameter_set(enabled_specs, parameter_set)
                param_tag = parameter_set_tag(parameter_set, run_specs)
                param_summary = model_param_summary(run_specs)
                run_hash = run_config_digest(parameter_set, run_specs, model_tag)
                run_dir = run_dir_name(param_tag, model_tag)
                print(f"parameter_set={parameter_set.get('name', param_tag)} | model_params={param_summary}")
                print(f"run_dir={run_dir}")
                SAVE_PATH = os.path.join(
                    base_save_path, noise_dir_name, max_freq_name,
                    run_dir)
                tuning_summary_path = os.path.join(base_save_path, "tuning_summary.csv")
                if is_completed_run(tuning_summary_path, run_dir, SAVE_PATH, snr_value):
                    print(f"[resume skip] completed run found: {run_dir}")
                    continue

                _makedirs(SAVE_PATH)
                write_run_manifest(
                    SAVE_PATH, job, parameter_set, run_specs,
                    param_tag, model_tag, run_hash, run_dir)

                kf = KFold(n_splits=DIVISIONS, shuffle=True, random_state=RANDOM_SEED)

                # 指標の保存先 (key -> metric -> [fold ごとの値])
                store = {k: defaultdict(list) for k in all_keys}
                train_meta = {k: defaultdict(list) for k in model_keys}
                # 重みの記録 (fold ごと)
                weight_log = []

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
                    f.write(f"model_params={param_summary}\n")
                    f.write(f"weight_strategy={WEIGHT_STRATEGY}, combine={ENSEMBLE_COMBINE}\n")
                    f.write("=" * 30 + "\n")

                    fold = 1
                    for train_index, val_index in kf.split(x):
                        x_train, x_val = x[train_index], x[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        # --- inner_holdout のときだけ学習 fold を内部分割 ---
                        if WEIGHT_STRATEGY == "inner_holdout":
                            x_fit, x_inner, y_fit, y_inner = train_test_split(
                                x_train, y_train, test_size=INNER_HOLDOUT_FRAC,
                                random_state=RANDOM_SEED)
                        else:
                            x_fit, y_fit = x_train, y_train
                            x_inner = y_inner = None

                        # スケーラは学習に使うラベル (y_fit) のみで fit する
                        scaler = MinMaxScaler()
                        y_fit_scaled = scaler.fit_transform(y_fit.reshape(-1, 1))

                        # sklearn 系モデル用の PCA (学習データのみで fit)
                        pca_model = None
                        if use_sklearn:
                            if EXPLAINABILITY_ENABLED:
                                x_fit_pca, (x_val_pca, x_inner_pca), pca_model = trainer.make_pca(
                                    x_fit, [x_val, x_inner], PCA_COMPONENTS, return_pca=True)
                            else:
                                x_fit_pca, (x_val_pca, x_inner_pca) = trainer.make_pca(
                                    x_fit, [x_val, x_inner], PCA_COMPONENTS)
                        else:
                            x_fit_pca = x_val_pca = x_inner_pca = None

                        mm = RegressionModelMaker((224, 224, COLOR_CHANNEL))

                        # --- 各モデルの学習と予測 ---
                        val_preds = {}        # key -> 検証 fold への予測 (元スケール)
                        errors_for_weight = {}  # key -> 重み用の誤差 (1 - R2)
                        for spec in run_specs:
                            if spec["kind"] == "keras":
                                print(f"  params for {spec['key']}: lr={spec['lr']}, batch_size={spec['batch_size']}")
                            print(f"[{spec['label']}] Fold {fold}/{DIVISIONS} 学習開始")
                            model, history = trainer.train_one_model(
                                spec, mm, x_fit, y_fit_scaled, x_fit_pca,
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

                            # 検証 fold への予測
                            val_preds[spec["key"]] = trainer.predict_one_model(
                                spec, model, x_val, x_val_pca, scaler)

                            maybe_explain_trained_model(
                                spec, model, scaler, x_val, y_val,
                                val_preds[spec["key"]], threshold, SAVE_PATH,
                                fold, max_freq_name, EXPLAINABILITY_CONFIG,
                                pca=pca_model)

                            # --- 重み用の誤差 (③ 戦略ごとにリークしない/する を切替) ---
                            if WEIGHT_STRATEGY == "inner_holdout":
                                inner_pred = trainer.predict_one_model(
                                    spec, model, x_inner, x_inner_pca, scaler)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_inner, inner_pred)
                            elif WEIGHT_STRATEGY == "val_fold_legacy":
                                # 旧来どおり検証 fold 誤差 (リークあり)
                                errors_for_weight[spec["key"]] = 1.0 - r2_score(y_val, val_preds[spec["key"]])

                            del model
                            del history
                            K.clear_session()
                            gc.collect()

                        # --- アンサンブル ---
                        weights = {}
                        ensemble_pred = None
                        if include_ensemble:
                            weights = weighting.compute_weights(
                                WEIGHT_STRATEGY, run_specs, errors_for_weight, FIXED_WEIGHTS)
                            weight_log.append((fold, dict(weights)))
                            ensemble_pred = weighting.combine_predictions(
                                val_preds, weights, ENSEMBLE_COMBINE)

                        # --- 指標の算出 (② 3 種類に分離) ---
                        preds_all = dict(val_preds)
                        if include_ensemble:
                            preds_all["ensemble"] = ensemble_pred
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

                        # --- 作図 (アンサンブルの散布図) ---
                        if include_ensemble and has_threshold(threshold):
                            ens_fold_metrics = {mk: store["ensemble"][mk][-1]
                                                for mk in store["ensemble"]}
                            plotter.plot_regression_scatter(
                                y_val, ensemble_pred, y, ens_fold_metrics,
                                threshold, SAVE_PATH, snr_value, fold)

                        # --- fold 結果を txt に追記 ---
                        f.write(f"Recorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                        f.write(f"Fold {fold} Results | weights={ {k: round(v,3) for k,v in weights.items()} }\n")
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
                        del x_fit, y_fit, x_inner, y_inner, y_fit_scaled
                        del pca_model
                        del x_fit_pca, x_val_pca, x_inner_pca
                        del val_preds, preds_all, ensemble_pred
                        K.clear_session()
                        gc.collect()

                    # --- 平均結果 (mean ± SE) ---
                    f.write(f"\nRecorded at: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                    f.write("Average Results (mean ± SE):\n")
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
                            f.write(f"    {mk:14s}: {mean:.4f} ± {se:.4f}\n")
                    f.write("=" * 30 + "\n\n")

                if include_ensemble:
                    weights_csv = os.path.join(SAVE_PATH, f"ensemble_weights_{snr_value}.csv")
                    with _open_text(weights_csv, "w", newline="", encoding="utf-8") as wf:
                        writer = csv.writer(wf)
                        writer.writerow(["fold"] + model_keys)
                        for fold_num, weights in weight_log:
                            writer.writerow(
                                [fold_num] + [f"{float(weights.get(key, 0.0)):.10g}" for key in model_keys]
                            )

                # --- 棒グラフ (モデル別: R2 と旧コード互換の二値化後 AUC) ---
                labels = [label_of[k] for k in all_keys]
                r2_means = [metrics.mean_se(store[k]["r2"])[0] for k in all_keys]
                r2_ses = [metrics.mean_se(store[k]["r2"])[1] for k in all_keys]
                auc_bin_means = [metrics.mean_se(store[k]["auc_binary"])[0] for k in all_keys]
                auc_bin_ses = [metrics.mean_se(store[k]["auc_binary"])[1] for k in all_keys]
                plotter.plot_bar("R2 Score", labels, r2_means, r2_ses, EPOCH_NUM, SAVE_PATH, snr_value)
                if has_threshold(threshold):
                    plotter.plot_bar("AUC (binary legacy)", labels, auc_bin_means, auc_bin_ses,
                                     EPOCH_NUM, SAVE_PATH, snr_value)

                # --- 指標 CSV (fold 平均をモデル別に保存。後で比較しやすくする) ---
                csv_path = os.path.join(SAVE_PATH, f'metrics_summary_{snr_value}.csv')
                with _open_text(csv_path, 'w', encoding='utf-8') as cf:
                    header = ["model"] + [f"{mk}_mean" for mk in summary_metrics] \
                                       + [f"{mk}_se" for mk in summary_metrics]
                    cf.write(",".join(header) + "\n")
                    for key in all_keys:
                        means = [f"{metrics.mean_se(store[key][mk])[0]:.6f}" for mk in summary_metrics]
                        ses = [f"{metrics.mean_se(store[key][mk])[1]:.6f}" for mk in summary_metrics]
                        cf.write(",".join([label_of[key]] + means + ses) + "\n")
                print(f"指標 CSV を保存: {csv_path}")

                append_tuning_summary(
                    tuning_summary_path, job, parameter_set, run_specs,
                    store, train_meta, summary_metrics, metrics,
                    param_tag, run_dir, run_hash, SAVE_PATH, all_keys)
                print(f"tuning summary saved: {tuning_summary_path}")

                if not FLG_ROOP:
                    break

        del x, y
        gc.collect()


if __name__ == '__main__':
    main()
