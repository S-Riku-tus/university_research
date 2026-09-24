"""
音響スペクトログラムから熱流束を回帰し、ONB判定と説明性を評価する実行コード。

実験条件はVALIDATION_CONFIGで指定する。主な処理は以下のとおり。
1. 元WAVを分離する交差検証、または実験日全体を除外する分割で学習する。
2. 同じ検証予測から、各単体モデルと指定したアンサンブル方式を評価する。
3. 1秒chunk単位の通常指標を、同じ学習外予測から保存する。
4. R²と連続予測のROC-AUCを記録する。
5. 説明性、予測散布図、モデル比較図を保存する。

学習・評価・統合・作図はutils配下の共通処理を呼び出す。
旧実行コードは再現用に保持し、通常の実験には本ファイルを使う。
"""

import os
import re
from datetime import datetime
from pathlib import Path
from pprint import pformat

# 学習前のGPUメモリ一括確保を避け、必要な分だけ順次確保する。
# Windowsでメモリ不足が起きた際、バッチサイズを下げた再試行を可能にする。
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
# Research runs compare small differences across repeated fits. Request
# deterministic TensorFlow/CUDA kernels before TensorFlow is imported.
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np

from utils.training.model_training import ModelTrainer
from utils.ensemble.ensemble_runtime import EnsembleManager
from utils.plotting.regression_plots import RegressionPlotter
from utils.config.parameter_sets import (
    expand_parameter_sets,
    resolve_parameter_set,
)
from utils.config.onb_defaults import apply_onb_defaults, onb_model_specs
from utils.explainability.training_integration import (
    resolve_explainability_scope,
)
from utils.experiment.dataset_jobs import build_dataset_jobs as make_dataset_jobs
from utils.experiment.learning_policy import (
    normalize_learning_policy, policy_result_date_dir, resolve_experiment_names,
)
from utils.experiment.learning_runner import run_learning_experiments
from utils.experiment.result_paths import existing_result_run_path, noise_trend_path
from utils.experiment.onb_thresholds import (
    onb_threshold_by_experiment,
    onb_threshold_provenance_by_experiment,
)
from utils.experiment.run_helpers import set_global_seed


#######################################################################
#                         実験条件の設定
#######################################################################
# 実験条件を変更するときは、まずVALIDATION_CONFIGを編集する。
# 実験ごとに変更する条件をここにまとめる。
# 固定的な出力・説明性とモデル対応表はutils/config/onb_defaults.py。
#
# 現在の設定の読み方:
# ・目的: 同じ検証予測から単体モデルと各アンサンブル方式を比較する。
# ・データ: explicit_daysでは学習日とテスト日の和集合、他の分割ではexperiment_namesを使う。
# ・モデル: RF、CNN＋Transformer、AlexNet。
# ・統合方式: ensembleに列挙した全方式を実行・評価する。
# ・説明性: 有効なデータ条件・モデル・foldについて指定手法を実行する。
#
# ensembleには実行する方式名を指定する。
# 重みの計算や予測の統合処理はutils/ensemble/で管理する。

VALIDATION_CONFIG = apply_onb_defaults({
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
        "noise_source": "waterflow",  # 水流音はwaterflow、白色雑音はwhitenoise
        "chunk_seconds": 1,
        "max_freq_hz_list": [
            "maxfreq=3kHz",
            "maxfreq=5kHz",
            "maxfreq=10kHz",
            "maxfreq=15kHz",
            "maxfreq=22kHz",
        ],
        "noise_dir_names": [
            "heatflux_no_noise",
            "heatflux_reference_SNR=0",
            "heatflux_reference_SNR=-4",
            "heatflux_reference_SNR=-8",
            "heatflux_reference_SNR=-12",
            "heatflux_reference_SNR=-16",
            "heatflux_reference_SNR=-20",
        ],
        "data_source_dir_by_experiment": {
            "2025.06.11_0.3_2": "waterflow_20260817_1s",
            "2025.06.18_0.3_3": "waterflow_20260817_1s",
            "2025.07.09_0.3_1": "waterflow_20260817_1s",
        },
        "skip_missing_datasets": False,
    },
    "learning_policy": {
        # 学習日・テスト日を明示指定。テスト日は重み・PCA・選別閾値のfitに使わない。
        # 旧within_day / leave_one_day_outでは下記のtrain/test指定を外し、
        # data.experiment_namesに評価対象日を指定する。
        "split_mode": "explicit_days",
        "train_experiments": [
            "2025.06.11_0.3_2",
            # "2025.07.09_0.3_1",
            "2025.06.18_0.3_3",
            ],
        "test_experiments": [
            # "2025.06.11_0.3_2",
            "2025.07.09_0.3_1",
            # "2025.06.18_0.3_3",
            ],
        # 重み決定でも同じ元WAVの1秒区間を学習・検証へ分けない。
        # 内部fold数はrun.folds。chunk_kfoldは旧比較の再現時だけ使う。
        "internal_validation": "wav_kfold",
        # matched: ノイズ条件ごとに独立して学習し、PCA・scaler・epoch・
        #          アンサンブル重みもそのノイズの学習データから毎回求める。
        # clean_only: 無雑音だけで学習し、同じモデル・前処理・重みで
        #             全評価ノイズを予測する固定clean診断。
        # 評価ノイズ一覧から無雑音を外しても、学習には無雑音を読み込む。
        "training_noise": "matched",
    },
    "acoustic_selection": {
        # スペクトルの縦軸に引く横線。図の「×10^-9」表示で高さ1に相当。
        # 0.3e-9なら弱い秒も含む。3e-9 / 10e-9なら大きいピークの秒に絞る。
        # Noneなら選別なし。特徴量・対象範囲などの固定条件はonb_defaults.pyで管理する。
        "peak_height_threshold": 1.0e-9,
    },
    "thresholds": {
        # ONBと確認された最初の測定点の熱流束と、その出典を一元管理する。
        # ONB直前の測定点を誤って閾値に使わないようにする。
        "by_experiment": onb_threshold_by_experiment(),
        "provenance_by_experiment": onb_threshold_provenance_by_experiment(),
        "require_experiment_threshold": True,
        "onb_band_frac": 0.10,
    },
    "models": {
        "active_model_keys": ["randomforest", "conformer", "alexnet"],
        # active_model_keysに指定したモデルだけを学習する。
        # 各候補リストが1要素なら固定条件、複数要素なら組み合わせを比較する。
        # 無効なモデルの候補設定は実行に影響しない。
        "parameter_sets": {
            "type": "active_model_grid",
            "model_grids": {
                "randomforest": {
                    # RFの木の数・深さを設定する。
                    # サンプルと特徴の抽出率もここで指定する。
                    "n_estimators": [300],
                    "max_depth": [4],
                    "subsample": [0.6],
                    "colsample_bynode": [0.6],
                },
                "conformer": {
                    "lr": [0.001],
                    "batch_size": [12],
                },
                "alexnet": {
                    "lr": [0.001],
                    "batch_size": [12],
                },
            },
            "default_keras": {
                # KerasのTTY依存バーではなく、全実行環境で残る共通進捗行を使う。
                "fit_verbose": 0,
                "progress_interval_epochs": 10,
            },
        },
    },
    "ensemble": {
        # 実行するアンサンブル方式
        "enabled_strategy_names": [
            # "performance_kfold",  # learning_policy.internal_validation単位のOOF単体性能で重み付け
            # "simple_equal",  # 等しい重みで平均
            "inner_holdout",  # 学習データの約20%を、元WAVが重ならないように一度だけ取り分ける
        ],
    },
    "features": {
        "pca_components": 100,
    },
})


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
DETERMINISTIC_OPS = True
FLG_ROOP = _cfg("run", "loop_parameter_sets")

NOISE_SOURCE_PREFIX = _noise_source_prefix(_cfg("data", "noise_source"))
CHUNK = _cfg("data", "chunk_seconds")
EXPERIMENT_DIR_NAMES = resolve_experiment_names(
    VALIDATION_CONFIG["data"], VALIDATION_CONFIG["learning_policy"])
MAX_FREQ_HZ_LIST = _cfg("data", "max_freq_hz_list")
NOISE_DIR_NAMES = _cfg("data", "noise_dir_names")
DATA_SOURCE_DIR_BY_EXPERIMENT = _cfg("data", "data_source_dir_by_experiment")
SKIP_MISSING_DATASETS = _cfg("data", "skip_missing_datasets")
LEARNING_POLICY = normalize_learning_policy(
    VALIDATION_CONFIG["learning_policy"], EXPERIMENT_DIR_NAMES, COLOR_CHANNEL)
EVALUATION_FOLDS = DIVISIONS if LEARNING_POLICY["split_mode"] == "within_day" else 1

THRESHOLD_BY_EXPERIMENT = _cfg("thresholds", "by_experiment")
THRESHOLD_PROVENANCE_BY_EXPERIMENT = _cfg(
    "thresholds", "provenance_by_experiment"
)
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
    else "randomforest" if ACTIVE_MODEL_KEYS == ["randomforest"]
    else "conformer" if ACTIVE_MODEL_KEYS == ["conformer"]
    else "alexnet" if ACTIVE_MODEL_KEYS == ["alexnet"]
    else "single_model"
)

PCA_COMPONENTS = _cfg("features", "pca_components")
TRAINING_VALIDATION_CONFIG = dict(
    VALIDATION_CONFIG.get("training_validation", {})
)

SAVE_DATE = _cfg("output", "save_date")
RESULT_DATE_DIR = _cfg("output", "result_date_dir") or SAVE_DATE
SAVE_FOLD_PREDICTIONS = _cfg("output", "save_fold_predictions")
SAVE_TUNING_SUMMARY = _cfg("output", "save_tuning_summary")
SAVE_FITTED_ARTIFACTS = _cfg("output", "save_fitted_artifacts")
VERIFY_RELOADED_ARTIFACTS = _cfg("output", "verify_reloaded_artifacts")
RESUME_COMPLETED_RUNS = _cfg("output", "resume_completed_runs")
NOISE_TREND_CONFIG = _cfg("output", "noise_trend_plots")
# 日付は上位フォルダにあるため、実行IDは時分秒の6桁だけにする。
# 同じRUN_IDを明示した場合だけ同一実行として再開する。
RUN_INSTANCE_ID = os.environ.get("RUN_ID") or datetime.now().strftime("%H%M%S")
if not re.fullmatch(r"\d{6}", RUN_INSTANCE_ID):
    raise ValueError("RUN_IDはHHMMSS形式の6桁（時分秒）で指定してください。")
# 同じRUN_IDを明示した再実行では、同じ保存先を参照して完了判定する。
EXECUTION_ID = RUN_INSTANCE_ID
FOLD_PREDICTIONS_DIR_NAME = "fold_pred"
EXPLAINABILITY_CONFIG = resolve_explainability_scope(
    VALIDATION_CONFIG.get("explainability", {}),
    experiment_names=EXPERIMENT_DIR_NAMES,
    max_freq_hz_list=MAX_FREQ_HZ_LIST,
    noise_dir_names=NOISE_DIR_NAMES,
    model_keys=ACTIVE_MODEL_KEYS,
    fold_count=EVALUATION_FOLDS,
)
EXPLAINABILITY_ENABLED = EXPLAINABILITY_CONFIG.get("enabled", False)

# ここから下の定数はVALIDATION_CONFIGから算出する。
# 通常の実験条件変更では、上の設定欄を編集する。
# 実行処理そのものを変更するとき以外は、導出値を直接変更しない。


# モデル名と、実際に呼び出すモデル構築関数の対応表。
# 学習モデルと表示ラベルを一致させるために使用する。
# モデル構造は各構築関数内に固定している。
# 学習率とバッチサイズなどは上のparameter_setsで指定する。


MODEL_SPECS = onb_model_specs()




# データフォルダの設定
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT.joinpath(*_cfg("data", "experiment_root_parts"))

# グラフ全体の表示書式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#######################################################################
#                            実行処理
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
        result_date_dir=policy_result_date_dir(RESULT_DATE_DIR, LEARNING_POLICY),
        color_channel=COLOR_CHANNEL,
        require_experiment_threshold=REQUIRE_EXPERIMENT_THRESHOLD,
        skip_missing_datasets=SKIP_MISSING_DATASETS,
        learning_policy=LEARNING_POLICY,
    )


def validation_config_snapshot():
    return {
        "learning_policy": dict(LEARNING_POLICY),
        "acoustic_selection": dict(VALIDATION_CONFIG.get("acoustic_selection", {})),
        "run": {
            "smoke_test": SMOKE_TEST,
            "epochs": EPOCH_NUM,
            "folds": DIVISIONS,
            "color_channel": COLOR_CHANNEL,
            "random_seed": RANDOM_SEED,
            "deterministic_ops": DETERMINISTIC_OPS,
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
            "provenance_by_experiment": THRESHOLD_PROVENANCE_BY_EXPERIMENT,
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
        "training_validation": dict(TRAINING_VALIDATION_CONFIG),
        "output": {
            "save_date": SAVE_DATE,
            "result_date_dir": RESULT_DATE_DIR,
            "run_instance_id": RUN_INSTANCE_ID,
            "execution_id": EXECUTION_ID,
            "run_scoped_result_dir": True,
            "save_fold_predictions": SAVE_FOLD_PREDICTIONS,
            "save_tuning_summary": SAVE_TUNING_SUMMARY,
            "save_fitted_artifacts": SAVE_FITTED_ARTIFACTS,
            "verify_reloaded_artifacts": VERIFY_RELOADED_ARTIFACTS,
            "resume_completed_runs": RESUME_COMPLETED_RUNS,
            "noise_trend_plots": dict(NOISE_TREND_CONFIG),
        },
        "explainability": EXPLAINABILITY_CONFIG,
    }


def validation_config_text():
    return pformat(validation_config_snapshot(), sort_dicts=False)


def update_noise_trend_plots(plotter, job, run_dir, run_hash, model_keys):
    """同じ実験日・周波数・学習設定のノイズ別指標を読み、比較図を更新する。"""
    if not NOISE_TREND_CONFIG.get("enabled", False):
        return []
    run_paths = [
        existing_result_run_path({**job, "noise_dir_name": noise_dir}, run_dir)
        for noise_dir in NOISE_DIR_NAMES
    ]
    output_dir = noise_trend_path(job, run_dir)
    artifacts = plotter.plot_noise_trends(
        run_paths,
        output_dir,
        noise_order=NOISE_DIR_NAMES,
        model_keys=model_keys,
        ensemble_strategy_names=NOISE_TREND_CONFIG["ensemble_strategy_names"],
        metrics=NOISE_TREND_CONFIG["metrics"],
        expected_run_hash=run_hash,
        formats=NOISE_TREND_CONFIG["formats"],
    )
    if artifacts:
        print(f"ノイズ強度別の比較図を更新: {output_dir}（{len(artifacts)}図）")
    return artifacts


def validate_validation_config(enabled_specs):
    if (LEARNING_POLICY["split_mode"] == "within_day" or "performance_kfold" in ENSEMBLE_MANAGER.selected_strategy_names) and DIVISIONS < 2:
        raise ValueError("folds must be at least 2.")
    if not PARAMETER_SETS:
        raise ValueError("VALIDATION_CONFIG['models']['parameter_sets'] must not be empty.")
    if int(PCA_COMPONENTS) <= 0:
        raise ValueError("pca_components must be a positive integer.")
    if TRAINING_VALIDATION_CONFIG.get("enabled", False):
        if TRAINING_VALIDATION_CONFIG.get("mode") != "wav_kfold":
            raise ValueError("training_validation.mode must be 'wav_kfold'.")
        if int(TRAINING_VALIDATION_CONFIG.get("folds", 0)) < 2:
            raise ValueError("training_validation.folds must be at least 2.")
        if int(TRAINING_VALIDATION_CONFIG.get("checkpoint_interval_epochs", 0)) < 1:
            raise ValueError(
                "training_validation.checkpoint_interval_epochs must be positive."
            )
        minimum_epoch = int(TRAINING_VALIDATION_CONFIG.get("minimum_epoch", 1))
        if not 1 <= minimum_epoch <= int(EPOCH_NUM):
            raise ValueError(
                "training_validation.minimum_epoch must be within the run epochs."
            )
        if TRAINING_VALIDATION_CONFIG.get("selection_metric") != "rmse_all":
            raise ValueError(
                "training_validation.selection_metric currently supports rmse_all only."
            )
    model_keys = [spec["key"] for spec in enabled_specs]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError(f"Duplicate active model keys: {model_keys}")

    for parameter_set in PARAMETER_SETS:
        if not isinstance(parameter_set, dict):
            raise TypeError("Each expanded parameter set must be a dict.")
        # 有効な深層モデルすべてに学習率とバッチサイズが設定されているか確認する。
        # 無効なモデルにだけ属するパラメータは無視する。
        resolve_parameter_set(enabled_specs, parameter_set)

    ENSEMBLE_MANAGER.validate(enabled_specs)
    if (LEARNING_POLICY != {"split_mode": "within_day", "training_noise": "matched"}
            and len(enabled_specs) > 1 and ENSEMBLE_MANAGER.has_leaky_strategy):
        raise ValueError("一般化評価では評価ラベルで重みを決めるval_fold_legacyを使用できません。")

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
            if not 1 <= int(fold) <= int(EVALUATION_FOLDS)
        ]
        if invalid_folds:
            raise ValueError(
                f"Explainability target_folds must be within 1..{EVALUATION_FOLDS}: "
                f"{invalid_folds}")
        if int(EXPLAINABILITY_CONFIG.get("max_samples_per_fold", 0)) <= 0:
            raise ValueError("Explainability max_samples_per_fold must be positive.")
        ig_steps = int(EXPLAINABILITY_CONFIG.get("ig_steps", 64))
        if ig_steps < 2 or int(EXPLAINABILITY_CONFIG.get("ig_max_steps", 4096)) < 2*ig_steps:
            raise ValueError("IG requires ig_steps >= 2 and ig_max_steps >= 2*ig_steps.")
        if int(EXPLAINABILITY_CONFIG.get("ig_batch_size", 8)) <= 0:
            raise ValueError("IG batch size must be positive.")
        for key in ("ig_rtol", "ig_atol", "ig_map_rtol"):
            if not np.isfinite(float(EXPLAINABILITY_CONFIG[key])) or float(EXPLAINABILITY_CONFIG[key]) < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")

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
    set_global_seed(RANDOM_SEED, deterministic_ops=DETERMINISTIC_OPS)
    # 指標計算、学習、統合、作図の共通処理を用意する。
    trainer = ModelTrainer(random_seed=RANDOM_SEED)
    plotter = RegressionPlotter()

    # 設定欄で指定したモデルの定義を取得する。
    spec_by_key = {s["key"]: s for s in MODEL_SPECS}
    unknown = [k for k in ACTIVE_MODEL_KEYS if k not in spec_by_key]
    if unknown:
        raise ValueError(f"ACTIVE_MODEL_KEYS has unknown keys: {unknown} "
                         f"(defined: {list(spec_by_key)})")
    enabled_specs = [spec_by_key[k] for k in ACTIVE_MODEL_KEYS]
    if not enabled_specs:
        raise ValueError("ACTIVE_MODEL_KEYS must select at least one model.")

    # Windowsのパス長制限を考慮し、結果フォルダ名を短くする。
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

    run_learning_experiments(
        dataset_jobs, LEARNING_POLICY, validation_config_snapshot(),
        enabled_specs, PARAMETER_SETS, ENSEMBLE_MANAGER, trainer, plotter,
        update_noise_trend_plots,
    )


if __name__ == '__main__':
    main()
