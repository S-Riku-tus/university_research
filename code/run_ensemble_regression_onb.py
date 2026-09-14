"""
音響スペクトログラムから熱流束を回帰し、ONB判定と説明性を評価する実行コード。

実験条件はVALIDATION_CONFIGで指定する。主な処理は以下のとおり。
1. 元WAVを分離した交差検証で、RF・CNN＋Transformer・AlexNetを学習する。
2. 同じ検証予測から、各単体モデルと指定したアンサンブル方式を評価する。
3. chunk単位のfold平均と、元WAV単位の全OOF集約評価を両方保存する。
4. R²、連続予測のROC-AUC、二値化後AUCを区別して記録する。
5. ONB遷移、説明性、予測散布図、モデル比較図を保存する。

学習・評価・統合・作図はutils配下の共通処理を呼び出す。
旧実行コードは再現用に保持し、通常の実験には本ファイルを使う。
"""

import os
import gc
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pformat

# 学習前のGPUメモリ一括確保を避け、必要な分だけ順次確保する。
# Windowsでメモリ不足が起きた際、バッチサイズを下げた再試行を可能にする。
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import backend as K

from utils.models.regression.base_regression import RegressionModelMaker
from utils.dataloading.dataloading_and_conversion import DataLoadingConversion
from utils.calculation.regression_detection_metrics import RegressionDetectionMetrics
from utils.calculation.wav_event_metrics import (
    build_fold_prediction_rows,
    save_wav_event_evaluation,
    write_fold_prediction_csv,
)
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
from utils.experiment.onb_thresholds import (
    onb_threshold_by_experiment,
    onb_threshold_provenance_by_experiment,
)
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
#                         実験条件の設定
#######################################################################
# 実験条件を変更するときは、まずVALIDATION_CONFIGを編集する。
# 学習・評価・説明性・作図の設定をここにまとめる。
#
# 現在の設定の読み方:
# ・目的: 同じ検証予測から単体モデルと各アンサンブル方式を比較する。
# ・データ: experiment_namesで有効にした実験日と周波数・ノイズ条件を使う。
# ・モデル: RF、CNN＋Transformer、AlexNet。
# ・統合方式: ensembleに列挙した方式を実行し、主方式を別途指定する。
# ・説明性: 有効なデータ条件・モデル・foldについて指定手法を実行する。
#
# ensembleには方式名と主方式を指定する。
# 重みの計算や予測の統合処理はutils/ensemble/で管理する。

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
        "noise_source": "waterflow",  # 水流音はwaterflow、白色雑音はwhitenoise
        "chunk_seconds": 1,
        "experiment_names": [
            "2025.06.11_0.3_2",
            "2025.06.18_0.3_3",
            # "2025.07.09_0.3_1",
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
    "thresholds": {
        # ONBと確認された最初の測定点の熱流束と、その出典を一元管理する。
        # ONB直前の測定点を誤って閾値に使わないようにする。
        "by_experiment": onb_threshold_by_experiment(),
        "provenance_by_experiment": onb_threshold_provenance_by_experiment(),
        "require_experiment_threshold": True,
        "onb_band_frac": 0.10,
    },
    "models": {
        "active_model_keys": ["rf", "cnntf_v2_gap", "alexnet"],
        # active_model_keysに指定したモデルだけを学習する。
        # 各候補リストが1要素なら固定条件、複数要素なら組み合わせを比較する。
        # 無効なモデルの候補設定は実行に影響しない。
        "parameter_sets": {
            "type": "active_model_grid",
            "model_grids": {
                "rf": {
                    # RFの木の数・深さを設定する。
                    # サンプルと特徴の抽出率もここで指定する。
                    "n_estimators": [300],
                    "max_depth": [4],
                    "subsample": [0.6],
                    "colsample_bynode": [0.6],
                },
                "cnntf_v2_gap": {
                    "lr": [0.001],
                    "batch_size": [16],
                },
                "alexnet": {
                    "lr": [0.001],
                    "batch_size": [16],
                },
            },
            "default_keras": {
                "fit_verbose": 1,
            },
        },
    },
    "ensemble": {
        # 実行する統合方式を名前で選択する。
        # 重み付け、内部検証の分割率、情報混在の制約は共通処理側で管理する。
        "enabled_strategy_names": [
            "simple_equal",
            "prediction_max",
            "inner_holdout",
        ],
        # 主方式は外側検証の正解値を使わない単純平均とする。
        "primary_strategy_name": "simple_equal",
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
        "noise_trend_plots": {
            # 各ノイズ条件の完了時と再開時に、保存指標から折れ線グラフを更新する。
            "enabled": True,
            # primaryは主方式のみ。allなら各方式につき単体3モデル＋統合の図を作る。
            # ["simple_equal", "inner_holdout"]のように方式を指定することもできる。
            "ensemble_strategy_names": "all",
            # R²、連続予測によるROC-AUC、卒論互換の二値化後AUCを別図で保存する。
            "metrics": ["r2", "roc_auc_cont", "auc_binary"],
            # chunkはfold平均±標準誤差、wavは全OOFの元録音集約値を表示する。
            "evaluation_units": ["chunk", "wav"],
            # WAVの集約方法にはevaluationのprimary_wav_aggregationを使用する。
            "formats": ["png", "pdf"],
        },
    },
    "evaluation": {
        # 同じ学習・検証予測から、chunk単位と元WAV単位の両方を評価する。
        # chunk指標はfoldごとに計算し、その平均と標準誤差を保存する。
        # WAV指標は全foldの学習外予測を集め、元録音ごとに集約して計算する。
        "wav_level_enabled": True,
        "wav_aggregations": ["mean", "median", "p90"],
        "primary_wav_aggregation": "median",
        # 最初の陽性点と、2 WAV連続で陽性になる区間の開始点を併記する。
        # 単発の誤警報と持続的な遷移を、熱流束の測定点順に確認する。
        # この測定点差は秒単位の検知遅れではない。
        "onb_transition_persistence_wavs": [1, 2],
        # WAV内のしきい値交差も診断用に保存する。
        # 同期したイベント正解がないため、気泡イベントの検出精度とは区別する。
        "predicted_event_summary_enabled": True,
    },
    "explainability": {
        # 学習済みの各foldモデルについて、検証データ上の説明性を追加評価する。
        # 各実行フォルダ内の次の場所に保存する。
        # 保存先: <SAVE_PATH>/explainability/fold{n}/{model_key}/
        #
        # 対象データ、モデル、foldはdata/models/runの設定から引き継ぐ。
        # 説明性だけ別の対象条件へずれないようにする。
        "enabled": True,
        "max_samples_per_fold": 5,
        "ig_steps": 64,
        # モデル構造に適した説明手法を指定する。
        # RFのTreeSHAPはPCA成分の監査用、物理帯域の比較にはマスクを使う。
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
        # マスク後の性能も、主評価と同じ元WAV中央値の単位で比較する。
        "performance_evaluation_unit": "source_wav",
        "performance_wav_aggregation": "median",
        "baseline_value": 0.0,
        "curve_fractions": [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
        # 入力への小さな非負摂動でIG画像の局所的な安定性を調べる。
        # ノイズ条件を変えたときの予測性能評価とは別の診断である。
        "stability": {
            "enabled": True,
            "methods": ["integrated_gradients"],
            "repeats": 2,
            "noise_fraction": 0.01,
            "clip_nonnegative": True,
            "random_seed": 42,
        },
        # 最終学習層だけをランダム化する簡易的な妥当性確認。
        # 全層を順次ランダム化する検証ではないため、部分的な診断として扱う。
        # 対象となる説明手法と乱数seedを以下で指定する。
        "sanity_check": {
            "enabled": True,
            "methods": ["integrated_gradients"],
            "random_seed": 42,
        },
        # 説明性の出力不足だけを理由に、完了した学習を自動で繰り返さない。
        # 説明性のために再学習したい場合だけTrueに変更する。
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
NOISE_TREND_CONFIG = _cfg("output", "noise_trend_plots")
RUN_INSTANCE_ID = os.environ.get("RUN_ID", datetime.now().strftime("%H%M%S"))
FOLD_PREDICTIONS_DIR_NAME = "fold_pred"
WAV_LEVEL_EVALUATION_ENABLED = _cfg("evaluation", "wav_level_enabled")
WAV_AGGREGATIONS = tuple(_cfg("evaluation", "wav_aggregations"))
PRIMARY_WAV_AGGREGATION = _cfg("evaluation", "primary_wav_aggregation")
ONB_TRANSITION_PERSISTENCE_WAVS = tuple(
    _cfg("evaluation", "onb_transition_persistence_wavs")
)
PREDICTED_EVENT_SUMMARY_ENABLED = _cfg(
    "evaluation", "predicted_event_summary_enabled"
)
EXPLAINABILITY_CONFIG = resolve_explainability_scope(
    VALIDATION_CONFIG.get("explainability", {}),
    experiment_names=EXPERIMENT_DIR_NAMES,
    max_freq_hz_list=MAX_FREQ_HZ_LIST,
    noise_dir_names=NOISE_DIR_NAMES,
    model_keys=ACTIVE_MODEL_KEYS,
    fold_count=DIVISIONS,
)
EXPLAINABILITY_ENABLED = EXPLAINABILITY_CONFIG.get("enabled", False)

# ここから下の定数はVALIDATION_CONFIGから算出する。
# 通常の実験条件変更では、上の設定欄を編集する。
# 実行処理そのものを変更するとき以外は、導出値を直接変更しない。


# モデル名と、実際に呼び出すモデル構築関数の対応表。
# 学習モデルと表示ラベルを一致させるために使用する。
# モデル構造は各構築関数内に固定している。
# 学習率とバッチサイズなどは上のparameter_setsで指定する。


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
        "builder": lambda mm, **params: mm.cnn_transformer_v2(**params),
        "input_axes_assumption": ["time_frame", "frequency_bin", "channel"],
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
        "builder": lambda mm, **params: mm.alexnet(**params),
        "architecture": {
            "input_transform": "log1p(power / 1e-12)",
            "regression_head": "Flatten-Dense4096-Dense4096",
        },
    },
]




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
        "output": {
            "save_date": SAVE_DATE,
            "result_date_dir": RESULT_DATE_DIR,
            "run_instance_id": RUN_INSTANCE_ID,
            "save_fold_predictions": SAVE_FOLD_PREDICTIONS,
            "save_tuning_summary": SAVE_TUNING_SUMMARY,
            "resume_completed_runs": RESUME_COMPLETED_RUNS,
            "noise_trend_plots": dict(NOISE_TREND_CONFIG),
        },
        "evaluation": {
            "wav_level_enabled": WAV_LEVEL_EVALUATION_ENABLED,
            "wav_aggregations": WAV_AGGREGATIONS,
            "primary_wav_aggregation": PRIMARY_WAV_AGGREGATION,
            "onb_transition_persistence_wavs": ONB_TRANSITION_PERSISTENCE_WAVS,
            "predicted_event_summary_enabled": PREDICTED_EVENT_SUMMARY_ENABLED,
            "event_ground_truth_status": "not_annotated",
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
        Path(job["save_base_path"]) / noise_dir / job["max_freq_hz"] / run_dir
        for noise_dir in NOISE_DIR_NAMES
    ]
    output_dir = (
        Path(job["save_base_path"]) / "noise_trends" / job["max_freq_hz"] / run_dir
    )
    artifacts = plotter.plot_noise_trends(
        run_paths,
        output_dir,
        noise_order=NOISE_DIR_NAMES,
        model_keys=model_keys,
        ensemble_strategy_names=NOISE_TREND_CONFIG["ensemble_strategy_names"],
        metrics=NOISE_TREND_CONFIG["metrics"],
        evaluation_units=NOISE_TREND_CONFIG["evaluation_units"],
        wav_aggregation=PRIMARY_WAV_AGGREGATION,
        expected_run_hash=run_hash,
        formats=NOISE_TREND_CONFIG["formats"],
    )
    if artifacts:
        print(f"ノイズ強度別の比較図を更新: {output_dir}（{len(artifacts)}図）")
    return artifacts


def validate_validation_config(enabled_specs):
    if DIVISIONS < 2:
        raise ValueError("folds must be at least 2.")
    if not PARAMETER_SETS:
        raise ValueError("VALIDATION_CONFIG['models']['parameter_sets'] must not be empty.")
    if int(PCA_COMPONENTS) <= 0:
        raise ValueError("pca_components must be a positive integer.")
    if WAV_LEVEL_EVALUATION_ENABLED and COLOR_CHANNEL != 1:
        raise ValueError(
            "WAV-level evaluation currently requires NPY input metadata "
            "(run.color_channel=1)."
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

    allowed_wav_aggregations = {"mean", "median", "p90", "p95"}
    unknown_wav_aggregations = set(WAV_AGGREGATIONS) - allowed_wav_aggregations
    if unknown_wav_aggregations:
        raise ValueError(
            "Unknown evaluation.wav_aggregations: "
            f"{sorted(unknown_wav_aggregations)}"
        )
    if WAV_LEVEL_EVALUATION_ENABLED and not WAV_AGGREGATIONS:
        raise ValueError("evaluation.wav_aggregations must not be empty.")
    if PRIMARY_WAV_AGGREGATION not in WAV_AGGREGATIONS:
        raise ValueError(
            "evaluation.primary_wav_aggregation must be included in "
            "evaluation.wav_aggregations."
        )
    if (
        not ONB_TRANSITION_PERSISTENCE_WAVS
        or any(int(value) <= 0 for value in ONB_TRANSITION_PERSISTENCE_WAVS)
    ):
        raise ValueError(
            "evaluation.onb_transition_persistence_wavs must contain "
            "positive integers."
        )

    if EXPLAINABILITY_CONFIG.get("enabled", False):
        xai_evaluation_unit = str(
            EXPLAINABILITY_CONFIG.get("performance_evaluation_unit", "chunk")
        ).lower()
        xai_wav_aggregation = str(
            EXPLAINABILITY_CONFIG.get("performance_wav_aggregation", "median")
        ).lower()
        if xai_evaluation_unit not in {"source_wav", "chunk"}:
            raise ValueError(
                "explainability.performance_evaluation_unit must be "
                "'source_wav' or 'chunk'."
            )
        if xai_wav_aggregation not in {"mean", "median"}:
            raise ValueError(
                "explainability.performance_wav_aggregation must be "
                "'mean' or 'median'."
            )
        if xai_evaluation_unit == "source_wav" and COLOR_CHANNEL != 1:
            raise ValueError(
                "source-WAV XAI performance evaluation requires NPY metadata "
                "(run.color_channel=1)."
            )

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
    # 指標計算、学習、統合、作図の共通処理を用意する。
    metrics = RegressionDetectionMetrics()
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
        sample_metadata = None
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
                claim_safe_by_model = {key: True for key in model_keys}
                claim_note_by_model = {key: "outer_GroupKFold_OOF" for key in model_keys}
                for strategy in ensemble_run.strategy_plan:
                    result_key = strategy["result_key"]
                    claim_safe_by_model[result_key] = bool(strategy["claim_safe"])
                    claim_note_by_model[result_key] = (
                        "outer_validation_labels_used_for_weights"
                        if not strategy["claim_safe"]
                        else "no_outer_validation_labels_used_for_weights"
                    )
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
                    # 再開時も保存済み指標から図を作れるため、作図だけの再学習は不要。
                    update_noise_trend_plots(plotter, job, run_dir, run_hash, model_keys)
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

                # 指標の保存先: モデル名 → 指標名 → foldごとの値のリスト。
                store = {k: defaultdict(list) for k in all_keys}
                train_meta = {k: defaultdict(list) for k in model_keys}
                oof_prediction_rows = []
                wav_evaluation = None

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
                            groups=(
                                sample_groups[train_index]
                                if sample_groups is not None else None
                            ),
                            pca_components=PCA_COMPONENTS,
                            input_shape=(224, 224, COLOR_CHANNEL),
                            epochs=EPOCH_NUM,
                            fold=fold,
                            total_folds=DIVISIONS,
                        )

                        # 外側foldの最終学習には、学習側の全標本を使用する。
                        # 前処理の学習には、外側検証のデータ・正解値を使わない。
                        scaler = MinMaxScaler()
                        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

                        # RF用のPCAを学習データだけで適合させる。
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

                        # 各モデルの学習と予測
                        val_preds = {}  # モデル名 → 検証foldの予測熱流束（元の単位）
                        for spec in run_specs:
                            # モデルごとにseedを再設定し、実行順による初期値差を抑える。
                            # fold番号を加えることで、fold間では異なるseedを使用する。
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

                            # 検証foldに対する予測
                            val_preds[spec["key"]] = trainer.predict_one_model(
                                spec, model, x_val, x_val_pca, scaler)

                            maybe_explain_trained_model(
                                spec, model, scaler, x_val, y_val,
                                val_preds[spec["key"]], threshold, SAVE_PATH,
                                fold, max_freq_name, EXPLAINABILITY_CONFIG,
                                pca=pca_model,
                                experiment_name=job["experiment_name"],
                                noise_dir_name=noise_dir_name,
                                source_wav_groups=(
                                    sample_groups[val_index]
                                    if sample_groups is not None else None
                                ))

                            ensemble_run.record_validation_error(
                                spec["key"],
                                y_val,
                                val_preds[spec["key"]],
                            )

                            del model
                            del history
                            K.clear_session()
                            gc.collect()

                        # 同じ外側foldの予測から、指定された全統合方式を評価する。
                        # 統合方式を切り替えるたびに単体モデルを再学習する必要はない。
                        # 内部holdout用の学習は、重み決定のため別途実施済み。
                        ensemble_outputs = ensemble_run.combine_predictions(
                            val_preds,
                            inner_errors,
                            fold,
                        )

                        # 回帰・連続スコア検知・二値判定の3種類の指標を計算する。
                        preds_all = ensemble_run.merge_predictions(
                            val_preds,
                            ensemble_outputs,
                        )
                        fold_prediction_rows = build_fold_prediction_rows(
                            val_indices=val_index,
                            y_true=y_val,
                            predictions=preds_all,
                            sample_metadata=sample_metadata,
                            fold=fold,
                        )
                        oof_prediction_rows.extend(fold_prediction_rows)
                        if SAVE_FOLD_PREDICTIONS:
                            pred_dir = os.path.join(SAVE_PATH, FOLD_PREDICTIONS_DIR_NAME)
                            _makedirs(pred_dir)
                            pred_csv = os.path.join(pred_dir, f"pred_f{fold}_{snr_value}.csv")
                            write_fold_prediction_csv(
                                pred_csv,
                                fold_prediction_rows,
                                all_keys,
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

                        # 主アンサンブルの予測と真値を散布図にする。
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

                        # foldごとの指標と重みをテキストへ追記する。
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
                        del fold_prediction_rows
                        K.clear_session()
                        gc.collect()

                    if WAV_LEVEL_EVALUATION_ENABLED:
                        wav_evaluation = save_wav_event_evaluation(
                            save_path=SAVE_PATH,
                            snr_value=snr_value,
                            chunk_rows=oof_prediction_rows,
                            model_keys=all_keys,
                            threshold=threshold,
                            band_frac=ONB_BAND_FRAC,
                            aggregations=WAV_AGGREGATIONS,
                            primary_aggregation=PRIMARY_WAV_AGGREGATION,
                            save_predicted_event_summary=(
                                PREDICTED_EVENT_SUMMARY_ENABLED
                            ),
                            onb_transition_persistence_wavs=(
                                ONB_TRANSITION_PERSISTENCE_WAVS
                            ),
                            claim_safe_by_model=claim_safe_by_model,
                            claim_note_by_model=claim_note_by_model,
                            threshold_provenance=(
                                THRESHOLD_PROVENANCE_BY_EXPERIMENT.get(
                                    job["experiment_name"]
                                )
                            ),
                        )
                        f.write("\nPooled OOF WAV-level Results:\n")
                        f.write(
                            "  unit=source_wav_id | "
                            f"n_wavs={len(wav_evaluation['wav_rows'])} | "
                            f"primary_aggregation={PRIMARY_WAV_AGGREGATION}\n"
                        )
                        for row in wav_evaluation["metric_rows"]:
                            if row["aggregation"] != PRIMARY_WAV_AGGREGATION:
                                continue
                            f.write(
                                f"  [{label_of[row['model_key']]}] "
                                f"R2={row.get('r2', float('nan')):.4f} "
                                f"RMSE={row.get('rmse_all', float('nan')):.1f} "
                                f"MAE={row.get('mae_all', float('nan')):.1f} | "
                                f"Acc={row.get('accuracy', float('nan')):.4f} "
                                f"Rec={row.get('recall', float('nan')):.4f} "
                                f"F1={row.get('f1', float('nan')):.4f}\n"
                            )
                        f.write(
                            "  Event CSV contains predicted threshold crossings "
                            "only; no event precision/recall is claimed.\n"
                        )
                        f.write(
                            "  ONB transition CSV reports heat-flux/operating-point "
                            "offsets; seconds require synchronized annotations.\n"
                        )

                    # chunk指標をfold間で平均し、標準誤差も記録する。
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

                # 単体モデルと主アンサンブルの指標を棒グラフにする。
                # 各図を読みやすくするため、主アンサンブルだけを単体モデルに加える。
                # その他の統合方式は、別の改善量比較グラフにまとめる。
                # ノイズ別の折れ線グラフも、既定ではこの主方式を使用する。
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

                # chunk指標のfold平均と標準誤差をモデル別CSVに保存する。
                csv_path = os.path.join(SAVE_PATH, f'metrics_summary_{snr_value}.csv')
                with _open_text(csv_path, 'w', encoding='utf-8') as cf:
                    header = ["model"] + [f"{mk}_mean" for mk in summary_metrics] \
                                       + [f"{mk}_se" for mk in summary_metrics]
                    cf.write(",".join(header) + "\n")
                    for key in all_keys:
                        means = [f"{metrics.mean_se(store[key][mk])[0]:.6f}" for mk in summary_metrics]
                        ses = [f"{metrics.mean_se(store[key][mk])[1]:.6f}" for mk in summary_metrics]
                        cf.write(",".join([label_of[key]] + means + ses) + "\n")
                print(f"指標CSVを保存: {csv_path}")

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

                # 各条件の指標保存後、完了済みのノイズ条件を含めた曲線を更新する。
                update_noise_trend_plots(plotter, job, run_dir, run_hash, model_keys)

                if not FLG_ROOP:
                    break

        del x, y
        gc.collect()


if __name__ == '__main__':
    main()
